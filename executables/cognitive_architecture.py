"""
M10 - Cognitive Architecture: Sense-Think-Act Loop
Integrates all 10 modules into a unified autonomous agent.

Usage:
    python executables/cognitive_architecture.py
"""

import sys
import os
import time
import json
import numpy as np
import pybullet as p
import pybullet_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src', 'modules'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'robot'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'environment'))

from world_builder import build_world
from sensor_preprocessing import preprocess_all
from perception import (
    depth_to_pointcloud, detect_target, detect_table,
    detect_obstacles, fit_table_plane, estimate_grasp_pose,
    estimate_obstacle_pose
)
from state_estimation import ParticleFilter
from motion_control import (
    DifferentialDriveController, ArmController, plan_path_prolog
)
from action_planning import MissionFSM, State
from knowledge_reasoning import KnowledgeBase
from learning import LearningModule


def get_joint_index(robot_id, joint_name):
    for i in range(p.getNumJoints(robot_id)):
        if p.getJointInfo(robot_id, i)[1].decode() == joint_name:
            return i
    raise ValueError(f"Joint '{joint_name}' not found.")


class OdometryEstimator:
    def __init__(self, wheel_radius=0.08, wheel_base=0.35, dt=1.0/240.0):
        self.r, self.wb, self.dt = wheel_radius, wheel_base, dt

    def update(self, joints):
        lv = joints.get('left_wheel_joint',  {}).get('velocity', 0.0)
        rv = joints.get('right_wheel_joint', {}).get('velocity', 0.0)
        v  = (lv + rv) / 2.0 * self.r
        w  = (rv - lv) / self.wb * self.r
        return v * self.dt, 0.0, w * self.dt


class _PerceptionLogger:
    def __init__(self, interval=5.0):
        self._last, self._interval = 0.0, interval

    def log(self, exc):
        now = time.monotonic()
        if now - self._last >= self._interval:
            print(f"[Perception] {exc}")
            self._last = now


def get_nav_camera_image(robot_id, lidar_link):
    ls  = p.getLinkState(robot_id, lidar_link)
    pos = np.array(ls[0])
    rot = p.getMatrixFromQuaternion(ls[1])
    fwd = np.array([rot[0], rot[3], rot[6]])
    up  = np.array([rot[2], rot[5], rot[8]])
    cam = pos + 0.05 * fwd
    vm  = p.computeViewMatrix(cam.tolist(), (cam + fwd).tolist(), up.tolist())
    pm  = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 10.0)
    _, _, rgb, depth, _ = p.getCameraImage(320, 240, vm, pm)
    return np.array(rgb), np.array(depth, dtype=np.float32), cam, fwd, vm


def cam_to_world(cp, cam_pos, fwd):
    fwd   = np.array(fwd)
    right = np.cross(fwd, [0, 0, 1])
    n     = np.linalg.norm(right)
    right = right / n if n > 1e-6 else np.array([1, 0, 0])
    up    = np.cross(right, fwd)
    return (np.array(cam_pos) + cp[2]*fwd + cp[0]*right + cp[1]*up).tolist()


def approach_goal(rx, ry, tx, ty, stand_off=0.85):
    dx, dy = tx - rx, ty - ry
    d = np.hypot(dx, dy)
    if d < 1e-3:
        return tx, ty
    return tx - dx/d * stand_off, ty - dy/d * stand_off


def wall_obstacles(room_half=3.8, thickness=0.8, n=6):
    walls, seg = [], 2 * room_half / n
    for i in range(n):
        t = -room_half + (i + 0.5) * seg
        walls += [
            {'position': [t,  room_half, 0], 'size': thickness},
            {'position': [t, -room_half, 0], 'size': thickness},
            {'position': [ room_half, t, 0], 'size': thickness},
            {'position': [-room_half, t, 0], 'size': thickness},
        ]
    return walls


def stop_wheels(robot_id, left_joint, right_joint):
    p.setJointMotorControl2(robot_id, left_joint,  p.VELOCITY_CONTROL,
                            targetVelocity=0, force=50.0)
    p.setJointMotorControl2(robot_id, right_joint, p.VELOCITY_CONTROL,
                            targetVelocity=0, force=50.0)


def main():
    print("[CogArch] Starting...")
    world    = build_world(gui=True)
    robot_id = world['robot_id']

    scene_path = os.path.join(ROOT, 'executables', 'scene_map.json')
    if os.path.exists(scene_path):
        with open(scene_path) as f:
            scene_map = json.load(f)
        print("[CogArch] Loaded scene_map.json")
    else:
        scene_map = world['scene_map']

    left_joint  = get_joint_index(robot_id, 'left_wheel_joint')
    right_joint = get_joint_index(robot_id, 'right_wheel_joint')
    arm_joints  = [
        get_joint_index(robot_id, 'arm_base_joint'),
        get_joint_index(robot_id, 'shoulder_joint'),
        get_joint_index(robot_id, 'elbow_joint'),
        get_joint_index(robot_id, 'wrist_joint'),
    ]
    ee_index    = get_joint_index(robot_id, 'end_effector_joint')
    lidar_link  = get_joint_index(robot_id, 'lidar_joint')
    camera_link = get_joint_index(robot_id, 'camera_joint')
    print(f"[CogArch] Joints: L={left_joint} R={right_joint} EE={ee_index}")

    kb = KnowledgeBase()
    kb.populate_from_scene_map(scene_map)

    learner      = LearningModule()
    trial_params = learner.start_trial()
    h_gains      = trial_params['heading_pid']
    d_gains      = trial_params['distance_pid']
    vision_tol   = learner.get_current_params()['vision_tol']

    robot_init = scene_map['robot']
    pf = ParticleFilter(n_particles=200)
    init_pos   = robot_init['position']
    init_orn   = robot_init['orientation']
    init_theta = p.getEulerFromQuaternion(init_orn)[2]
    pf.initialize_at(init_pos[0], init_pos[1], init_theta)

    nav_ctrl = DifferentialDriveController(robot_id, left_joint, right_joint)
    nav_ctrl.heading_pid.update_gains(**h_gains)
    nav_ctrl.distance_pid.update_gains(**d_gains)
    arm_ctrl = ArmController(robot_id, ee_index, arm_joints)

    fsm = MissionFSM()
    fsm.table_position = scene_map['table']['position'][:2]

    odometry    = OdometryEstimator()
    perc_logger = _PerceptionLogger(interval=5.0)

    proj_np = np.array(
        p.computeProjectionMatrixFOV(60, 1.0, 0.1, 10.0), dtype=np.float64
    ).reshape(4, 4).T

    # Obstacle lists
    obs_pf    = [(o['position'][0], o['position'][1]) for o in scene_map['obstacles']]
    table_xy  = scene_map['table']['position'][:2]
    table_obs = [{'position': [table_xy[0], table_xy[1], 0], 'size': 1.6}]
    _walls    = wall_obstacles(room_half=3.8, thickness=0.8, n=6)

    def obs_for_path():
        r  = [{'position': [ox, oy, 0], 'size': 0.55} for ox, oy in obs_pf]
        r += table_obs
        r += _walls
        return r

    # ---- Constants ----
    COLLISION_DIST     = 0.15   # only trigger on very close contact
    COLLISION_COOLDOWN = 240    # 1 second
    WP_ARRIVE          = 0.55
    WP_TIMEOUT         = 240 * 10  # 10 s per waypoint
    STAND_OFF          = 0.85
    PF_EVERY           = 3
    CAM_EVERY          = 10
    LOG_EVERY          = 240 * 5
    SEARCH_TIMEOUT     = 480
    RECOVER_DRIVE_BACK = 120    # 0.5 s backup
    RECOVER_HOLD       = 60     # 0.25 s pause
    RECOVER_TOTAL      = RECOVER_DRIVE_BACK + RECOVER_HOLD

    # ---- State ----
    collision_count     = 0
    search_rotation     = init_theta
    current_waypoints   = []
    waypoint_idx        = 0
    waypoint_steps      = 0
    nav_goal            = None
    grasp_target_pos    = None
    grasp_target_locked = False
    lift_z              = 0.85
    step_count          = 0
    path_planned        = False
    collision_cooldown  = 0
    recover_steps       = 0
    search_steps        = 0

    _nav_rgb = _nav_depth = _nav_cam_pos = _nav_fwd = _nav_view = None
    _w_rgb   = _w_depth   = None

    fsm_state, cmd = fsm.transition('tick')
    print("[CogArch] Loop started.")

    while p.isConnected():  # DO NOT TOUCH
        step_count += 1
        cam_tick = (step_count % CAM_EVERY == 0)
        pf_tick  = (step_count % PF_EVERY  == 0)
        if collision_cooldown > 0:
            collision_cooldown -= 1

        # ===== SENSE =====
        sensors = preprocess_all(robot_id, lidar_link, camera_link,
                                  include_camera=cam_tick)
        lidar   = sensors['lidar']
        joints  = sensors['joints']
        if cam_tick:
            _w_rgb, _w_depth, _ = sensors['camera']

        dx, dy, dtheta = odometry.update(joints)
        pf.predict(dx, dy, dtheta)
        if pf_tick and lidar is not None:
            pf.update(lidar, obs_pf)
            pf.resample()

        # Ground-truth position for control
        gt_pos, gt_orn = p.getBasePositionAndOrientation(robot_id)
        est_x, est_y   = gt_pos[0], gt_pos[1]
        est_theta       = p.getEulerFromQuaternion(gt_orn)[2]

        # ===== THINK =====
        mission_event = 'tick'
        mission_data  = {}

        # Collision: only fire when wheels are actively driving AND very close
        if (min(lidar) < COLLISION_DIST
                and collision_cooldown == 0
                and fsm.state == State.NAVIGATE):
            collision_count   += 1
            collision_cooldown = COLLISION_COOLDOWN
            mission_event      = 'collision'

        if fsm.state == State.RECOVER:
            recover_steps += 1
        else:
            recover_steps = 0

        # Camera update
        if cam_tick:
            try:
                _nav_rgb, _nav_depth, _nav_cam_pos, _nav_fwd, _nav_view = \
                    get_nav_camera_image(robot_id, lidar_link)
            except Exception as e:
                perc_logger.log(e)

        # Perception: SEARCH / NAVIGATE
        if (_nav_depth is not None
                and mission_event == 'tick'
                and fsm.state in (State.SEARCH, State.NAVIGATE)):
            try:
                pts, cols = depth_to_pointcloud(
                    _nav_depth, _nav_rgb, proj_np, _nav_view)
                if len(pts) > 10:
                    tgt_pts   = detect_target(pts, cols, tol=vision_tol)
                    table_pts = detect_table(pts, cols, tol=vision_tol)

                    if len(tgt_pts) > 20 and not grasp_target_locked:
                        dist_to_goal = (
                            np.hypot(est_x - nav_goal[0], est_y - nav_goal[1])
                            if nav_goal else 999
                        )
                        gp, _ = estimate_grasp_pose(tgt_pts)
                        if gp is not None:
                            w    = cam_to_world(gp, _nav_cam_pos, _nav_fwd)
                            w[2] = 0.685
                            kb.assert_target_estimate(*w)
                            grasp_target_pos = w
                            if dist_to_goal < 1.5:
                                grasp_target_locked = True
                                print(f"[CogArch] Target LOCKED {w[:2]}")
                            nav_goal      = w[:2]
                            mission_event = 'target_found'
                            mission_data  = {'target_pos': w[:2]}
                            path_planned  = False

                    elif (len(table_pts) > 50
                          and fsm.state == State.SEARCH
                          and not grasp_target_locked):
                        _, tc = fit_table_plane(pts, cols, tol=vision_tol)
                        if tc is not None:
                            wt    = cam_to_world(tc, _nav_cam_pos, _nav_fwd)
                            nav_goal      = wt[:2]
                            mission_event = 'table_found'
                            mission_data  = {'table_pos': wt[:2]}
                            path_planned  = False
                            print(f"[CogArch] Table spotted {wt[:2]}")
            except Exception as e:
                perc_logger.log(e)

        # Wrist camera: ALIGN / GRASP
        if (fsm.state in (State.ALIGN, State.GRASP)
                and cam_tick and _w_depth is not None):
            try:
                cs    = p.getLinkState(robot_id, camera_link)
                cp, co = cs[0], cs[1]
                rot   = p.getMatrixFromQuaternion(co)
                fwd   = [rot[0], rot[3], rot[6]]
                up    = [rot[2], rot[5], rot[8]]
                wv    = p.computeViewMatrix(
                    cp,
                    [cp[0]+fwd[0], cp[1]+fwd[1], cp[2]+fwd[2]],
                    up
                )
                wp, wc = depth_to_pointcloud(_w_depth, _w_rgb, proj_np, wv)
                if len(wp) > 10:
                    ct = detect_target(wp, wc, tol=vision_tol)
                    if len(ct) > 10:
                        gp, _ = estimate_grasp_pose(ct)
                        if gp is not None:
                            wg    = cam_to_world(gp, np.array(cp), np.array(fwd))
                            wg[2] = 0.685
                            grasp_target_pos = wg
                            kb.assert_target_estimate(*wg)
            except Exception as e:
                perc_logger.log(e)

        # SEARCH timeout -> KB table fallback
        if fsm.state == State.SEARCH and mission_event == 'tick':
            search_steps += 1
            if search_steps >= SEARCH_TIMEOUT:
                tp = kb.get_table_position()
                if tp:
                    nav_goal      = list(tp)
                    mission_event = 'table_found'
                    mission_data  = {'table_pos': nav_goal}
                    path_planned  = False
                    search_steps  = 0
                    print(f"[CogArch] Search timeout -> KB table {tp}")
        else:
            search_steps = 0

        # Waypoint arrival
        if (fsm.state == State.NAVIGATE
                and current_waypoints
                and waypoint_idx < len(current_waypoints)
                and mission_event == 'tick'):
            wx, wy     = current_waypoints[waypoint_idx]
            dist_wp    = np.hypot(est_x - wx, est_y - wy)
            waypoint_steps += 1
            timed_out  = waypoint_steps >= WP_TIMEOUT
            if timed_out:
                print(f"[CogArch] WP{waypoint_idx} timeout dist={dist_wp:.2f}m -> advance")
            if dist_wp < WP_ARRIVE or timed_out:
                waypoint_steps = 0
                waypoint_idx  += 1
                if waypoint_idx >= len(current_waypoints):
                    mission_event = 'at_table'
                    print("[CogArch] All WPs reached -> ALIGN")
                else:
                    mission_event = 'waypoint_reached'

        # ALIGN: fire arm_aligned when close enough to table
        if fsm.state == State.ALIGN and mission_event == 'tick':
            tp = kb.get_table_position()
            if tp and np.hypot(est_x - tp[0], est_y - tp[1]) < 1.2:
                if grasp_target_pos is None:
                    grasp_target_pos = [tp[0], tp[1], 0.685]
                    print("[CogArch] Using table centre as grasp pos")
                mission_event = 'arm_aligned'

        # GRASP success check
        if (fsm.state == State.GRASP
                and grasp_target_pos
                and mission_event == 'tick'):
            ee = np.array(p.getLinkState(robot_id, ee_index)[0])
            if np.linalg.norm(ee - np.array(grasp_target_pos)) < 0.08:
                mission_event = 'grasp_success'

        # LIFT success check
        if fsm.state == State.LIFT and mission_event == 'tick':
            if p.getLinkState(robot_id, ee_index)[0][2] > lift_z - 0.05:
                mission_event = 'lift_success'

        # RECOVER timeout -> back to NAVIGATE with fresh plan
        if fsm.state == State.RECOVER and recover_steps >= RECOVER_TOTAL:
            fsm.state       = State.NAVIGATE
            fsm._state_steps = 0
            recover_steps   = 0
            collision_cooldown = COLLISION_COOLDOWN
            if nav_goal is not None:
                gx, gy = approach_goal(
                    est_x, est_y, nav_goal[0], nav_goal[1], STAND_OFF
                )
                path = plan_path_prolog(
                    est_x, est_y, gx, gy, obs_for_path(), kb
                )
                if len(path) > 1:
                    path = path[1:]
                fsm.set_waypoints(path)
                current_waypoints = path
                waypoint_idx      = 0
                waypoint_steps    = 0
                path_planned      = True
                print(f"[CogArch] Post-recover re-plan: {len(path)} wps")
            mission_event = 'tick'  # keep FSM in NAVIGATE, skip regular transition
            # Manually set action
            fsm_state = fsm.state
            cmd = {'cmd': 'continue_nav'}

        # Log every 5 s
        if step_count % LOG_EVERY == 0:
            wp_d = (
                np.hypot(est_x - current_waypoints[waypoint_idx][0],
                         est_y - current_waypoints[waypoint_idx][1])
                if current_waypoints and waypoint_idx < len(current_waypoints)
                else -1
            )
            print(f"[CogArch] t={step_count//240}s | {fsm.state} "
                  f"pos=({est_x:.2f},{est_y:.2f}) "
                  f"th={np.degrees(est_theta):.0f}deg "
                  f"wp={waypoint_idx}/{len(current_waypoints)} "
                  f"dist={wp_d:.2f}m "
                  f"lidar={min(lidar):.2f} "
                  f"locked={grasp_target_locked}")

        # ===== DRIVE FSM (only if not already handled above) =====
        if mission_event != 'tick' or fsm.state not in (State.NAVIGATE,):
            fsm_state, cmd = fsm.transition(mission_event, mission_data)
        else:
            # In NAVIGATE with no special event: just keep driving
            fsm_state = fsm.state
            cmd = {'cmd': 'continue_nav'}

        action = cmd.get('cmd', 'idle')

        # ===== ACT =====
        if action == 'rotate_search':
            search_rotation += 0.005
            nav_ctrl.rotate_to(search_rotation, est_theta)

        elif action == 'plan_path' and not path_planned:
            goal = nav_goal or fsm.table_position
            if goal is not None:
                gx, gy = approach_goal(
                    est_x, est_y, goal[0], goal[1], STAND_OFF
                )
                path = plan_path_prolog(
                    est_x, est_y, gx, gy, obs_for_path(), kb
                )
                if len(path) > 1:
                    path = path[1:]
                print(f"[CogArch] Path: {len(path)} wps -> {goal}")
                fsm.set_waypoints(path)
                current_waypoints = path
                waypoint_idx   = 0
                waypoint_steps = 0
                path_planned   = True
                if path:
                    nav_ctrl.drive_to(
                        path[0][0], path[0][1], est_x, est_y, est_theta
                    )

        elif action in ('continue_nav', 'next_waypoint'):
            if current_waypoints and waypoint_idx < len(current_waypoints):
                wx, wy = current_waypoints[waypoint_idx]
                nav_ctrl.drive_to(wx, wy, est_x, est_y, est_theta)

        elif action == 'backup':
            p.setJointMotorControl2(robot_id, left_joint,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=-2.0, force=10.0)
            p.setJointMotorControl2(robot_id, right_joint,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=-2.0, force=10.0)

        elif action == 'stop_and_align':
            stop_wheels(robot_id, left_joint, right_joint)
            if grasp_target_pos is None:
                tp = kb.get_table_position()
                if tp:
                    grasp_target_pos = [tp[0], tp[1], 0.685]
            if grasp_target_pos:
                arm_ctrl.move_to_position([
                    grasp_target_pos[0],
                    grasp_target_pos[1],
                    grasp_target_pos[2] + 0.15
                ])

        elif action == 'align_arm':
            stop_wheels(robot_id, left_joint, right_joint)
            if grasp_target_pos is None:
                tp = kb.get_table_position()
                if tp:
                    grasp_target_pos = [tp[0], tp[1], 0.685]
            if grasp_target_pos:
                arm_ctrl.move_to_position([
                    grasp_target_pos[0],
                    grasp_target_pos[1],
                    grasp_target_pos[2] + 0.15
                ])

        elif action == 'execute_grasp':
            stop_wheels(robot_id, left_joint, right_joint)
            if grasp_target_pos is None:
                tp = kb.get_table_position()
                if tp:
                    grasp_target_pos = [tp[0], tp[1], 0.685]
            if grasp_target_pos:
                arm_ctrl.move_to_position(grasp_target_pos)

        elif action == 'lift_object':
            stop_wheels(robot_id, left_joint, right_joint)
            if grasp_target_pos:
                done = arm_ctrl.move_to_position([
                    grasp_target_pos[0],
                    grasp_target_pos[1],
                    lift_z
                ])
                if done:
                    mission_event = 'lift_success'

        elif action in ('stop', 'idle', 'hold'):
            stop_wheels(robot_id, left_joint, right_joint)

        # During RECOVER first half: drive backward; second half: hold still
        if fsm.state == State.RECOVER:
            if recover_steps <= RECOVER_DRIVE_BACK:
                p.setJointMotorControl2(robot_id, left_joint,
                                        p.VELOCITY_CONTROL,
                                        targetVelocity=-1.5, force=10.0)
                p.setJointMotorControl2(robot_id, right_joint,
                                        p.VELOCITY_CONTROL,
                                        targetVelocity=-1.5, force=10.0)
            else:
                stop_wheels(robot_id, left_joint, right_joint)

        if fsm.is_terminal():
            tp = kb.get_table_position() or [0, 0]
            learner.end_trial(
                success=(fsm.state == State.DONE),
                distance_remaining=np.hypot(
                    est_x - tp[0], est_y - tp[1]
                ),
                collision_count=collision_count
            )
            print(f"[CogArch] Mission ended: {fsm.state}")
            break

        p.stepSimulation()   # DO NOT TOUCH
        time.sleep(1./240.)  # DO NOT TOUCH

    print("[CogArch] Done.")


if __name__ == '__main__':
    main()
