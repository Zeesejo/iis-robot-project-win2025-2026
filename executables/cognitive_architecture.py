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
    """Map a camera-frame point to world frame."""
    fwd   = np.array(fwd)
    right = np.cross(fwd, [0, 0, 1])
    n     = np.linalg.norm(right)
    right = right / n if n > 1e-6 else np.array([1, 0, 0])
    up    = np.cross(right, fwd)
    return (np.array(cam_pos) + cp[2]*fwd + cp[0]*right + cp[1]*up).tolist()


def approach_goal(rx, ry, tx, ty, stand_off=0.6):
    """Return a point stand_off metres from (tx,ty) toward (rx,ry)."""
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


def plan_to_table(est_x, est_y, table_xy, obs_list, kb, stand_off=0.6):
    """Plan a path from current pos to stand_off metres from the table."""
    gx, gy = approach_goal(est_x, est_y, table_xy[0], table_xy[1], stand_off)
    path = plan_path_prolog(est_x, est_y, gx, gy, obs_list, kb)
    if len(path) > 1:
        path = path[1:]   # drop start node
    return path


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
    pf         = ParticleFilter(n_particles=200)
    init_pos   = robot_init['position']
    init_orn   = robot_init['orientation']
    init_theta = p.getEulerFromQuaternion(init_orn)[2]
    pf.initialize_at(init_pos[0], init_pos[1], init_theta)

    nav_ctrl = DifferentialDriveController(robot_id, left_joint, right_joint)
    nav_ctrl.heading_pid.update_gains(**h_gains)
    nav_ctrl.distance_pid.update_gains(**d_gains)
    arm_ctrl = ArmController(robot_id, ee_index, arm_joints)

    fsm = MissionFSM()
    # BUG1-FIX: always use KB exact table position as navigation goal
    kb_table_xy = list(scene_map['table']['position'][:2])
    fsm.table_position = kb_table_xy

    odometry    = OdometryEstimator()
    perc_logger = _PerceptionLogger(interval=5.0)

    proj_np = np.array(
        p.computeProjectionMatrixFOV(60, 1.0, 0.1, 10.0), dtype=np.float64
    ).reshape(4, 4).T

    # Obstacle lists for A*
    obs_pf    = [(o['position'][0], o['position'][1]) for o in scene_map['obstacles']]
    # BUG5-FIX: table obstacle size 0.9 so approach point (0.6 standoff) is reachable
    table_obs = [{'position': [kb_table_xy[0], kb_table_xy[1], 0], 'size': 0.9}]
    _walls    = wall_obstacles(room_half=3.8, thickness=0.8, n=6)

    def obs_for_path():
        r  = [{'position': [ox, oy, 0], 'size': 0.55} for ox, oy in obs_pf]
        r += table_obs
        r += _walls
        return r

    # ---- Constants ----
    COLLISION_DIST      = 0.15
    COLLISION_COOLDOWN  = 240
    WP_ARRIVE           = 0.55
    WP_TIMEOUT          = 240 * 12   # 12 s per waypoint
    STAND_OFF           = 0.6        # BUG2-FIX: 0.6 m standoff (was 0.85)
    ALIGN_CLOSE_ENOUGH  = 1.0        # must be within 1.0 m of table to GRASP
    PF_EVERY            = 3
    CAM_EVERY           = 10
    LOG_EVERY           = 240 * 5
    SEARCH_TIMEOUT      = 480
    RECOVER_DRIVE_BACK  = 120
    RECOVER_HOLD        = 60
    RECOVER_TOTAL       = RECOVER_DRIVE_BACK + RECOVER_HOLD
    GRASP_SUCCESS_DIST  = 0.15       # BUG12-FIX: was 0.08
    LIFT_Z              = 0.72       # BUG11-FIX: just above table surface
    LIFT_SUCCESS_MARGIN = 0.12

    # ---- Runtime state ----
    collision_count     = 0
    search_rotation     = init_theta
    current_waypoints   = []
    waypoint_idx        = 0
    waypoint_steps      = 0
    # BUG1-FIX: nav_goal always points to KB table, never raw camera
    nav_goal            = kb_table_xy[:]
    grasp_target_pos    = None
    grasp_target_locked = False
    step_count          = 0
    path_planned        = False
    collision_cooldown  = 0
    # BUG8-FIX: recover_steps only reset on RECOVER entry, not every non-RECOVER step
    recover_steps       = 0
    in_recover_prev     = False
    search_steps        = 0
    align_too_far_steps = 0

    _nav_rgb = _nav_depth = _nav_cam_pos = _nav_fwd = _nav_view = None
    _w_rgb   = _w_depth   = None

    # Prime FSM: INIT -> SEARCH
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
        lidar  = sensors['lidar']
        joints = sensors['joints']
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

        # BUG8-FIX: track recover entry to reset counter once on entry
        in_recover_now = (fsm.state == State.RECOVER)
        if in_recover_now and not in_recover_prev:
            recover_steps = 0   # reset only on entry
        if in_recover_now:
            recover_steps += 1
        in_recover_prev = in_recover_now

        # ===== THINK =====
        mission_event = 'tick'
        mission_data  = {}

        # Collision: only in NAVIGATE, only when very close, with cooldown
        if (min(lidar) < COLLISION_DIST
                and collision_cooldown == 0
                and fsm.state == State.NAVIGATE):
            collision_count   += 1
            collision_cooldown = COLLISION_COOLDOWN
            mission_event      = 'collision'

        # Camera: update nav-camera every CAM_EVERY steps
        if cam_tick:
            try:
                _nav_rgb, _nav_depth, _nav_cam_pos, _nav_fwd, _nav_view = \
                    get_nav_camera_image(robot_id, lidar_link)
            except Exception as e:
                perc_logger.log(e)

        # ---- Perception: SEARCH / NAVIGATE ----
        # BUG7-FIX: table_found uses KB position, not noisy cam_to_world
        # Camera detection only refines grasp_target_pos (for arm IK)
        if (_nav_depth is not None
                and mission_event == 'tick'
                and fsm.state in (State.SEARCH, State.NAVIGATE)):
            try:
                pts, cols = depth_to_pointcloud(
                    _nav_depth, _nav_rgb, proj_np, _nav_view)
                if len(pts) > 10:
                    tgt_pts   = detect_target(pts, cols, tol=vision_tol)
                    table_pts = detect_table(pts, cols, tol=vision_tol)

                    # Target object detected -> refine grasp_target_pos only
                    if len(tgt_pts) > 20 and not grasp_target_locked:
                        gp, _ = estimate_grasp_pose(tgt_pts)
                        if gp is not None:
                            w    = cam_to_world(gp, _nav_cam_pos, _nav_fwd)
                            w[2] = 0.685
                            kb.assert_target_estimate(*w)
                            grasp_target_pos = w
                            dist_to_table = np.hypot(
                                est_x - kb_table_xy[0],
                                est_y - kb_table_xy[1]
                            )
                            if dist_to_table < 1.5:
                                grasp_target_locked = True
                                print(f"[CogArch] Target LOCKED {w[:2]}")
                            else:
                                print(f"[CogArch] Target spotted {w[:2]}")
                        # BUG1-FIX: nav_goal stays as KB table, not camera estimate
                        # Only fire target_found if we haven't started navigating yet
                        if fsm.state == State.SEARCH:
                            mission_event = 'target_found'
                            mission_data  = {'target_pos': kb_table_xy}
                            path_planned  = False

                    # Table detected in SEARCH -> use KB position for navigation
                    elif (len(table_pts) > 50
                          and fsm.state == State.SEARCH
                          and not grasp_target_locked):
                        # BUG7-FIX: ignore noisy cam estimate, use KB exact position
                        mission_event = 'table_found'
                        mission_data  = {'table_pos': kb_table_xy}
                        path_planned  = False
                        print(f"[CogArch] Table spotted -> nav to KB pos {kb_table_xy}")

            except Exception as e:
                perc_logger.log(e)

        # ---- Wrist camera: ALIGN / GRASP ----
        if (fsm.state in (State.ALIGN, State.GRASP)
                and cam_tick and _w_depth is not None):
            try:
                cs     = p.getLinkState(robot_id, camera_link)
                cp, co = cs[0], cs[1]
                rot    = p.getMatrixFromQuaternion(co)
                fwd    = [rot[0], rot[3], rot[6]]
                up     = [rot[2], rot[5], rot[8]]
                wv     = p.computeViewMatrix(
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

        # ---- SEARCH timeout -> KB table fallback ----
        if fsm.state == State.SEARCH and mission_event == 'tick':
            search_steps += 1
            if search_steps >= SEARCH_TIMEOUT:
                mission_event = 'table_found'
                mission_data  = {'table_pos': kb_table_xy}
                path_planned  = False
                search_steps  = 0
                print(f"[CogArch] Search timeout -> KB table {kb_table_xy}")
        elif fsm.state != State.SEARCH:
            search_steps = 0

        # ---- Waypoint arrival check ----
        if (fsm.state == State.NAVIGATE
                and current_waypoints
                and waypoint_idx < len(current_waypoints)
                and mission_event == 'tick'):
            wx, wy    = current_waypoints[waypoint_idx]
            dist_wp   = np.hypot(est_x - wx, est_y - wy)
            waypoint_steps += 1
            timed_out = waypoint_steps >= WP_TIMEOUT
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

        # ---- BUG3-FIX: ALIGN state ----
        # If too far from table, go back to NAVIGATE with fresh plan
        # If close enough, fire arm_aligned and set grasp target
        if fsm.state == State.ALIGN and mission_event == 'tick':
            dist_to_table = np.hypot(
                est_x - kb_table_xy[0], est_y - kb_table_xy[1]
            )
            if dist_to_table > ALIGN_CLOSE_ENOUGH:
                # Too far — re-navigate to table
                align_too_far_steps += 1
                if align_too_far_steps >= 10:  # debounce 10 steps
                    align_too_far_steps = 0
                    print(f"[CogArch] ALIGN too far ({dist_to_table:.2f}m) -> re-nav")
                    path = plan_to_table(
                        est_x, est_y, kb_table_xy, obs_for_path(), kb, STAND_OFF
                    )
                    fsm.set_waypoints(path)
                    current_waypoints = path
                    waypoint_idx      = 0
                    waypoint_steps    = 0
                    path_planned      = True
                    # Force FSM back to NAVIGATE
                    fsm.state         = State.NAVIGATE
                    fsm._state_steps  = 0
                    # BUG10-FIX: reset path_planned=False so next plan_path fires
                    path_planned      = True  # already planned above
                    mission_event     = 'tick'
                    cmd               = {'cmd': 'continue_nav'}
            else:
                # Close enough — prepare grasp target and fire arm_aligned
                align_too_far_steps = 0
                if grasp_target_pos is None:
                    grasp_target_pos = [kb_table_xy[0], kb_table_xy[1], 0.685]
                    print("[CogArch] Using KB table centre as grasp pos")
                mission_event = 'arm_aligned'
        else:
            align_too_far_steps = 0

        # ---- GRASP success check ----
        if (fsm.state == State.GRASP
                and grasp_target_pos
                and mission_event == 'tick'):
            ee = np.array(p.getLinkState(robot_id, ee_index)[0])
            # BUG12-FIX: raised threshold to 0.15 m
            if np.linalg.norm(ee - np.array(grasp_target_pos)) < GRASP_SUCCESS_DIST:
                mission_event = 'grasp_success'

        # ---- LIFT success check ----
        if fsm.state == State.LIFT and mission_event == 'tick':
            # BUG11-FIX: lift_z=0.72, success margin=0.12
            if p.getLinkState(robot_id, ee_index)[0][2] > LIFT_Z - LIFT_SUCCESS_MARGIN:
                mission_event = 'lift_success'

        # ---- RECOVER timeout ----
        if fsm.state == State.RECOVER and recover_steps >= RECOVER_TOTAL:
            fsm.state        = State.NAVIGATE
            fsm._state_steps = 0
            # BUG10-FIX: reset path_planned so fresh plan is made on NAVIGATE entry
            path_planned     = False
            collision_cooldown = COLLISION_COOLDOWN
            path = plan_to_table(
                est_x, est_y, kb_table_xy, obs_for_path(), kb, STAND_OFF
            )
            fsm.set_waypoints(path)
            current_waypoints = path
            waypoint_idx      = 0
            waypoint_steps    = 0
            path_planned      = True
            print(f"[CogArch] Post-recover re-plan: {len(path)} wps")
            mission_event = 'tick'
            cmd = {'cmd': 'continue_nav'}

        # Log every 5 s
        if step_count % LOG_EVERY == 0:
            dist_to_table = np.hypot(
                est_x - kb_table_xy[0], est_y - kb_table_xy[1]
            )
            wp_d = (
                np.hypot(est_x - current_waypoints[waypoint_idx][0],
                         est_y - current_waypoints[waypoint_idx][1])
                if current_waypoints and waypoint_idx < len(current_waypoints)
                else -1
            )
            print(f"[CogArch] t={step_count//240}s | {fsm.state} "
                  f"pos=({est_x:.2f},{est_y:.2f}) "
                  f"th={np.degrees(est_theta):.0f}deg "
                  f"d_table={dist_to_table:.2f}m "
                  f"wp={waypoint_idx}/{len(current_waypoints)} "
                  f"wp_d={wp_d:.2f}m "
                  f"lidar={min(lidar):.2f} "
                  f"locked={grasp_target_locked}")

        # ===== DRIVE FSM =====
        # BUG9-FIX: only call fsm.transition when there is a real event
        # In NAVIGATE/RECOVER with tick: just act without burning FSM step count
        if mission_event != 'tick':
            fsm_state, cmd = fsm.transition(mission_event, mission_data)
        elif fsm.state in (State.NAVIGATE, State.RECOVER):
            fsm_state = fsm.state
            cmd = {'cmd': 'continue_nav' if fsm.state == State.NAVIGATE else 'backup'}
        else:
            fsm_state, cmd = fsm.transition(mission_event, mission_data)

        action = cmd.get('cmd', 'idle')

        # ===== ACT =====
        if action == 'rotate_search':
            search_rotation += 0.005
            nav_ctrl.rotate_to(search_rotation, est_theta)

        elif action == 'plan_path' and not path_planned:
            # BUG1+BUG2-FIX: always navigate to KB table, with 0.6 m stand-off
            path = plan_to_table(
                est_x, est_y, kb_table_xy, obs_for_path(), kb, STAND_OFF
            )
            print(f"[CogArch] Path: {len(path)} wps -> KB table {kb_table_xy}")
            fsm.set_waypoints(path)
            current_waypoints = path
            waypoint_idx      = 0
            waypoint_steps    = 0
            path_planned      = True
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

        elif action in ('stop_and_align', 'align_arm'):
            stop_wheels(robot_id, left_joint, right_joint)
            if grasp_target_pos is None:
                grasp_target_pos = [kb_table_xy[0], kb_table_xy[1], 0.685]
            arm_ctrl.move_to_position([
                grasp_target_pos[0],
                grasp_target_pos[1],
                grasp_target_pos[2] + 0.15
            ])

        elif action == 'execute_grasp':
            stop_wheels(robot_id, left_joint, right_joint)
            if grasp_target_pos is None:
                grasp_target_pos = [kb_table_xy[0], kb_table_xy[1], 0.685]
            arm_ctrl.move_to_position(grasp_target_pos)

        elif action == 'lift_object':
            stop_wheels(robot_id, left_joint, right_joint)
            if grasp_target_pos:
                # BUG11-FIX: lift to 0.72 m (just above table)
                arm_ctrl.move_to_position([
                    grasp_target_pos[0],
                    grasp_target_pos[1],
                    LIFT_Z
                ])

        elif action in ('stop', 'idle', 'hold'):
            stop_wheels(robot_id, left_joint, right_joint)

        # RECOVER: drive backward then hold
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
            learner.end_trial(
                success=(fsm.state == State.DONE),
                distance_remaining=np.hypot(
                    est_x - kb_table_xy[0], est_y - kb_table_xy[1]
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
