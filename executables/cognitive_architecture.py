"""
M10 - Cognitive Architecture: Sense-Think-Act Loop
Integrates all 10 modules into a unified autonomous agent.
Entry point for the IIS robot project.

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

# ---- Path setup ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src', 'modules'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'robot'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'environment'))

# ---- Module imports ----
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


# ===================== JOINT INDEX HELPERS =====================

def get_joint_index(robot_id, joint_name):
    n = p.getNumJoints(robot_id)
    for i in range(n):
        info = p.getJointInfo(robot_id, i)
        if info[1].decode('utf-8') == joint_name:
            return i
    raise ValueError(f"Joint '{joint_name}' not found.")


def get_link_index(robot_id, link_name):
    """Return joint index whose child link name matches link_name.
    In PyBullet the joint index == the child link index.
    info[12] is the joint frame link name; info[1] is the joint name.
    For URDF links, the child link name equals the joint name stripped
    of '_joint' -> we match against the child link name stored in info[12].
    Falls back to -1 (base) if not found.
    """
    n = p.getNumJoints(robot_id)
    for i in range(n):
        info = p.getJointInfo(robot_id, i)
        # info[12] = link name of the child link
        if info[12].decode('utf-8') == link_name:
            return i
    return -1


# ===================== ODOMETRY ESTIMATE =====================

class OdometryEstimator:
    """Estimate robot delta motion from wheel joint velocities."""

    def __init__(self, wheel_radius=0.08, wheel_base=0.35, dt=1.0 / 240.0):
        self.r = wheel_radius
        self.wb = wheel_base
        self.dt = dt

    def update(self, joints):
        """Returns (dx, dy, dtheta) in robot frame."""
        left_vel  = joints.get('left_wheel_joint',  {}).get('velocity', 0.0)
        right_vel = joints.get('right_wheel_joint', {}).get('velocity', 0.0)
        v_left  = left_vel  * self.r
        v_right = right_vel * self.r
        v     = (v_left + v_right) / 2.0
        omega = (v_right - v_left) / self.wb
        dx     = v * self.dt
        dy     = 0.0
        dtheta = omega * self.dt
        return dx, dy, dtheta


# ===================== PERCEPTION ERROR LOGGER =====================

class _PerceptionLogger:
    """Rate-limited logger so perception errors don't spam stdout."""
    def __init__(self, interval=5.0):
        self._last = 0.0
        self._interval = interval

    def log(self, exc):
        now = time.monotonic()
        if now - self._last >= self._interval:
            print(f"[Perception] Error (suppressed at {self._interval}s rate): {exc}")
            self._last = now


# ===================== FORWARD-LOOKING CAMERA HELPER =====================

def get_nav_camera_image(robot_id):
    """
    Render a forward-looking camera from the robot base (not the wrist camera).
    The wrist camera (camera_joint) is angled down for grasping.
    This nav camera looks straight ahead from ~0.3 m height on the base.
    Returns (rgb, depth, mask) matching sensor_wrapper format.
    """
    base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
    rot = p.getMatrixFromQuaternion(base_orn)
    # Forward vector in world frame
    forward = [rot[0], rot[3], rot[6]]
    up      = [rot[2], rot[5], rot[8]]
    # Camera sits 0.3 m above base origin, 0.2 m forward
    cam_pos = [
        base_pos[0] + 0.2 * forward[0],
        base_pos[1] + 0.2 * forward[1],
        base_pos[2] + 0.30
    ]
    target = [
        cam_pos[0] + forward[0],
        cam_pos[1] + forward[1],
        cam_pos[2]
    ]
    view_matrix = p.computeViewMatrix(cam_pos, target, up)
    proj_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 10.0)
    w, h, rgb, depth, mask = p.getCameraImage(320, 240, view_matrix, proj_matrix)
    return rgb, np.array(depth, dtype=np.float32), mask, view_matrix


# ===================== MAIN COGNITIVE LOOP =====================

def main():
    print("[CogArch] Starting IIS Robot Cognitive Architecture...")

    # ---- M2: Build World ----
    world = build_world(gui=True)
    robot_id = world['robot_id']

    # ---- Load scene map ----
    scene_path = os.path.join(ROOT, 'executables', 'scene_map.json')
    if os.path.exists(scene_path):
        with open(scene_path) as f:
            scene_map = json.load(f)
        print("[CogArch] Loaded scene_map.json from disk.")
    else:
        scene_map = world['scene_map']
        print("[CogArch] scene_map.json not found — using build_world() scene_map.")

    # ---- Discover joint indices ----
    # In PyBullet, joint_index == child_link_index.
    # We look up by joint name (reliable) not child link name.
    left_joint  = get_joint_index(robot_id, 'left_wheel_joint')
    right_joint = get_joint_index(robot_id, 'right_wheel_joint')
    arm_joints  = [
        get_joint_index(robot_id, 'arm_base_joint'),
        get_joint_index(robot_id, 'shoulder_joint'),
        get_joint_index(robot_id, 'elbow_joint'),
        get_joint_index(robot_id, 'wrist_joint'),
    ]
    # end_effector_joint is fixed -> its child link is the EE
    ee_index     = get_joint_index(robot_id, 'end_effector_joint')
    # Sensors: use joint index (== child link index)
    lidar_link   = get_joint_index(robot_id, 'lidar_joint')
    camera_link  = get_joint_index(robot_id, 'camera_joint')  # wrist cam (grasping)

    print(f"[CogArch] Joint indices: left={left_joint} right={right_joint} "
          f"lidar={lidar_link} cam={camera_link} ee={ee_index}")

    # ---- M8: Knowledge Base ----
    kb = KnowledgeBase()
    kb.populate_from_scene_map(scene_map)

    # ---- M9: Learning ----
    learner = LearningModule()
    trial_params = learner.start_trial()
    h_gains   = trial_params['heading_pid']
    d_gains   = trial_params['distance_pid']
    vision_tol = learner.get_current_params()['vision_tol']

    # ---- M5: Particle Filter ----
    robot_init = scene_map['robot']
    pf = ParticleFilter(n_particles=300)
    init_pos  = robot_init['position']
    init_orn  = robot_init['orientation']
    init_theta = p.getEulerFromQuaternion(init_orn)[2]
    pf.initialize_at(init_pos[0], init_pos[1], init_theta)

    # ---- M6: Controllers ----
    nav_ctrl = DifferentialDriveController(robot_id, left_joint, right_joint)
    nav_ctrl.heading_pid.update_gains(**h_gains)
    nav_ctrl.distance_pid.update_gains(**d_gains)
    arm_ctrl = ArmController(robot_id, ee_index, arm_joints)

    # ---- M7: FSM ----
    fsm = MissionFSM()
    fsm.table_position = scene_map['table']['position'][:2]

    # ---- Odometry ----
    odometry = OdometryEstimator()

    # ---- Perception logger ----
    perc_logger = _PerceptionLogger(interval=5.0)

    # ---- State variables ----
    state_est = (init_pos[0], init_pos[1], init_theta)
    obstacle_map_for_pf = [
        (obs['position'][0], obs['position'][1])
        for obs in scene_map['obstacles']
    ]
    collision_count   = 0
    search_rotation   = init_theta
    current_waypoints = []
    waypoint_idx      = 0
    grasp_target_pos  = None
    lift_z            = 0.9
    mission_event     = 'tick'
    mission_data      = {}

    # Search timeout: if no table/target found visually after N steps,
    # use the KB table position from scene_map directly
    search_step_count = 0
    SEARCH_TIMEOUT_STEPS = 480   # 2 seconds at 240 Hz

    # Initial FSM tick
    fsm_state, cmd = fsm.transition('tick')

    print("[CogArch] Entering Sense-Think-Act loop...")

    # ===================== MAIN LOOP =====================
    while p.isConnected():  # DO NOT TOUCH

        # ======= SENSE =======
        sensors = preprocess_all(robot_id, lidar_link, camera_link)
        lidar  = sensors['lidar']
        imu    = sensors['imu']
        joints = sensors['joints']
        # Wrist camera (for grasping close-up)
        rgb, depth, mask = sensors['camera']

        dx, dy, dtheta = odometry.update(joints)

        # M5: Update state estimate
        state_est = pf.step(dx, dy, dtheta, lidar, obstacle_map_for_pf)
        est_x, est_y, est_theta = state_est

        # ======= THINK =======
        mission_event = 'tick'
        mission_data  = {}

        # Collision detection via LIDAR
        min_lidar = min(lidar)
        if min_lidar < 0.25:
            collision_count += 1
            mission_event = 'collision'

        # --- Navigation camera (forward-looking from base) ---
        try:
            nav_rgb, nav_depth, nav_mask, nav_view = get_nav_camera_image(robot_id)
            proj_np = np.array(
                p.computeProjectionMatrixFOV(60, 1.0, 0.1, 10.0),
                dtype=np.float64
            ).reshape(4, 4).T

            points, colors = depth_to_pointcloud(
                nav_depth, nav_rgb, proj_np, nav_view
            )

            if len(points) > 10:
                target_pts = detect_target(points, colors, tol=vision_tol)
                table_pts  = detect_table(points, colors, tol=vision_tol)

                if len(target_pts) > 20 and fsm.state in (State.SEARCH, State.NAVIGATE):
                    grasp_pos, grasp_axes = estimate_grasp_pose(target_pts)
                    if grasp_pos is not None:
                        world_grasp = [
                            est_x + grasp_pos[2] * np.cos(est_theta),
                            est_y + grasp_pos[2] * np.sin(est_theta),
                            grasp_pos[1] + 0.30   # nav cam height offset
                        ]
                        kb.assert_target_estimate(*world_grasp)
                        grasp_target_pos = world_grasp
                        mission_event = 'target_found'
                        mission_data  = {'target_pos': world_grasp[:2]}
                        print(f"[CogArch] Target spotted at {world_grasp[:2]}")

                elif len(table_pts) > 50 and fsm.state == State.SEARCH:
                    _, table_center = fit_table_plane(points, colors, tol=vision_tol)
                    if table_center is not None:
                        world_table = [
                            est_x + table_center[2] * np.cos(est_theta),
                            est_y + table_center[2] * np.sin(est_theta)
                        ]
                        kb.assert_table_position(*world_table)
                        mission_event = 'table_found'
                        mission_data  = {'table_pos': world_table}
                        print(f"[CogArch] Table spotted at {world_table}")

        except Exception as exc:
            perc_logger.log(exc)

        # --- Wrist camera for close-up grasp detection ---
        if fsm.state in (State.ALIGN, State.GRASP) and depth is not None:
            try:
                wrist_proj = np.array(
                    p.computeProjectionMatrixFOV(60, 1.0, 0.1, 10.0),
                    dtype=np.float64
                ).reshape(4, 4).T
                cam_state = p.getLinkState(robot_id, camera_link)
                cam_pos, cam_orn = cam_state[0], cam_state[1]
                rot = p.getMatrixFromQuaternion(cam_orn)
                fwd = [rot[0], rot[3], rot[6]]
                up  = [rot[2], rot[5], rot[8]]
                tgt = [cam_pos[0]+fwd[0], cam_pos[1]+fwd[1], cam_pos[2]+fwd[2]]
                wrist_view = p.computeViewMatrix(cam_pos, tgt, up)

                w_pts, w_cols = depth_to_pointcloud(
                    depth, rgb, wrist_proj, wrist_view
                )
                if len(w_pts) > 10:
                    close_target = detect_target(w_pts, w_cols, tol=vision_tol)
                    if len(close_target) > 10:
                        grasp_pos, _ = estimate_grasp_pose(close_target)
                        if grasp_pos is not None:
                            # Wrist cam is already at grasp range—use directly
                            grasp_target_pos = list(cam_pos)
                            grasp_target_pos[2] = 0.685   # table surface + cylinder
                            kb.assert_target_estimate(*grasp_target_pos)
            except Exception as exc:
                perc_logger.log(exc)

        # --- Search timeout: fall back to KB table position ---
        if fsm.state == State.SEARCH:
            search_step_count += 1
            if search_step_count >= SEARCH_TIMEOUT_STEPS:
                table_pos = kb.get_table_position()
                if table_pos:
                    mission_event = 'table_found'
                    mission_data  = {'table_pos': list(table_pos)}
                    print(f"[CogArch] Search timeout — navigating to KB table pos {table_pos}")
                    search_step_count = 0

        # State-specific event checks
        if fsm.state == State.NAVIGATE:
            if current_waypoints and waypoint_idx < len(current_waypoints):
                wx, wy = current_waypoints[waypoint_idx]
                dist_to_wp = np.hypot(est_x - wx, est_y - wy)
                if dist_to_wp < 0.2:
                    waypoint_idx += 1
                    if waypoint_idx >= len(current_waypoints):
                        mission_event = 'at_table'
                    else:
                        mission_event = 'waypoint_reached'

        elif fsm.state == State.ALIGN:
            if grasp_target_pos:
                table_pos = kb.get_table_position()
                if table_pos:
                    dist_to_table = np.hypot(
                        est_x - table_pos[0], est_y - table_pos[1]
                    )
                    if dist_to_table < 0.8:
                        mission_event = 'arm_aligned'

        elif fsm.state == State.GRASP:
            if grasp_target_pos:
                ee_state = p.getLinkState(robot_id, ee_index)
                ee_pos   = ee_state[0]
                dist = np.linalg.norm(
                    np.array(ee_pos) - np.array(grasp_target_pos)
                )
                if dist < 0.08:
                    mission_event = 'grasp_success'

        elif fsm.state == State.LIFT:
            ee_state = p.getLinkState(robot_id, ee_index)
            if ee_state[0][2] > lift_z - 0.05:
                mission_event = 'lift_success'

        # ---- Drive FSM ----
        fsm_state, cmd = fsm.transition(mission_event, mission_data)
        action = cmd.get('cmd', 'idle')

        # ======= ACT =======
        if action == 'rotate_search':
            search_rotation += 0.005   # slower, more deliberate scan
            nav_ctrl.rotate_to(search_rotation, est_theta)

        elif action == 'plan_path':
            goal = cmd.get('goal', fsm.table_position)
            if goal is not None:
                obs_for_path = [
                    {'position': [ox, oy, 0], 'size': 0.4}
                    for ox, oy in obstacle_map_for_pf
                ]
                path = plan_path_prolog(
                    est_x, est_y, goal[0], goal[1], obs_for_path, kb
                )
                fsm.set_waypoints(path)
                current_waypoints = path
                waypoint_idx = 0
                fsm.state = State.NAVIGATE

        elif action in ('continue_nav', 'next_waypoint'):
            if current_waypoints and waypoint_idx < len(current_waypoints):
                wx, wy = current_waypoints[waypoint_idx]
                nav_ctrl.drive_to(wx, wy, est_x, est_y, est_theta)

        elif action == 'backup':
            p.setJointMotorControl2(robot_id, left_joint,
                                     p.VELOCITY_CONTROL, targetVelocity=-2.0, force=10.0)
            p.setJointMotorControl2(robot_id, right_joint,
                                     p.VELOCITY_CONTROL, targetVelocity=-2.0, force=10.0)

        elif action == 'align_arm':
            if grasp_target_pos:
                pre_grasp = [
                    grasp_target_pos[0],
                    grasp_target_pos[1],
                    grasp_target_pos[2] + 0.15
                ]
                arm_ctrl.move_to_position(pre_grasp)

        elif action == 'execute_grasp':
            if grasp_target_pos:
                arm_ctrl.move_to_position(grasp_target_pos)

        elif action == 'lift_object':
            if grasp_target_pos:
                lift_pos = [
                    grasp_target_pos[0],
                    grasp_target_pos[1],
                    lift_z
                ]
                done = arm_ctrl.move_to_position(lift_pos)
                if done:
                    mission_event = 'lift_success'

        elif action in ('stop', 'idle', 'hold'):
            p.setJointMotorControl2(robot_id, left_joint,
                                     p.VELOCITY_CONTROL, targetVelocity=0, force=10.0)
            p.setJointMotorControl2(robot_id, right_joint,
                                     p.VELOCITY_CONTROL, targetVelocity=0, force=10.0)

        # M9: Report at termination
        if fsm.is_terminal():
            table_pos = kb.get_table_position() or [0, 0]
            dist_rem  = np.hypot(est_x - table_pos[0], est_y - table_pos[1])
            learner.end_trial(
                success=(fsm.state == State.DONE),
                distance_remaining=dist_rem,
                collision_count=collision_count
            )
            vision_tol = learner.get_current_params()['vision_tol']
            print(f"[CogArch] Mission ended: {fsm.state}")
            break

        p.stepSimulation()   # DO NOT TOUCH
        time.sleep(1. / 240.)  # DO NOT TOUCH

    print("[CogArch] Simulation complete.")


if __name__ == '__main__':
    main()
