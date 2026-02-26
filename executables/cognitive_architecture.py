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
    n = p.getNumJoints(robot_id)
    for i in range(n):
        info = p.getJointInfo(robot_id, i)
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
        self._prev_left = 0.0
        self._prev_right = 0.0

    def update(self, joints):
        """Returns (dx, dy, dtheta) in robot frame."""
        left_vel = joints.get('left_wheel_joint', {}).get('velocity', 0.0)
        right_vel = joints.get('right_wheel_joint', {}).get('velocity', 0.0)

        v_left = left_vel * self.r
        v_right = right_vel * self.r

        v = (v_left + v_right) / 2.0
        omega = (v_right - v_left) / self.wb

        dx = v * self.dt
        dy = 0.0
        dtheta = omega * self.dt
        return dx, dy, dtheta


# ===================== MAIN COGNITIVE LOOP =====================

def main():
    print("[CogArch] Starting IIS Robot Cognitive Architecture...")

    # ---- M2: Build World ----
    world = build_world(gui=True)
    robot_id = world['robot_id']
    scene_map = world['scene_map']

    # ---- Load scene map ----
    with open(os.path.join(ROOT, 'executables', 'scene_map.json')) as f:
        scene_map = json.load(f)

    # ---- Discover joint/link indices ----
    left_joint = get_joint_index(robot_id, 'left_wheel_joint')
    right_joint = get_joint_index(robot_id, 'right_wheel_joint')
    arm_joints = [
        get_joint_index(robot_id, 'arm_base_joint'),
        get_joint_index(robot_id, 'shoulder_joint'),
        get_joint_index(robot_id, 'elbow_joint'),
        get_joint_index(robot_id, 'wrist_joint'),
    ]
    ee_index = get_link_index(robot_id, 'end_effector')
    lidar_link = get_link_index(robot_id, 'lidar_link')
    camera_link = get_link_index(robot_id, 'camera_link')

    # ---- M8: Knowledge Base ----
    kb = KnowledgeBase()
    kb.populate_from_scene_map(scene_map)

    # ---- M9: Learning ----
    learner = LearningModule()
    trial_params = learner.start_trial()
    h_gains = trial_params['heading_pid']
    d_gains = trial_params['distance_pid']
    vision_tol = learner.get_current_params()['vision_tol']

    # ---- M5: Particle Filter ----
    robot_init = scene_map['robot']
    pf = ParticleFilter(n_particles=300)
    init_pos = robot_init['position']
    init_orn = robot_init['orientation']
    init_theta = p.getEulerFromQuaternion(init_orn)[2]
    pf.initialize_at(init_pos[0], init_pos[1], init_theta)

    # ---- M6: Controllers ----
    nav_ctrl = DifferentialDriveController(
        robot_id, left_joint, right_joint
    )
    nav_ctrl.heading_pid.update_gains(**h_gains)
    nav_ctrl.distance_pid.update_gains(**d_gains)

    arm_ctrl = ArmController(robot_id, ee_index, arm_joints)

    # ---- M7: FSM ----
    fsm = MissionFSM()
    fsm.table_position = scene_map['table']['position'][:2]

    # ---- Odometry ----
    odometry = OdometryEstimator()

    # ---- M3 Projection matrices (cached) ----
    proj_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 10.0)
    view_matrix = None  # Updated each frame

    # ---- State variables ----
    state_est = (init_pos[0], init_pos[1], init_theta)
    obstacle_map_for_pf = [(obs['position'][0], obs['position'][1])
                           for obs in scene_map['obstacles']]
    collision_count = 0
    search_rotation = 0.0
    current_waypoints = []
    waypoint_idx = 0
    grasp_target_pos = None
    lift_z = 0.9
    mission_event = 'tick'
    mission_data = {}

    # Initial FSM tick
    fsm_state, cmd = fsm.transition('tick')

    print("[CogArch] Entering Sense-Think-Act loop...")

    # ===================== MAIN LOOP =====================
    while p.isConnected():  # DO NOT TOUCH

        # ======= SENSE =======
        sensors = preprocess_all(robot_id, lidar_link, camera_link)
        lidar = sensors['lidar']
        imu = sensors['imu']
        joints = sensors['joints']
        rgb, depth, mask = sensors['camera']

        dx, dy, dtheta = odometry.update(joints)

        # M5: Update state estimate
        state_est = pf.step(dx, dy, dtheta, lidar, obstacle_map_for_pf)
        est_x, est_y, est_theta = state_est

        # ======= THINK =======
        mission_event = 'tick'
        mission_data = {}

        # Collision detection via LIDAR (min reading < 0.25m)
        min_lidar = min(lidar)
        if min_lidar < 0.25:
            collision_count += 1
            mission_event = 'collision'

        # Perception
        try:
            # Rebuild point cloud from current camera pose
            cam_state = p.getLinkState(robot_id, camera_link)
            cam_pos, cam_orn = cam_state[0], cam_state[1]
            rot = p.getMatrixFromQuaternion(cam_orn)
            fwd = [rot[0], rot[3], rot[6]]
            up = [rot[2], rot[5], rot[8]]
            tgt = [cam_pos[0]+fwd[0], cam_pos[1]+fwd[1], cam_pos[2]+fwd[2]]
            view_matrix = p.computeViewMatrix(cam_pos, tgt, up)

            # M4: Detect objects
            if depth is not None:
                proj_list = list(p.computeProjectionMatrixFOV(60, 1.0, 0.1, 10.0))
                import numpy as np
                proj_np = np.array(proj_list).reshape(4, 4).T

                points, colors = depth_to_pointcloud(
                    depth, rgb, proj_np, view_matrix
                )

                if len(points) > 10:
                    target_pts = detect_target(points, colors)
                    table_pts = detect_table(points, colors)

                    if len(target_pts) > 20 and fsm.state in (State.SEARCH, State.NAVIGATE):
                        grasp_pos, grasp_axes = estimate_grasp_pose(target_pts)
                        if grasp_pos is not None:
                            # Transform from camera frame to world frame (approx)
                            world_grasp = [
                                est_x + grasp_pos[2] * np.cos(est_theta),
                                est_y + grasp_pos[2] * np.sin(est_theta),
                                grasp_pos[1] + cam_pos[2]
                            ]
                            kb.assert_target_estimate(*world_grasp)
                            grasp_target_pos = world_grasp
                            mission_event = 'target_found'
                            mission_data = {'target_pos': world_grasp[:2]}

                    elif len(table_pts) > 50 and fsm.state == State.SEARCH:
                        table_plane, table_center = fit_table_plane(points, colors)
                        if table_center is not None:
                            world_table = [
                                est_x + table_center[2] * np.cos(est_theta),
                                est_y + table_center[2] * np.sin(est_theta)
                            ]
                            kb.assert_table_position(*world_table)
                            mission_event = 'table_found'
                            mission_data = {'table_pos': world_table}
        except Exception as e:
            pass  # Perception is best-effort

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
                    dist_to_table = np.hypot(est_x - table_pos[0], est_y - table_pos[1])
                    if dist_to_table < 0.8:
                        mission_event = 'arm_aligned'

        elif fsm.state == State.GRASP:
            if grasp_target_pos:
                ee_state = p.getLinkState(robot_id, ee_index)
                ee_pos = ee_state[0]
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
            search_rotation += 0.01
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
                # Immediately start navigating
                fsm.state = State.NAVIGATE

        elif action in ('continue_nav', 'next_waypoint'):
            if current_waypoints and waypoint_idx < len(current_waypoints):
                wx, wy = current_waypoints[waypoint_idx]
                nav_ctrl.drive_to(wx, wy, est_x, est_y, est_theta)

        elif action == 'backup':
            # Reverse away from obstacle
            p.setJointMotorControl2(robot_id, left_joint,
                                     p.VELOCITY_CONTROL, targetVelocity=-2.0, force=10.0)
            p.setJointMotorControl2(robot_id, right_joint,
                                     p.VELOCITY_CONTROL, targetVelocity=-2.0, force=10.0)

        elif action == 'align_arm':
            if grasp_target_pos:
                pre_grasp = [grasp_target_pos[0],
                              grasp_target_pos[1],
                              grasp_target_pos[2] + 0.15]
                arm_ctrl.move_to_position(pre_grasp)

        elif action == 'execute_grasp':
            if grasp_target_pos:
                arm_ctrl.move_to_position(grasp_target_pos)

        elif action == 'lift_object':
            if grasp_target_pos:
                lift_pos = [grasp_target_pos[0],
                             grasp_target_pos[1],
                             lift_z]
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
            dist_rem = np.hypot(est_x - table_pos[0], est_y - table_pos[1])
            learner.end_trial(
                success=(fsm.state == State.DONE),
                distance_remaining=dist_rem,
                collision_count=collision_count
            )
            print(f"[CogArch] Mission ended: {fsm.state}")
            break

        p.stepSimulation()   # DO NOT TOUCH
        time.sleep(1. / 240.)  # DO NOT TOUCH

    print("[CogArch] Simulation complete.")


if __name__ == '__main__':
    main()
