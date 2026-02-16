"""
Module 10: Cognitive Architecture - SENSE-THINK-ACT Loop
Integrates all 10 modules for autonomous navigate-to-grasp mission.

This is the main executive controller that combines:
- M1: Task specification (defined in README)
- M2: URDF robot hardware
- M3: Sensor preprocessing (sensor_wrapper.py)
- M4: Perception (object detection, RANSAC)
- M5: State estimation (Particle Filter)  
- M6: Motion control (PID, IK)
- M7: Action planning (FSM, task sequencing)
- M8: Knowledge representation (Prolog KB)
- M9: Learning (parameter optimization)
- M10: This cognitive architecture

SENSE-THINK-ACT Cycle:
    SENSE:  Read sensors, estimate state, detect objects
    THINK:  Update knowledge, plan actions, make decisions
    ACT:    Execute motion commands, control gripper
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2
import math
import sys
import os
import json

# Add project paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# M2: Hardware (URDF) & Environment
from src.environment.world_builder import build_world

# M3: Sensors
from src.robot.sensor_wrapper import (
    get_camera_image, get_lidar_data, get_imu_data, get_joint_states
)

# M4: Perception
from src.modules.perception import detect_objects_by_color, RANSAC_Segmentation

# M5: State Estimation
from src.modules.state_estimation import ParticleFilter

# M6: Motion Control
from src.modules.motion_control import PIDController, move_to_goal, grasp_object

# M7: Action Planning
from src.modules.fsm import RobotFSM, RobotState
from src.modules.action_planning import get_action_planner, get_grasp_planner

# M8: Knowledge Representation
from src.modules.knowledge_reasoning import get_knowledge_base

# M9: Learning
from src.modules.learning import Learner
from src.modules.learning import DEFAULT_PARAMETERS



# Robot physical constants (must match URDF and state_estimation)
WHEEL_RADIUS = 0.1
WHEEL_BASELINE = 0.45
CAMERA_HEIGHT = 0.25    # Base spawn height (0.1) + camera Z offset (0.15)
DEPTH_NEAR = 0.1
DEPTH_FAR = 10.0


class CognitiveArchitecture:
    """
    Main cognitive architecture implementing the Sense-Think-Act loop.
    """
    
    def __init__(self, robot_id, table_id, room_id, target_id, parameters=None):
        # Robot IDs from PyBullet
        self.robot_id = robot_id
        self.table_id = table_id
        self.room_id = room_id
        self.target_id = target_id

        self.prev_time = time.time()

        # M5: State Estimator (Particle Filter)
        self.state_estimator = ParticleFilter(num_particles=500)
        
        # M7: FSM for high-level control
        self.fsm = RobotFSM()
        
        # M7: Action Planners
        self.action_planner = get_action_planner()
        self.grasp_planner = get_grasp_planner()
        
        # M8: Knowledge Base
        self.kb = get_knowledge_base()
        
        # M9: Learner (for parameter optimization)
        parameters = parameters or {}
        self.learner = Learner(self)

        # Load best parameters from experience
        _, best_params = self.learner.offline_learning()

        if best_params:
            print("[Learning] Loaded best learned parameters")
            self.apply_parameters(best_params)
        else:
            self.apply_parameters(DEFAULT_PARAMETERS)
            print("[Learning] No learned parameters found, using defaults")
        
        # Robot configuration - wheel joints [FL, FR, BL, BR]
        self.wheel_joints = [0, 1, 2, 3]
        self.wheel_names = ['fl_wheel_joint', 'fr_wheel_joint',
                            'bl_wheel_joint', 'br_wheel_joint']
        
        # Auto-detect special joint indices fromclea URDF
        self.arm_joints = []
        self.gripper_joints = []
        self.lift_joint_idx = None
        self.camera_link_idx = None
        self._detect_robot_joints()
        
        # Task state
        self.target_position = None
        self.table_position = None
        self.table_orientation = None
        self.table_size = None
        self.obstacles = []
        self.current_waypoint = None
        self.approach_standoff = None
        
        # Timing
        self.step_counter = 0
        self.dt = 1.0 / 240.0
        # self.dt = p.getPhysicsEngineParameters()['fixedTimeStep']

        
        # Initialize robot knowledge
        self._initialize_world_knowledge()
        
        # Enable wheel motors
        self._initialize_motors()
        
    def _initialize_motors(self):
        """Enable and reset wheel motors"""
        for i in self.wheel_joints:
            p.setJointMotorControl2(
                self.robot_id, i, p.VELOCITY_CONTROL,
                targetVelocity=0, force=5000
            )
            p.enableJointForceTorqueSensor(self.robot_id, i, True)
        print("[CogArch] Motors initialized")
        
    def _detect_robot_joints(self):
        """Detect all special joint indices from robot URDF"""
        num_joints = p.getNumJoints(self.robot_id)
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            link_name = joint_info[12].decode('utf-8')
            
            # Gripper finger joints
            if 'left_finger_joint' in joint_name or 'right_finger_joint' in joint_name:
                self.gripper_joints.append(i)
            
            # Lift joint (prismatic on torso)
            if joint_name == 'lift_joint':
                self.lift_joint_idx = i
            
            # Camera link
            if 'rgbd_camera' in link_name or 'camera' in link_name:
                self.camera_link_idx = i
            
            # Arm joints (revolute joints in the arm chain)
            if joint_name in ['arm_base_joint', 'shoulder_joint', 'elbow_joint',
                              'wrist_pitch_joint', 'wrist_roll_joint']:
                self.arm_joints.append(i)
        
        print(f"[CogArch] Detected: {len(self.gripper_joints)} gripper joints, "
              f"{len(self.arm_joints)} arm joints, "
              f"lift_joint={self.lift_joint_idx}, camera_link={self.camera_link_idx}")
        
    def _initialize_world_knowledge(self):
        """Initialize knowledge base with world state from initial map"""
        map_file = "initial_map.json"
        if os.path.exists(map_file):
            with open(map_file, 'r') as f:
                world_map = json.load(f)
                
            # Add table to knowledge base
            if 'table' in world_map:
                table_data = world_map['table']
                pos = table_data['position']
                self.kb.add_position('table', pos[0], pos[1], pos[2])
                self.kb.add_detected_object('table', 'furniture', 'brown', pos)
                self.table_position = pos
                if 'orientation' in table_data:
                    self.table_orientation = table_data['orientation']
                if 'size' in table_data:
                    self.table_size = table_data['size']
                # Add table as a large obstacle so path planner routes around it
                self.obstacles.append(pos[:2])
                
            # Add obstacles to knowledge base
            if 'obstacles' in world_map:
                for i, obs in enumerate(world_map['obstacles']):
                    pos = obs['position']
                    color = self._rgba_to_color_name(obs['color_rgba'])
                    obj_id = f'obstacle{i+1}'
                    self.kb.add_position(obj_id, pos[0], pos[1], pos[2])
                    self.kb.add_detected_object(obj_id, 'static', color, pos)
                    self.obstacles.append(pos[:2])
                    
            print(f"[CogArch] Loaded {len(self.obstacles)} obstacles from initial map")
    
    def _rgba_to_color_name(self, rgba):
        """Convert RGBA to color name"""
        r, g, b, _ = rgba
        if r > 0.9 and g < 0.1 and b < 0.1:
            return 'red'
        elif r < 0.1 and g < 0.1 and b > 0.9:
            return 'blue'
        elif r > 0.9 and g > 0.6 and b < 0.1:
            return 'orange'
        elif r > 0.9 and g > 0.9 and b < 0.1:
            return 'yellow'
        elif r > 0.9 and g > 0.7:
            return 'pink'
        elif r < 0.1 and g < 0.1 and b < 0.1:
            return 'black'
        elif r > 0.4 and r < 0.6 and g > 0.2 and g < 0.4:
            return 'brown'
        return 'unknown'
    
    def _check_gripper_contact(self):
        """
        M3: Check if gripper fingers are in contact with target object.
        Uses joint state feedback (torque + position) to detect grasp.
        Fingers that are stopped partway with resistance torque indicate contact.
        """
        if not self.gripper_joints:
            return False
        
        # Use sensor wrapper to read joint states (legal API)
        joint_states = get_joint_states(self.robot_id)
        
        contact_count = 0
        for finger_idx in self.gripper_joints:
            # Find the joint name for this index
            for name, data in joint_states.items():
                if data['index'] == finger_idx:
                    torque = abs(data['applied_torque'])
                    position = abs(data['position'])
                    # If the finger has significant resistance torque and has moved
                    # from its open position, something is between the fingers
                    if torque > 0.5 and position > 0.005:
                        contact_count += 1
                    break
        
        # Need contact on at least one finger
        if contact_count > 0:
            if self.step_counter % 60 == 0:
                print("[Sense] Gripper contact detected via joint torque feedback")
            return True
        
        return False
    
    def _convert_depth_buffer(self, depth_buffer_value):
        """
        Convert PyBullet's normalized depth buffer to actual depth in meters.
        Depth buffer values are in [0, 1] range.
        """
        if depth_buffer_value <= 0 or depth_buffer_value >= 1.0:
            return float('inf')
        true_depth = DEPTH_FAR * DEPTH_NEAR / (DEPTH_FAR - (DEPTH_FAR - DEPTH_NEAR) * depth_buffer_value)
        return true_depth
    
    def _compute_approach_standoff(self, target_pos, robot_pose):
        """
        Compute optimal approach standoff position near the table.
        Approaches from the long side (1.5m) of the table so the arm
        only needs to reach across the short dimension (0.4m from edge to center).
        """
        standoff_dist = 1.10  # meters from target center (table half-width 0.4 + robot 0.3 + margin 0.4)
        
        if self.table_orientation is not None:
            # Get table yaw from orientation quaternion
            euler = p.getEulerFromQuaternion(self.table_orientation)
            table_yaw = euler[2]
            
            # Approach directions perpendicular to long side (along short axis)
            dir1 = [-math.sin(table_yaw), math.cos(table_yaw)]
            dir2 = [math.sin(table_yaw), -math.cos(table_yaw)]
            
            # Choose direction that faces the robot
            dx = robot_pose[0] - target_pos[0]
            dy = robot_pose[1] - target_pos[1]
            dot1 = dx * dir1[0] + dy * dir1[1]
            dot2 = dx * dir2[0] + dy * dir2[1]
            
            approach_dir = dir1 if dot1 > dot2 else dir2
            
            standoff = [
                target_pos[0] + standoff_dist * approach_dir[0],
                target_pos[1] + standoff_dist * approach_dir[1]
            ]
        else:
            # Fallback: approach from robot's current direction
            dx = target_pos[0] - robot_pose[0]
            dy = target_pos[1] - robot_pose[1]
            dist = np.hypot(dx, dy)
            if dist > 0.01:
                standoff = [
                    target_pos[0] - standoff_dist * dx / dist,
                    target_pos[1] - standoff_dist * dy / dist
                ]
            else:
                standoff = list(target_pos[:2])
        
        return standoff

    def _distance_to_table(self, x, y):
        """
        Compute approximate 2D distance from point (x,y) to table footprint.
        Returns 0 if point is inside the table, positive if outside.
        """
        if self.table_position is None:
            return float('inf')
        # Transform point into table-local frame
        tx, ty = self.table_position[0], self.table_position[1]
        dx, dy = x - tx, y - ty
        if self.table_orientation is not None:
            euler = p.getEulerFromQuaternion(self.table_orientation)
            yaw = -euler[2]  # inverse rotation
            local_x = dx * math.cos(yaw) - dy * math.sin(yaw)
            local_y = dx * math.sin(yaw) + dy * math.cos(yaw)
        else:
            local_x, local_y = dx, dy
        # Table half-dimensions (1.5 x 0.8)
        hx, hy = 0.75, 0.40
        # Distance from box boundary
        cx = max(abs(local_x) - hx, 0.0)
        cy = max(abs(local_y) - hy, 0.0)
        return math.hypot(cx, cy)

    def _get_lidar_obstacle_avoidance(self, lidar, forward_vel, relaxed=False, robot_pose=None):
        """
        M4/M6: Adjust motion to avoid obstacles detected by lidar.
        Also enforces a virtual keep-out zone around the table (lidar can't see it).
        
        Args:
            lidar: list of 36 ray distances (0=forward, going counter-clockwise)
            forward_vel: desired forward velocity
            relaxed: if True, use smaller thresholds (for approaching the table)
            robot_pose: [x, y, theta] for table keep-out check
            
        Returns:
            (adjusted_forward_vel, avoidance_angular_vel)
        """
        if lidar is None or len(lidar) == 0:
            return forward_vel, 0.0
        
        # Virtual table keep-out zone (lidar can't detect the table)
        if robot_pose is not None and self.table_position is not None:
            table_dist = self._distance_to_table(robot_pose[0], robot_pose[1])
            table_keepout = 0.2 if relaxed else 0.6
            if table_dist < table_keepout:
                # Robot is dangerously close to or inside table footprint
                # Override: push away from table center
                away_angle = math.atan2(
                    robot_pose[1] - self.table_position[1],
                    robot_pose[0] - self.table_position[0]
                )
                heading_diff = away_angle - robot_pose[2]
                heading_diff = math.atan2(math.sin(heading_diff), math.cos(heading_diff))
                if abs(heading_diff) < math.pi / 2:
                    return min(forward_vel, 3.0), 5.0 * heading_diff  # drive away
                else:
                    return -2.0, 5.0 if heading_diff > 0 else -5.0  # back up and turn
        
        num_rays = len(lidar)
        if relaxed:
            obstacle_threshold = 0.3   # meters - allow getting close to table
            emergency_threshold = 0.15
        else:
            obstacle_threshold = 0.8  # meters
            emergency_threshold = 0.3  # meters
        
        # Check front cone (about +/- 30 degrees)
        front_indices = [i % num_rays for i in range(-3, 4)]
        front_dists = [lidar[i] for i in front_indices]
        min_front = min(front_dists)
        
        # Check left side (rays 3-8)
        left_dists = [lidar[i] for i in range(3, 9)]
        avg_left = np.mean(left_dists)
        
        # Check right side (rays 28-33)
        right_dists = [lidar[i] for i in range(num_rays - 8, num_rays - 2)]
        avg_right = np.mean(right_dists)
        
        # Check rear cone (about +/- 30 degrees behind)
        rear_center = num_rays // 2  # ray 18 for 36-ray
        rear_indices = [i % num_rays for i in range(rear_center - 3, rear_center + 4)]
        rear_dists = [lidar[i] for i in rear_indices]
        min_rear = min(rear_dists)
        
        avoidance_angular = 0.0
        
        if forward_vel < 0:
            # Moving backward: check rear obstacles
            if min_rear < emergency_threshold:
                forward_vel = 1.0  # Drive forward to escape
                avoidance_angular = 5.0 if avg_left > avg_right else -5.0
            elif min_rear < obstacle_threshold:
                slowdown = min_rear / obstacle_threshold
                forward_vel *= slowdown
                turn_strength = 3.0 * (1.0 - slowdown)
                avoidance_angular = turn_strength if avg_left > avg_right else -turn_strength
        else:
            # Moving forward: check front obstacles
            if min_front < emergency_threshold:
                forward_vel = -1.0  # Back up slightly
                avoidance_angular = 5.0 if avg_left > avg_right else -5.0
            elif min_front < obstacle_threshold:
                slowdown = min_front / obstacle_threshold
                forward_vel *= slowdown
                turn_strength = 3.0 * (1.0 - slowdown)
                avoidance_angular = turn_strength if avg_left > avg_right else -turn_strength
        
        return forward_vel, avoidance_angular
    
    # ==================== SENSE ====================
    def sense(self):
        """
        SENSE phase: Acquire sensor data and update state estimate.
        Returns sensor_data dict for use in THINK phase.
        """
        # M3: Get raw sensor data using sensor wrapper (legal API)
        camera_link = self.camera_link_idx if self.camera_link_idx is not None else -1
        rgb, depth, _ = get_camera_image(self.robot_id, sensor_link_id=camera_link)
        lidar = get_lidar_data(self.robot_id, num_rays=36)
        imu = get_imu_data(self.robot_id)
        joint_states = get_joint_states(self.robot_id)
        
        # M5: Get wheel velocities from joint states using correct URDF joint names
        wheel_vels = []
        for name in self.wheel_names:
            if name in joint_states:
                wheel_vels.append(joint_states[name]['velocity'])
            else:
                wheel_vels.append(0.0)
        
        # M5: Update state estimate with sensor fusion
        sensor_data_for_pf = {
            'imu': imu,
            'lidar': lidar,
            'joint_states': joint_states
        }
        
        self.state_estimator.predict(wheel_vels, self.dt)
        self.state_estimator.measurement_update(sensor_data_for_pf)
        
        # Resample periodically to prevent particle degeneracy
        if self.step_counter % 10 == 0:
            self.state_estimator.resample()
        
        estimated_pose = self.state_estimator.estimate_pose()
        
        # M4: Perception - Detect objects every 10 steps
        if rgb is not None and self.step_counter % 10 == 0:
            rgb_array = np.reshape(rgb, (240, 320, 4)).astype(np.uint8)
            bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)
            
            detections = detect_objects_by_color(bgr, min_area=10)
            
            # Log detections periodically
            if self.step_counter % 240 == 0 and len(detections) > 0:
                colors_found = [d['color'] for d in detections]
                print(f"[Perception] Detected {len(detections)} objects: {colors_found}")
            
            # Look for red target
            for det in detections:
                if det['color'] == 'red':
                    bbox = det['bbox']
                    center_x = int(bbox[0] + bbox[2] / 2)
                    center_y = int(bbox[1] + bbox[3] / 2)
                    
                    if depth is not None and 0 <= center_y < 240 and 0 <= center_x < 320:
                        if len(depth.shape) == 1:
                            depth_reshaped = np.reshape(depth, (240, 320))
                        else:
                            depth_reshaped = depth
                        
                        raw_depth = depth_reshaped[center_y, center_x]
                        
                        # Convert normalized depth buffer to actual meters
                        true_depth = self._convert_depth_buffer(raw_depth)
                        
                        if true_depth < 0.1 or true_depth > 10.0 or np.isnan(true_depth) or np.isinf(true_depth):
                            continue
                        
                        # Camera intrinsics (FOV=60 deg, width=320, height=240)
                        fx = fy = (320 / 2.0) / np.tan(np.deg2rad(60 / 2.0))
                        cx, cy_cam = 160.0, 120.0
                        
                        # Convert pixel + depth to camera coordinates
                        cam_x = (center_x - cx) * true_depth / fx
                        cam_y = (center_y - cy_cam) * true_depth / fy
                        cam_z = true_depth  # forward distance
                        
                        # Transform to world frame
                        robot_x, robot_y, robot_theta = estimated_pose
                        cos_t = math.cos(robot_theta)
                        sin_t = math.sin(robot_theta)
                        
                        # Camera frame to robot body frame:
                        #   robot_forward(+X) = cam_z (depth)
                        #   robot_left(+Y)    = -cam_x (camera right is robot -Y)
                        robot_body_x = cam_z
                        robot_body_y = -cam_x
                        
                        world_x = robot_x + robot_body_x * cos_t - robot_body_y * sin_t
                        world_y = robot_y + robot_body_x * sin_t + robot_body_y * cos_t
                        # Height: camera is at CAMERA_HEIGHT, cam_y positive = object lower
                        world_z = CAMERA_HEIGHT - cam_y
                        
                        self.target_position = [world_x, world_y, world_z]
                        
                        # Add to knowledge base
                        self.kb.add_position('target', *self.target_position)
                        print(f"[CogArch] TARGET DETECTED at ({world_x:.2f}, {world_y:.2f}, {world_z:.2f}), "
                              f"depth={true_depth:.2f}m")
                        break
        
        # M3: Check gripper contact using joint state feedback (legal)
        gripper_contact = self._check_gripper_contact()
        
        return {
            'pose': estimated_pose,
            'rgb': rgb,
            'depth': depth,
            'lidar': lidar,
            'imu': imu,
            'joint_states': joint_states,
            'target_detected': self.target_position is not None,
            'target_position': self.target_position,
            'gripper_contact': gripper_contact
        }
    
    # ==================== THINK ====================
    def think(self, sensor_data):
        """
        THINK phase: Process sensor data, update knowledge, plan actions.
        Returns control commands for ACT phase.
        """
        pose = sensor_data['pose']
        
        # M8: Query knowledge base for decision making
        # M8: Query knowledge base
        target_pos = self.kb.query_position('target')
        # Override with sensor data if available
        if sensor_data['target_detected']:
            target_pos = sensor_data['target_position']
        

        if target_pos:
            print(f"[DEBUG] Target position from KB: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})")
        else:
            print("[DEBUG] Target position unknown in KB")

        # --- Table detection ---
        if self.table_position:
            print(f"[DEBUG] Table detected at: ({self.table_position[0]:.2f}, {self.table_position[1]:.2f})")
        else:
            print("[DEBUG] Table not yet detected")


        # Calculate 2D horizontal distance to target
        if target_pos:
            dx = target_pos[0] - pose[0]
            dy = target_pos[1] - pose[1]
            distance_2d = np.sqrt(dx**2 + dy**2)
            print(f"[DEBUG] Distance to target: {distance_2d:.2f}m")
            
            # Compute standoff as soon as we know the target
            if self.approach_standoff is None:
                self.approach_standoff = self._compute_approach_standoff(
                    target_pos, pose
                )
                print(f"[CogArch] Computed approach standoff: "
                      f"({self.approach_standoff[0]:.2f}, {self.approach_standoff[1]:.2f})")
            
            # In NAVIGATE/APPROACH states, FSM distance = dist to standoff
            if self.approach_standoff is not None and self.fsm.state == RobotState.NAVIGATE:
                sdx = self.approach_standoff[0] - pose[0]
                sdy = self.approach_standoff[1] - pose[1]
                distance_for_fsm = np.sqrt(sdx**2 + sdy**2)
            else:
                distance_for_fsm = distance_2d
        else:
            distance_2d = float('inf')
            distance_for_fsm = float('inf')
        
        # M7: Update FSM with sensor feedback
        fsm_sensor_data = {
            'target_visible': sensor_data['target_detected'],
            'target_position': target_pos,
            'distance_to_target': distance_for_fsm,
            'collision_detected': False,
            'gripper_contact': sensor_data.get('gripper_contact', False),
            'object_grasped': sensor_data.get('gripper_contact', False),
            'estimated_pose': pose
        }
        
        self.fsm.update(fsm_sensor_data)
        print(f"[DEBUG] FSM state: {self.fsm.state}")
        
        # M7: Action planning based on FSM state
        control_commands = {
            'mode': 'idle',
            'target': None,
            'gripper': 'open'
        }
        
        if self.fsm.state == RobotState.SEARCH:
            # If table position is known, navigate toward it while searching
            if self.table_position:
                table_dx = self.table_position[0] - pose[0]
                table_dy = self.table_position[1] - pose[1]
                table_dist = np.hypot(table_dx, table_dy)
                print(f"[DEBUG] Distance to table: {table_dist:.2f}m")
                
                if table_dist < 2.0:
                    # Close to table: orbit around it to scan all sides
                    control_commands = {
                        'mode': 'search_orbit',
                        'table_pos': self.table_position[:2],
                        'pose': pose,
                        'orbit_radius': 2.0,
                        'lidar': sensor_data['lidar']
                    }
                    print("[DEBUG] Orbiting table to search")
                else:
                    control_commands = {
                        'mode': 'search_approach',
                        'target': self.table_position[:2],
                        'pose': pose,
                        'angular_vel': 2.0,
                        'lidar': sensor_data['lidar']
                    }
                    print("[DEBUG] Approaching table to search")
            else:
                control_commands = {
                    'mode': 'search_rotate',
                    'angular_vel': 3.0
                }
                print("[DEBUG] Rotating to find table")
            
        elif self.fsm.state == RobotState.NAVIGATE:
            # Navigate to standoff point (not directly at the cylinder)
            nav_goal = self.approach_standoff if self.approach_standoff else (
                target_pos[:2] if target_pos else None
            )
            if nav_goal and self.current_waypoint is None:
                self.action_planner.create_plan(
                    pose[:2], nav_goal, self.obstacles
                )
                self.current_waypoint = self.action_planner.get_next_waypoint()
                print(f"[DEBUG] Created navigation plan to: ({nav_goal[0]:.2f}, {nav_goal[1]:.2f})")
                
            if self.current_waypoint:
                control_commands = {
                    'mode': 'navigate',
                    'target': self.current_waypoint,
                    'pose': pose,
                    'lidar': sensor_data['lidar']
                }
                
                # Check if waypoint reached
                dist = np.hypot(self.current_waypoint[0] - pose[0],
                               self.current_waypoint[1] - pose[1])
                print(f"[DEBUG] Distance to waypoint: {dist:.2f}m")
                
                if dist < 0.3:
                    print("[DEBUG] Waypoint reached, advancing to next")
                    self.action_planner.advance_waypoint()
                    self.current_waypoint = self.action_planner.get_next_waypoint()
                
                    
        elif self.fsm.state == RobotState.APPROACH:
            if target_pos:
                control_commands = {
                    'mode': 'approach',
                    'target': target_pos[:2],
                    'pose': pose,
                    'lidar': sensor_data['lidar'],
                    'relaxed_avoidance': True
                }
                
        elif self.fsm.state == RobotState.GRASP:
            if target_pos:
                grasp_plan = self.grasp_planner.plan_grasp(target_pos)
                grasp_time = self.fsm.get_time_in_state()
                
                # Multi-phase grasp sequence (8s total timeout)
                if grasp_time < 2.5:
                    phase = 'reach_above'   # Open gripper, raise lift, IK above target
                elif grasp_time < 5.5:
                    phase = 'reach_target'  # IK to grasp position, gripper still open
                else:
                    phase = 'close_gripper' # Close gripper on target
                
                control_commands = {
                    'mode': 'grasp',
                    'approach_pos': grasp_plan['approach_pos'],
                    'grasp_pos': grasp_plan['grasp_pos'],
                    'orientation': grasp_plan['orientation'],
                    'phase': phase
                }
                
        elif self.fsm.state == RobotState.LIFT:
            control_commands = {
                'mode': 'lift',
                'lift_height': 0.2,
                'gripper': 'close'
            }
            
        elif self.fsm.state == RobotState.SUCCESS:
            control_commands = {
                'mode': 'success',
                'gripper': 'close'
            }
            
        elif self.fsm.state == RobotState.FAILURE:
            # Reset standoff and waypoint so new ones are computed on retry
            self.approach_standoff = None
            self.current_waypoint = None
            control_commands = {
                'mode': 'failure',
                'gripper': 'open',
                'lidar': sensor_data['lidar']
            }

        print(f"[DEBUG] Control command: {control_commands}")
        print(f"[DEBUG] approach_standoff: {self.approach_standoff}, current_waypoint: {self.current_waypoint}")

        
        return control_commands
    
    # ==================== ACT ====================
    def act(self, control_commands):
        """
        ACT phase: Execute motion commands on the robot.
        """
        mode = control_commands.get('mode', 'idle')
        
        if mode == 'search_rotate':
            # Rotate in place to search for target
            angular_vel = np.clip(
                control_commands.get('angular_vel', 3.0),
                -self.max_angular_speed,
                self.max_angular_speed
            )
            if self.step_counter % 240 == 0:
                print(f"[Act] SEARCH: rotating at {angular_vel:.1f} rad/s")
            
            # Differential drive: left backward, right forward -> counter-clockwise
            for i in [0, 2]:  # Left wheels
                p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                       targetVelocity=-angular_vel, force=5000)
            for i in [1, 3]:  # Right wheels
                p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                       targetVelocity=angular_vel, force=5000)
        
        elif mode == 'search_approach':
            # Navigate toward table while searching for red cylinder
            target = control_commands['target']
            pose = control_commands['pose']
            lidar = control_commands.get('lidar')
            
            dx = target[0] - pose[0]
            dy = target[1] - pose[1]
            dist = np.hypot(dx, dy)
            
            angle_to_target = np.arctan2(dy, dx)
            heading_error = angle_to_target - pose[2]
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            
            forward_vel = np.clip(2.0 * dist, -self.max_linear_speed, self.max_linear_speed)
            angular_vel = np.clip(4.0 * heading_error, -self.max_angular_speed, self.max_angular_speed)

            
            # M4: Lidar obstacle avoidance
            forward_vel, avoidance_turn = self._get_lidar_obstacle_avoidance(
                lidar, forward_vel, robot_pose=pose)
            angular_vel += avoidance_turn

            max_turn = abs(forward_vel) + 0.1
            angular_vel = np.clip(angular_vel, -max_turn, max_turn)


            forward_vel = np.clip(forward_vel, -self.max_linear_speed, self.max_linear_speed)
            angular_vel = np.clip(angular_vel, -self.max_angular_speed, self.max_angular_speed)
            
            wheel_limit = self.max_linear_speed + self.max_angular_speed
            left_vel = np.clip(forward_vel - angular_vel, -wheel_limit, wheel_limit)
            right_vel = np.clip(forward_vel + angular_vel, -wheel_limit, wheel_limit)


            print(f"Look Here -------------- {forward_vel} {angular_vel} {left_vel} {right_vel}")

            
            if self.step_counter % 240 == 0:
                print(f"[Act] SEARCH_APPROACH: dist={dist:.2f}m, heading={np.degrees(heading_error):.0f} deg")
            
            for i in [0, 2]:
                p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                       targetVelocity=left_vel, force=5000)
            for i in [1, 3]:
                p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                       targetVelocity=right_vel, force=5000)
        
        elif mode == 'search_orbit':
            # Orbit around the table at a fixed radius to scan all sides
            table_pos = control_commands['table_pos']
            pose = control_commands['pose']
            orbit_radius = control_commands.get('orbit_radius', 2.0)
            lidar = control_commands.get('lidar')
            
            dx = pose[0] - table_pos[0]
            dy = pose[1] - table_pos[1]
            current_dist = np.hypot(dx, dy)
            
            # Desired tangent direction (counter-clockwise orbit)
            angle_from_table = np.arctan2(dy, dx)
            tangent_angle = angle_from_table + np.pi / 2  # perpendicular, CCW
            
            # Radial correction to maintain orbit radius
            radial_error = current_dist - orbit_radius
            correction_angle = angle_from_table + np.pi  # toward table
            
            # Blend tangent and radial correction
            desired_angle = tangent_angle + 0.5 * radial_error
            heading_error = desired_angle - pose[2]
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            
            forward_vel = self.max_linear_speed
            angular_vel = 4.0 * heading_error
            
            # M4: Lidar obstacle avoidance
            forward_vel, avoidance_turn = self._get_lidar_obstacle_avoidance(
                lidar, forward_vel, robot_pose=pose)
            angular_vel += avoidance_turn

            forward_vel = np.clip(forward_vel, -self.max_linear_speed, self.max_linear_speed)
            angular_vel = np.clip(angular_vel, -self.max_angular_speed, self.max_angular_speed)
            
            wheel_limit = self.max_linear_speed + self.max_angular_speed
            left_vel = np.clip(forward_vel - angular_vel, -wheel_limit, wheel_limit)
            right_vel = np.clip(forward_vel + angular_vel, -wheel_limit, wheel_limit)
            
            if self.step_counter % 240 == 0:
                print(f"[Act] SEARCH_ORBIT: r={current_dist:.2f}m (target={orbit_radius:.1f}m)")
            
            for i in [0, 2]:
                p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                       targetVelocity=left_vel, force=5000)
            for i in [1, 3]:
                p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                       targetVelocity=right_vel, force=5000)
                                       
        elif mode in ['navigate', 'approach']:
            # M6: Navigate using differential drive with PID
            target = control_commands['target']
            pose = control_commands['pose']
            lidar = control_commands.get('lidar')
            
            dx = target[0] - pose[0]
            dy = target[1] - pose[1]
            dist = np.hypot(dx, dy)
            
            angle_to_target = np.arctan2(dy, dx)
            heading_error = angle_to_target - pose[2]
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))


            forward_vel = self.nav_pid_dist.compute(dist, self.dt)
            angular_vel = self.nav_pid_angle.compute(heading_error, self.dt)
        
            
            # M4: Lidar obstacle avoidance (relaxed near table during approach)
            relaxed = control_commands.get('relaxed_avoidance', False)
            forward_vel, avoidance_turn = self._get_lidar_obstacle_avoidance(
                lidar, forward_vel, relaxed=relaxed, robot_pose=pose)
            angular_vel += avoidance_turn

            forward_vel = np.clip(
                forward_vel,
                -self.max_linear_speed,
                self.max_linear_speed
            )

            angular_vel = np.clip(
                angular_vel,
                -self.max_angular_speed,
                self.max_angular_speed
            )
            
            left_vel = forward_vel - angular_vel
            right_vel = forward_vel + angular_vel

            wheel_limit = self.max_linear_speed + self.max_angular_speed
            left_vel = np.clip(forward_vel - angular_vel, -wheel_limit, wheel_limit)
            right_vel = np.clip(forward_vel + angular_vel, -wheel_limit, wheel_limit)

            print(f"[DEBUG ACT] Mode={mode}, left_vel={left_vel:.2f}, right_vel={right_vel:.2f}")


            if self.step_counter % 240 == 0:
                print(f"[Act] {mode.upper()}: dist={dist:.2f}m, heading={np.degrees(heading_error):.0f} deg, "
                      f"fwd={forward_vel:.1f}, turn={angular_vel:.1f}")
            
            for i in [0, 2]:
                p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                       targetVelocity=left_vel, force=5000)
            for i in [1, 3]:
                p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                       targetVelocity=right_vel, force=5000)
                                       
        elif mode == 'grasp':
            # M6: Stop wheels, multi-phase arm control for grasping
            for i in self.wheel_joints:
                p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                       targetVelocity=0, force=5000)
            
            phase = control_commands.get('phase', 'close_gripper')
            approach_pos = control_commands['approach_pos']
            grasp_pos = control_commands['grasp_pos']
            orientation = control_commands['orientation']
            orn_quat = p.getQuaternionFromEuler(orientation)
            
            # Raise lift joint to bring arm near table height (all phases)
            if self.lift_joint_idx is not None:
                p.setJointMotorControl2(self.robot_id, self.lift_joint_idx,
                                       p.POSITION_CONTROL,
                                       targetPosition=0.3,  # Max lift
                                       force=100, maxVelocity=0.5)
            
            if phase == 'reach_above':
                # Phase 1: Open gripper, IK arm to position above target
                if self.step_counter % 120 == 0:
                    print(f"[Act] GRASP phase 1: reaching above target at {approach_pos}")
                grasp_object(self.robot_id, approach_pos, orn_quat,
                             arm_joints=self.arm_joints if self.arm_joints else None,
                             close_gripper=False)
            
            elif phase == 'reach_target':
                # Phase 2: IK arm down to grasp position, gripper still open
                if self.step_counter % 120 == 0:
                    print(f"[Act] GRASP phase 2: lowering to target at {grasp_pos}")
                grasp_object(self.robot_id, grasp_pos, orn_quat,
                             arm_joints=self.arm_joints if self.arm_joints else None,
                             close_gripper=False)
            
            elif phase == 'close_gripper':
                # Phase 3: Hold arm at grasp position and close gripper
                if self.step_counter % 120 == 0:
                    print("[Act] GRASP phase 3: closing gripper")
                grasp_object(self.robot_id, grasp_pos, orn_quat,
                             arm_joints=self.arm_joints if self.arm_joints else None,
                             close_gripper=True)
            
        elif mode == 'lift':
            # M6: Stop wheels, keep gripper closed, raise lift joint
            for i in self.wheel_joints:
                p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                       targetVelocity=0, force=5000)
            
            # Keep gripper closed
            for finger_idx in self.gripper_joints:
                joint_info = p.getJointInfo(self.robot_id, finger_idx)
                joint_name = joint_info[1].decode('utf-8')
                if 'left' in joint_name:
                    target_pos = -0.04
                else:
                    target_pos = 0.04
                p.setJointMotorControl2(self.robot_id, finger_idx,
                                       p.POSITION_CONTROL,
                                       targetPosition=target_pos, force=50)
            
            # Raise lift joint
            if self.lift_joint_idx is not None:
                p.setJointMotorControl2(self.robot_id, self.lift_joint_idx,
                                       p.POSITION_CONTROL,
                                       targetPosition=0.3,  # Max lift
                                       force=100, maxVelocity=0.5)
            
            if self.step_counter % 120 == 0:
                print("[Act] LIFT: Raising object")
            
        elif mode in ['idle', 'success', 'failure']:
            if mode == 'failure':
                # Back up to clear the table, stow arm, open gripper
                if self.step_counter % 240 == 0:
                    print("[Act] FAILURE: backing up, stowing arm, opening gripper")
                
                # Use lidar rear avoidance to prevent reversing into obstacles
                lidar = control_commands.get('lidar')
                reverse_vel = -self.max_linear_speed
                if lidar is not None:
                    reverse_vel, avoidance_turn = self._get_lidar_obstacle_avoidance(
                        lidar, reverse_vel, relaxed=False, robot_pose=None)
                else:
                    avoidance_turn = 0.0
                
                left_vel = reverse_vel - avoidance_turn
                right_vel = reverse_vel + avoidance_turn
                
                for i in [0, 2]:  # Left wheels
                    p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                           targetVelocity=left_vel, force=5000)
                for i in [1, 3]:  # Right wheels
                    p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                           targetVelocity=right_vel, force=5000)
                # Stow arm joints back to zero
                for joint_idx in self.arm_joints:
                    p.setJointMotorControl2(self.robot_id, joint_idx,
                                           p.POSITION_CONTROL,
                                           targetPosition=0.0, force=200,
                                           maxVelocity=1.0)
                # Lower lift
                if self.lift_joint_idx is not None:
                    p.setJointMotorControl2(self.robot_id, self.lift_joint_idx,
                                           p.POSITION_CONTROL,
                                           targetPosition=0.0, force=100,
                                           maxVelocity=0.5)
                # Open gripper
                for finger_idx in self.gripper_joints:
                    p.setJointMotorControl2(self.robot_id, finger_idx,
                                           p.POSITION_CONTROL,
                                           targetPosition=0.0, force=50)
            else:
                # Idle/Success: stop all wheels  
                for i in self.wheel_joints:
                    p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                           targetVelocity=0, force=1500)

    def apply_parameters(self, parameters):

        self.parameters = parameters.copy()

        def get_param(name, default=0.0):
            return parameters.get(name, default)

        self.nav_pid_dist = PIDController(
            np.clip(get_param("nav_kp"), 0.1, 3.0),
            get_param("nav_ki", 0),
            get_param("nav_kd", 0.05),
            setpoint=0.0
        )


        self.nav_pid_angle = PIDController(
            np.clip(get_param("angle_kp"), 0.5, 5.0),
            get_param("angle_ki", 0.0),
            get_param("angle_kd", 0.01)
        )

        self.arm_pid = PIDController(
            get_param("arm_kp", 1.2),
            get_param("arm_ki", 0.0),
            get_param("arm_kd", 0.05)
        )

        self.vision_threshold = get_param("vision_threshold", 0.5)
        self.max_linear_speed = np.clip(get_param("max_linear_speed", 0.5), 0.1, 1.0)
        self.max_angular_speed = np.clip(get_param("max_angular_speed", 1.0), 0.2, 2.0)

        print("[Learning] Parameters applied successfully")

    def reset(self):

        if p.isConnected():
            p.resetSimulation()
        else:
            p.connect(p.DIRECT)

        # rebuild world again
        robot_id, table_id, room_id, target_id = build_world(gui=False)

        self.robot_id = robot_id
        self.table_id = table_id
        self.room_id = room_id
        self.target_id = target_id

        self.target_position = None
        self.current_waypoint = None
        self.obstacles = []
        self.step_counter = 0

        self._initialize_world_knowledge()

        self.state_estimator = ParticleFilter(num_particles=500)
        self.action_planner.reset()
        self.fsm.reset()

        # ⭐ VERY IMPORTANT – reset learned controllers
        if hasattr(self, "nav_pid_dist"):
            self.nav_pid_dist.reset()

        if hasattr(self, "nav_pid_angle"):
            self.nav_pid_angle.reset()


def main():
    """Main execution loop with Sense-Think-Act cycle"""
    
    print("="*60)
    print("  IIS Cognitive Architecture - Navigate-to-Grasp Mission")
    print("  Integrating all 10 modules in Sense-Think-Act loop")
    print("="*60)
    
    # M2: Build world (hardware initialization)
    robot_id, table_id, room_id, target_id = build_world(gui=True)
    
    # M10: Create cognitive architecture
    cog_arch = CognitiveArchitecture(robot_id, table_id, room_id, target_id, parameters=DEFAULT_PARAMETERS)
    
    # Report initial state
    print("\n[Init] Robot at: (0.00, 0.00) - starting position")
    if cog_arch.table_position:
        print(f"[Init] Table at: ({cog_arch.table_position[0]:.2f}, {cog_arch.table_position[1]:.2f})")
    else:
        print("[Init] Table position unknown - will search")
    print("[Init] Mission: Navigate to table and grasp red cylinder\n")
    
    # Main Sense-Think-Act loop
    while p.isConnected():  # DO NOT TOUCH
        # ========== SENSE ==========
        sensor_data = cog_arch.sense()
        
        # ========== THINK ==========
        control_commands = cog_arch.think(sensor_data)
        
        # ========== ACT ==========
        cog_arch.act(control_commands)
        
        # Status output every ~1 second
        if cog_arch.step_counter % 240 == 0:
            pose = sensor_data['pose']
            state_name = cog_arch.fsm.state.name
            print(f"[t={cog_arch.step_counter/240:.0f}s] State: {state_name}, "
                  f"Pose: ({pose[0]:.2f}, {pose[1]:.2f}, {np.degrees(pose[2]):.0f} deg)")
        
        cog_arch.step_counter += 1
        
        p.stepSimulation()  # DO NOT TOUCH
        time.sleep(1./240.)  # DO NOT TOUCH


if __name__ == "__main__":
    main()
