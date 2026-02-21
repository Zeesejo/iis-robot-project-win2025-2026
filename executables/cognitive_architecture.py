"""
Module 10: Cognitive Architecture - SENSE-THINK-ACT Loop
Integrates all 10 modules for autonomous navigate-to-grasp mission.

This is the main executive controller that combines:
- M1: Task specification (defined in README)
- M2: URDF robot hardware
- M3: Sensor preprocessing (sensor_preprocessing.py wrapping sensor_wrapper.py)
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

# M3: Sensor Preprocessing (wraps sensor_wrapper with noise filtering)
from src.modules.sensor_preprocessing import get_sensor_data, get_sensor_id
from src.robot.sensor_wrapper import get_joint_states  # for gripper contact check

# M4: Perception
from src.modules.perception import detect_objects_by_color, RANSAC_Segmentation

# M5: State Estimation
from src.modules.state_estimation import state_estimate, initialize_state_estimator

# M6: Motion Control
from src.modules.motion_control import PIDController, move_to_goal, grasp_object

# M7: Action Planning
from src.modules.fsm import RobotFSM, RobotState
from src.modules.action_planning import get_action_planner, get_grasp_planner

# M8: Knowledge Representation
from src.modules.knowledge_reasoning import get_knowledge_base

# M9: Learning
from src.modules.learning import Learner, DEFAULT_PARAMETERS as LEARNING_DEFAULTS

# Robot physical constants (must match URDF and state_estimation)
WHEEL_RADIUS = 0.1
WHEEL_BASELINE = 0.45
CAMERA_HEIGHT = 0.55    # Base spawn (0.1) + torso origin (0.3) + camera Z (0.15)
CAMERA_FORWARD = 0.25   # Camera is 0.25m forward of robot center on torso
DEPTH_NEAR = 0.1
DEPTH_FAR = 10.0


class CognitiveArchitecture:
    """
    Main cognitive architecture implementing the Sense-Think-Act loop.
    """
    
    def __init__(self, robot_id, table_id, room_id, target_id):
        # Robot IDs from PyBullet
        self.robot_id = robot_id
        self.table_id = table_id
        self.room_id = room_id
        self.target_id = target_id
        
        # M5: State Estimator (via state_estimate function)
        initialize_state_estimator()
        
        # M3: Get sensor link IDs from URDF
        self.sensor_camera_id, self.sensor_lidar_id = get_sensor_id(self.robot_id)
        
        # M7: FSM for high-level control
        self.fsm = RobotFSM()
        
        # M7: Action Planners
        self.action_planner = get_action_planner()
        self.grasp_planner = get_grasp_planner()
        
        # M8: Knowledge Base (backed by Dynamic_KB.pl via PySwip)
        self.kb = get_knowledge_base()
        
        # M9: Learner (for parameter optimization)
        # Pass self as robot so learner can call run_episode if needed
        self.learner = Learner(robot=None)  # No live robot connection during main loop
        
        # M9: Use learned or default PID parameters
        best_params = self.learner.get_best_parameters()
        nav_kp = best_params.get('nav_kp', LEARNING_DEFAULTS['nav_kp'])
        nav_ki = best_params.get('nav_ki', LEARNING_DEFAULTS['nav_ki'])
        nav_kd = best_params.get('nav_kd', LEARNING_DEFAULTS['nav_kd'])
        
        # M6: PID Controllers (initialized from M9 learning parameters)
        self.nav_pid = PIDController(Kp=nav_kp, Ki=nav_ki, Kd=nav_kd)
        
        # Robot configuration - wheel joints [FL, FR, BL, BR]
        self.wheel_joints = [0, 1, 2, 3]
        self.wheel_names = ['fl_wheel_joint', 'fr_wheel_joint',
                            'bl_wheel_joint', 'br_wheel_joint']
        
        # Auto-detect special joint indices from URDF
        self.arm_joints = []
        self.gripper_joints = []
        self.lift_joint_idx = None
        self.camera_link_idx = None
        self._detect_robot_joints()
        
        # Task state
        self.target_position = None
        self.target_position_smoothed = None  # EMA-filtered target position
        self.target_detection_count = 0
        self.target_camera_bearing = 0.0  # Camera-relative bearing to target (rad)
        self.target_camera_depth = float('inf')  # Camera depth to target (m)
        self.table_position = None
        self.table_orientation = None
        self.table_size = None
        self.obstacles = []
        self.current_waypoint = None
        self.approach_standoff = None
        
        # Timing
        self.step_counter = 0
        self.dt = 1.0 / 240.0
        
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
            
            # Lift joint (only if movable - prismatic/revolute)
            if joint_name == 'lift_joint':
                joint_type = joint_info[2]
                if joint_type != p.JOINT_FIXED:
                    self.lift_joint_idx = i
                else:
                    self.lift_joint_idx = None  # Fixed in new URDF, can't actuate
            
            # Camera link
            if 'rgbd_camera' in link_name or 'camera' in link_name:
                self.camera_link_idx = i
            
            # Arm joints (revolute joints in the arm chain, skip fixed cosmetic joints)
            if joint_name in ['arm_base_joint', 'shoulder_joint', 'elbow_joint',
                              'wrist_pitch_joint', 'wrist_roll_joint']:
                joint_type = joint_info[2]
                if joint_type != p.JOINT_FIXED:
                    self.arm_joints.append(i)
        
        print(f"[CogArch] Detected: {len(self.gripper_joints)} gripper joints, "
              f"{len(self.arm_joints)} arm joints, "
              f"lift_joint={self.lift_joint_idx}, camera_link={self.camera_link_idx}")
        
    def _initialize_world_knowledge(self):
        """Initialize knowledge base with world state from initial map.
        Updates Dynamic_KB.pl positions via Prolog update_position/4."""
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
                    obj_id = f'obstacle{i}'
                    self.kb.add_position(obj_id, pos[0], pos[1], pos[2])
                    self.kb.add_detected_object(obj_id, 'static', color, pos)
                    self.obstacles.append(pos[:2])
                    
            print(f"[CogArch] Loaded {len(self.obstacles)} obstacles from initial map")
            
            # M8: Log KB state
            try:
                known_objects = self.kb.objects()
                pickable = self.kb.pickable_objects()
                print(f"[CogArch] KB objects: {known_objects}")
                print(f"[CogArch] KB pickable: {pickable}")
            except Exception as e:
                print(f"[CogArch] KB query info: {e}")
    
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
        standoff_dist = 0.65  # meters from target center (table half-width 0.4 + robot 0.2 + margin 0.05)
        
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
        # M3: Get preprocessed sensor data via sensor_preprocessing module
        preprocessed = get_sensor_data(self.robot_id, self.sensor_camera_id, self.sensor_lidar_id)
        rgb = preprocessed['camera_rgb']
        depth = preprocessed['camera_depth']
        lidar = preprocessed['lidar']
        imu = preprocessed['imu']
        joint_states = preprocessed['joint_states']
        
        # M5: Get wheel velocities from joint states using correct URDF joint names
        wheel_vels = []
        for name in self.wheel_names:
            if name in joint_states:
                wheel_vels.append(joint_states[name]['velocity'])
            else:
                wheel_vels.append(0.0)
        
        # M5: State estimation via state_estimate() function
        sensors_for_pf = {
            'imu': imu,
            'lidar': lidar,
            'joint_states': joint_states
        }
        control_inputs = {
            'wheel_left': (wheel_vels[0] + wheel_vels[2]) / 2.0,   # avg of FL + BL
            'wheel_right': (wheel_vels[1] + wheel_vels[3]) / 2.0,  # avg of FR + BR
        }
        estimated_pose = state_estimate(sensors_for_pf, control_inputs)
        
        # M8: Update robot position in Knowledge Base (Prolog)
        if self.step_counter % 50 == 0:
            self.kb.add_position('robot', 
                                float(estimated_pose[0]), 
                                float(estimated_pose[1]), 0.0)
        
        # M4: Perception - Detect objects every 10 steps
        if rgb is not None and self.step_counter % 10 == 0:
            rgb_array = np.reshape(rgb, (240, 320, 4)).astype(np.uint8)
            bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)
            
            detections = detect_objects_by_color(bgr, min_area=10)
            
            # Log detections periodically
            if self.step_counter % 240 == 0 and len(detections) > 0:
                colors_found = [d['color'] for d in detections]
                print(f"[Perception] Detected {len(detections)} objects: {colors_found}")
            
            # Look for red target - pick the CLOSEST (largest bounding box) red detection
            red_detections = [d for d in detections if d['color'] == 'red']
            # Sort by bbox area (larger = closer), take biggest
            red_detections.sort(key=lambda d: d['bbox'][2] * d['bbox'][3], reverse=True)
            for det in red_detections[:1]:  # Only process the single closest red detection
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
                        # Projection uses aspect=1.0 so both H and V FOV are 60 deg
                        # But image is 4:3, so fx ≠ fy (non-square pixels)
                        fx = (320 / 2.0) / np.tan(np.deg2rad(60 / 2.0))   # horizontal
                        fy = (240 / 2.0) / np.tan(np.deg2rad(60 / 2.0))   # vertical
                        cx, cy_cam = 160.0, 120.0
                        
                        # Convert pixel + depth to camera coordinates
                        cam_x = (center_x - cx) * true_depth / fx
                        cam_y = (center_y - cy_cam) * true_depth / fy
                        cam_z = true_depth  # forward distance
                        
                        # Transform to world frame
                        robot_x, robot_y, robot_theta = estimated_pose
                        cos_t = math.cos(robot_theta)
                        sin_t = math.sin(robot_theta)
                        
                        # Camera frame to robot body frame (no tilt, camera level on torso):
                        #   robot_forward(+X) = cam_z (depth) + camera forward offset
                        #   robot_left(+Y)    = -cam_x (camera right is robot -Y)
                        robot_body_x = cam_z + CAMERA_FORWARD  # camera is 0.25m ahead of robot center
                        robot_body_y = -cam_x
                        
                        world_x = robot_x + robot_body_x * cos_t - robot_body_y * sin_t
                        world_y = robot_y + robot_body_x * sin_t + robot_body_y * cos_t
                        # Height: camera is at CAMERA_HEIGHT, cam_y positive = object lower
                        world_z = CAMERA_HEIGHT - cam_y
                        
                        new_target = [world_x, world_y, world_z]
                        
                        # Filter: reject detections that jump > 2m from smoothed estimate
                        accept = True
                        if self.target_position_smoothed is not None:
                            jump = np.hypot(new_target[0] - self.target_position_smoothed[0],
                                            new_target[1] - self.target_position_smoothed[1])
                            if jump > 2.0 and self.target_detection_count > 5:
                                accept = False  # Likely false positive
                        
                        if accept:
                            self.target_detection_count += 1
                            # Store camera-relative bearing and depth for reactive steering
                            self.target_camera_bearing = math.atan2(-(center_x - cx), fx)
                            self.target_camera_depth = true_depth
                            # Exponential moving average for stability
                            alpha = 0.4 if self.target_detection_count > 3 else 0.8
                            if self.target_position_smoothed is None:
                                self.target_position_smoothed = list(new_target)
                            else:
                                for k in range(3):
                                    self.target_position_smoothed[k] = (
                                        alpha * new_target[k]
                                        + (1 - alpha) * self.target_position_smoothed[k]
                                    )
                            self.target_position = list(self.target_position_smoothed)
                            
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
            'target_camera_bearing': self.target_camera_bearing,
            'target_camera_depth': self.target_camera_depth,
            'gripper_contact': gripper_contact
        }
    
    # ==================== THINK ====================
    def think(self, sensor_data):
        """
        THINK phase: Process sensor data, update knowledge, plan actions.
        Uses M8 Knowledge Base for reasoning and M7 FSM for state management.
        Returns control commands for ACT phase.
        """
        pose = sensor_data['pose']
        
        # M8: Query knowledge base for target position
        target_pos = self.kb.query_position('target')
        
        # Override with sensor data if available (more recent)
        if sensor_data['target_detected']:
            target_pos = sensor_data['target_position']
        
        # M8: Check if target is the goal object (Prolog reasoning)
        if target_pos and self.step_counter % 240 == 0:
            is_goal = self.kb.is_goal_object('target')
            can_grasp = self.kb.check_can_grasp()
            if is_goal:
                print(f"[M8-KB] Target confirmed as goal object (red)")
            if can_grasp:
                print(f"[M8-KB] Prolog confirms: robot can grasp target")
        
        # Calculate 2D horizontal distance to target
        if target_pos:
            dx = target_pos[0] - pose[0]
            dy = target_pos[1] - pose[1]
            distance_2d = np.sqrt(dx**2 + dy**2)
            
            # Compute standoff as soon as we know the target
            if self.approach_standoff is None:
                self.approach_standoff = self._compute_approach_standoff(
                    target_pos, pose
                )
                print(f"[CogArch] Computed approach standoff: "
                      f"({self.approach_standoff[0]:.2f}, {self.approach_standoff[1]:.2f})")
            
            # Use camera depth for FSM distance during NAVIGATE and APPROACH
            # Camera depth is a direct measurement - no state estimation drift
            cam_depth = sensor_data.get('target_camera_depth', float('inf'))
            if cam_depth < 10.0 and self.fsm.state in (RobotState.NAVIGATE, RobotState.APPROACH):
                distance_for_fsm = cam_depth
            elif self.approach_standoff is not None and self.fsm.state == RobotState.NAVIGATE:
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
                
                if table_dist < 2.0:
                    # Close to table: orbit around it to scan all sides
                    control_commands = {
                        'mode': 'search_orbit',
                        'table_pos': self.table_position[:2],
                        'pose': pose,
                        'orbit_radius': 2.0,
                        'lidar': sensor_data['lidar']
                    }
                else:
                    control_commands = {
                        'mode': 'search_approach',
                        'target': self.table_position[:2],
                        'pose': pose,
                        'angular_vel': 2.0,
                        'lidar': sensor_data['lidar']
                    }
            else:
                control_commands = {
                    'mode': 'search_rotate',
                    'angular_vel': 3.0
                }
            
        elif self.fsm.state == RobotState.NAVIGATE:
            cam_depth = sensor_data.get('target_camera_depth', float('inf'))
            # Use relaxed avoidance when target is visible nearby (near the table)
            # This reduces table keepout from 0.6m to 0.2m so we can approach
            use_relaxed = cam_depth < 2.5 and sensor_data['target_detected']
            self._in_approach = False  # reset approach depth smoother flag
            
            # Waypoint navigation toward standoff position
            nav_goal = self.approach_standoff if self.approach_standoff else (
                target_pos[:2] if target_pos else None
            )
            if nav_goal and self.current_waypoint is None:
                self.action_planner.create_plan(
                    pose[:2], nav_goal, self.obstacles
                )
                self.current_waypoint = self.action_planner.get_next_waypoint()
                
            if self.current_waypoint:
                control_commands = {
                    'mode': 'navigate',
                    'target': self.current_waypoint,
                    'pose': pose,
                    'lidar': sensor_data['lidar'],
                    'relaxed_avoidance': use_relaxed
                }
                
                # Check if waypoint reached
                dist = np.hypot(self.current_waypoint[0] - pose[0],
                               self.current_waypoint[1] - pose[1])
                if dist < 0.3:
                    self.action_planner.advance_waypoint()
                    self.current_waypoint = self.action_planner.get_next_waypoint()
                    
        elif self.fsm.state == RobotState.APPROACH:
            # Reset depth smoother on first approach tick
            if not hasattr(self, '_in_approach') or not self._in_approach:
                self._approach_depth_smooth = float('inf')
                self._in_approach = True
            if target_pos:
                # During approach, use camera-relative visual servoing for reactive control
                control_commands = {
                    'mode': 'approach_visual',
                    'target': target_pos[:2],
                    'pose': pose,
                    'lidar': sensor_data['lidar'],
                    'relaxed_avoidance': True,
                    'camera_bearing': sensor_data.get('target_camera_bearing', 0.0),
                    'camera_depth': sensor_data.get('target_camera_depth', float('inf'))
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
            # Reset standoff, waypoint, and target smoothing so new ones are computed on retry
            self.approach_standoff = None
            self.current_waypoint = None
            self.target_position_smoothed = None
            self.target_detection_count = 0
            control_commands = {
                'mode': 'failure',
                'gripper': 'open',
                'lidar': sensor_data['lidar']
            }
        
        return control_commands
    
    # ==================== ACT ====================
    def _stow_arm(self):
        """Hold arm joints at stowed (zero) positions to prevent flailing during driving."""
        for joint_idx in self.arm_joints:
            p.setJointMotorControl2(self.robot_id, joint_idx,
                                   p.POSITION_CONTROL,
                                   targetPosition=0.0, force=500,
                                   maxVelocity=2.0)
    
    def act(self, control_commands):
        """
        ACT phase: Execute motion commands on the robot.
        """
        mode = control_commands.get('mode', 'idle')
        
        # Stow arm during all non-grasp/lift modes to prevent spinning
        if mode not in ('grasp', 'lift', 'success'):
            self._stow_arm()
        
        if mode == 'search_rotate':
            # Rotate in place to search for target
            angular_vel = control_commands.get('angular_vel', 3.0)
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
            
            forward_vel = min(3.0, 2.0 * dist)
            angular_vel = 4.0 * heading_error
            
            # M4: Lidar obstacle avoidance
            forward_vel, avoidance_turn = self._get_lidar_obstacle_avoidance(
                lidar, forward_vel, robot_pose=pose)
            angular_vel += avoidance_turn
            
            left_vel = forward_vel - angular_vel
            right_vel = forward_vel + angular_vel
            
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
            
            forward_vel = 2.0
            angular_vel = 4.0 * heading_error
            
            # M4: Lidar obstacle avoidance
            forward_vel, avoidance_turn = self._get_lidar_obstacle_avoidance(
                lidar, forward_vel, robot_pose=pose)
            angular_vel += avoidance_turn
            
            left_vel = forward_vel - angular_vel
            right_vel = forward_vel + angular_vel
            
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
            
            # Speed control
            if mode == 'approach':
                max_speed = 5.0
                kp_dist = 3.0
                kp_angle = 5.0
            else:
                max_speed = 5.0
                kp_dist = 4.0
                kp_angle = 5.0
            
            forward_vel = np.clip(kp_dist * dist, 0, max_speed)
            angular_vel = kp_angle * heading_error
            
            # M4: Lidar obstacle avoidance (relaxed near table during approach)
            relaxed = control_commands.get('relaxed_avoidance', False)
            forward_vel, avoidance_turn = self._get_lidar_obstacle_avoidance(
                lidar, forward_vel, relaxed=relaxed, robot_pose=pose)
            angular_vel += avoidance_turn
            
            left_vel = forward_vel - angular_vel
            right_vel = forward_vel + angular_vel
            
            if self.step_counter % 240 == 0:
                print(f"[Act] {mode.upper()}: dist={dist:.2f}m, heading={np.degrees(heading_error):.0f} deg, "
                      f"fwd={forward_vel:.1f}, turn={angular_vel:.1f}")
            
            for i in [0, 2]:
                p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                       targetVelocity=left_vel, force=5000)
            for i in [1, 3]:
                p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                       targetVelocity=right_vel, force=5000)
        
        elif mode == 'approach_visual':
            # M6: Reactive visual servoing - steer using camera pixel offset
            # Bypasses state estimation heading drift for approach accuracy
            camera_bearing = control_commands.get('camera_bearing', 0.0)
            camera_depth = control_commands.get('camera_depth', float('inf'))
            lidar = control_commands.get('lidar')
            pose = control_commands.get('pose')
            
            # Smooth depth to avoid jumps (e.g. detection flicker to far pixel)
            if not hasattr(self, '_approach_depth_smooth'):
                self._approach_depth_smooth = camera_depth
            # Only allow depth to INCREASE slowly (prevents overshoot on flicker)
            if camera_depth < self._approach_depth_smooth:
                self._approach_depth_smooth = camera_depth  # trust closer readings immediately
            else:
                self._approach_depth_smooth = 0.7 * self._approach_depth_smooth + 0.3 * camera_depth
            smooth_depth = self._approach_depth_smooth
            
            # Use camera bearing directly for steering (no state estimation needed)
            kp_bearing = 8.0
            max_speed = 4.0   # gentler max speed during approach
            kp_dist = 3.0
            
            forward_vel = np.clip(kp_dist * smooth_depth, 0.5, max_speed)
            angular_vel = kp_bearing * camera_bearing
            
            # Slow down progressively when close
            if smooth_depth < 1.0:
                forward_vel = np.clip(2.0 * smooth_depth, 0.3, 2.0)
            
            # M4: Lidar obstacle avoidance - only use when far from target
            # When close to target, lidar seeing the table pushes robot sideways
            if smooth_depth > 1.5:
                forward_vel, avoidance_turn = self._get_lidar_obstacle_avoidance(
                    lidar, forward_vel, relaxed=True, robot_pose=pose)
                angular_vel += avoidance_turn
            # When close, skip lidar avoidance to go straight at target
            
            left_vel = forward_vel - angular_vel
            right_vel = forward_vel + angular_vel
            
            if self.step_counter % 240 == 0:
                print(f"[Act] APPROACH_VISUAL: depth={camera_depth:.2f}m, bearing={np.degrees(camera_bearing):.0f} deg, "
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
                reverse_vel = -3.0
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


def main():
    """Main execution loop with Sense-Think-Act cycle"""
    
    print("="*60)
    print("  IIS Cognitive Architecture - Navigate-to-Grasp Mission")
    print("  Integrating all 10 modules in Sense-Think-Act loop")
    print("="*60)
    print("  M1:  Task Specification (README)")
    print("  M2:  URDF Robot Hardware & Environment")
    print("  M3:  Sensor Preprocessing (sensor_preprocessing.py)")
    print("  M4:  Perception (HSV color, SIFT, RANSAC)")
    print("  M5:  State Estimation (Particle Filter)")
    print("  M6:  Motion Control (PID, IK, Differential Drive)")
    print("  M7:  Action Planning (FSM + Waypoint Planner)")
    print("  M8:  Knowledge Representation (Prolog Dynamic_KB.pl)")
    print("  M9:  Learning (Parameter Optimization)")
    print("  M10: Cognitive Architecture (Sense-Think-Act)")
    print("="*60)
    
    # M2: Build world (hardware initialization)
    robot_id, table_id, room_id, target_id = build_world(gui=True)
    
    # M10: Create cognitive architecture
    cog_arch = CognitiveArchitecture(robot_id, table_id, room_id, target_id)
    
    # Report initial state
    print("\n[Init] Robot at: (0.00, 0.00) - starting position")
    if cog_arch.table_position:
        print(f"[Init] Table at: ({cog_arch.table_position[0]:.2f}, {cog_arch.table_position[1]:.2f})")
    else:
        print("[Init] Table position unknown - will search")
    
    # M8: Report KB state
    try:
        sensors = cog_arch.kb.sensors()
        caps = cog_arch.kb.robot_capabilities()
        print(f"[Init] M8 KB sensors: {sensors}")
        print(f"[Init] M8 KB robot capabilities: {caps}")
    except Exception:
        pass
    
    # M9: Report learning state
    best_params = cog_arch.learner.get_best_parameters()
    print(f"[Init] M9 Learning params: nav_kp={best_params.get('nav_kp', '?'):.2f}, "
          f"angle_kp={best_params.get('angle_kp', '?'):.2f}")
    
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
