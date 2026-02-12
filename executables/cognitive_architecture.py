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
        self.learner = Learner()
        
        # M6: PID Controllers
        self.nav_pid = PIDController(Kp=2.5, Ki=0.0, Kd=0.1)
        
        # Robot configuration
        self.wheel_joints = [0, 1, 2, 3]  # Based on custom robot URDF
        self.arm_joints = None  # Will be auto-detected
        self.gripper_joints = None
        
        # Task state
        self.target_position = None
        self.table_position = None
        self.obstacles = []
        self.current_waypoint = None
        
        # Timing
        self.step_counter = 0
        self.dt = 1.0 / 240.0
        
        # Initialize robot knowledge
        self._initialize_world_knowledge()
        
    def _initialize_world_knowledge(self):
        """Initialize knowledge base with world state from initial map"""
        # Load initial map if it exists
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
                
            # Add obstacles to knowledge base
            if 'obstacles' in world_map:
                for i, obs in enumerate(world_map['obstacles']):
                    pos = obs['position']
                    color = self._rgba_to_color_name(obs['color_rgba'])
                    obj_id = f'obstacle{i+1}'
                    self.kb.add_position(obj_id, pos[0], pos[1], pos[2])
                    self.kb.add_detected_object(obj_id, 'static', color, pos)
                    self.obstacles.append(pos[:2])  # Store x,y for planning
                    
            print(f"[CogArch] Loaded {len(self.obstacles)} obstacles from initial map")
    
    def _rgba_to_color_name(self, rgba):
        """Convert RGBA to color name"""
        r, g, b, a = rgba
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
    
    # ==================== SENSE ====================
    def sense(self):
        """
        SENSE phase: Acquire sensor data and update state estimate.
        Returns sensor_data dict for use in THINK phase.
        """
        # M3: Get raw sensor data
        rgb, depth, mask = get_camera_image(self.robot_id)
        lidar = get_lidar_data(self.robot_id, num_rays=36)
        imu = get_imu_data(self.robot_id)
        joint_states = get_joint_states(self.robot_id)
        
        # M5: Update state estimate with sensor fusion
        sensor_data = {
            'imu': imu,
            'lidar': lidar,
            'joint_states': joint_states
        }
        
        # Get wheel velocities for prediction step
        wheel_vels = [joint_states[f'wheel_{i}']['velocity'] 
                      if f'wheel_{i}' in joint_states else 0.0 
                      for i in range(4)]
        
        self.state_estimator.predict(wheel_vels, self.dt)
        self.state_estimator.measurement_update(sensor_data)
        estimated_pose = self.state_estimator.estimate_pose()
        
        # M4: Perception - Detect objects
        if rgb is not None and self.step_counter % 60 == 0:  # Process vision every 60 steps
            rgb_array = np.reshape(rgb, (240, 320, 4)).astype(np.uint8)
            bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)
            
            detections = detect_objects_by_color(bgr, min_area=100)
            
            # Look for red target
            for det in detections:
                if det['color'] == 'red' and self.target_position is None:
                    # Found target! Estimate its 3D position
                    # Simple estimation: use depth at bbox center
                    x, y, w, h = det['bbox']
                    cx, cy = x + w//2, y + h//2
                   
                    if depth is not None:
                        depth_array = np.array(depth)
                        depth_value = depth_array[cy, cx] if cy < depth_array.shape[0] and cx < depth_array.shape[1] else 1.0
                        
                        # Rough 3D position estimation (needs camera calibration for accuracy)
                        self.target_position = [estimated_pose[0] + depth_value, 
                                              estimated_pose[1], 
                                              0.7]  # Approximate table height
                        
                        # Add to knowledge base
                        self.kb.add_position('target', *self.target_position)
                        print(f"[CogArch] Target detected at: {self.target_position}")
        
        # Compile sensor data for THINK phase
        return {
            'pose': estimated_pose,
            'rgb': rgb,
            'depth': depth,
            'lidar': lidar,
            'imu': imu,
            'joint_states': joint_states,
            'target_detected': self.target_position is not None,
            'target_position': self.target_position
        }
    
    # ==================== THINK ====================
    def think(self, sensor_data):
        """
        THINK phase: Process sensor data, update knowledge, plan actions.
        Returns control_commands dict for ACT phase.
        """
        pose = sensor_data['pose']
        
        # M8: Query knowledge base for decision making
        graspable_objects = self.kb.query_graspable()
        target_pos = self.kb.query_position('target')
        
        # Override with sensor data if available
        if sensor_data['target_detected']:
            target_pos = sensor_data['target_position']
        
        # M7: Update FSM with sensor feedback
        fsm_sensor_data = {
            'target_visible': sensor_data['target_detected'],
            'target_position': target_pos,
            'distance_to_target': np.hypot(target_pos[0] - pose[0], target_pos[1] - pose[1]) if target_pos else float('inf'),
            'collision_detected': False,  # Can be enhanced with touch sensor
            'object_grasped': False,  # Can be enhanced with force sensor
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
            # Rotate slowly to search for target
            control_commands = {
                'mode': 'search_rotate',
                'angular_vel': 0.5
            }
            
        elif self.fsm.state == RobotState.NAVIGATE:
            # Navigate to target using action planner
            if target_pos and self.current_waypoint is None:
                # M7: Create navigation plan
                plan = self.action_planner.create_plan(
                    pose[:2], target_pos[:2], self.obstacles
                )
                self.current_waypoint = self.action_planner.get_next_waypoint()
                
            if self.current_waypoint:
                control_commands = {
                    'mode': 'navigate',
                    'target': self.current_waypoint,
                    'pose': pose
                }
                
                # Check if waypoint reached
                dist = np.hypot(self.current_waypoint[0] - pose[0], 
                              self.current_waypoint[1] - pose[1])
                if dist < 0.3:
                    self.action_planner.advance_waypoint()
                    self.current_waypoint = self.action_planner.get_next_waypoint()
                    
        elif self.fsm.state == RobotState.APPROACH:
            # Fine approach to grasping position
            if target_pos:
                control_commands = {
                    'mode': 'approach',
                    'target': target_pos[:2],
                    'pose': pose
                }
                
        elif self.fsm.state == RobotState.GRASP:
            # M7: Plan and execute grasp
            if target_pos:
                grasp_plan = self.grasp_planner.plan_grasp(target_pos)
                control_commands = {
                    'mode': 'grasp',
                    'approach_pos': grasp_plan['approach_pos'],
                    'grasp_pos': grasp_plan['grasp_pos'],
                    'orientation': grasp_plan['orientation'],
                    'gripper': 'close'
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
            control_commands = {
                'mode': 'failure',
                'gripper': 'open'
            }
        
        return control_commands
    
    # ==================== ACT ====================
    def act(self, control_commands):
        """
        ACT phase: Execute motion commands on the robot.
        """
        mode = control_commands.get('mode', 'idle')
        
        if mode == 'search_rotate':
            # Rotate in place to search
            angular_vel = control_commands.get('angular_vel', 0.5)
            # Left wheels forward, right wheels backward
            for i in [0, 2]:  # Left wheels
                p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                       targetVelocity=angular_vel, force=1500)
            for i in [1, 3]:  # Right wheels
                p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                       targetVelocity=-angular_vel, force=1500)
                                       
        elif mode in ['navigate', 'approach']:
            # M6: Navigate using differential drive with PID
            target = control_commands['target']
            pose = control_commands['pose']
            
            # Calculate heading to target
            dx = target[0] - pose[0]
            dy = target[1] - pose[1]
            dist = np.hypot(dx, dy)
            
            angle_to_target = np.arctan2(dy, dx)
            heading_error = angle_to_target - pose[2]
            # Normalize to [-pi, pi]
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            
            # Differential drive control
            Kp_dist = 2.5
            Kp_angle = 5.0
            base_speed = 5.0
            
            forward_vel = np.clip(Kp_dist * dist, -base_speed, base_speed)
            angular_vel = Kp_angle * heading_error
            
            left_vel = forward_vel - angular_vel
            right_vel = forward_vel + angular_vel
            
            # Apply to wheels
            for i in [0, 2]:  # Left wheels
                p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                       targetVelocity=left_vel, force=1500)
            for i in [1, 3]:  # Right wheels  
                p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                       targetVelocity=right_vel, force=1500)
                                       
        elif mode == 'grasp':
            # M6: Use IK to move arm and close gripper
            approach_pos = control_commands['approach_pos']
            grasp_pos = control_commands['grasp_pos']
            orientation = control_commands['orientation']
            
            # Convert Euler to quaternion
            orn_quat = p.getQuaternionFromEuler(orientation)
            
            # Use grasp_object function from motion_control
            success = grasp_object(self.robot_id, grasp_pos, orn_quat, 
                                  arm_joints=self.arm_joints, close_gripper=True)
            
        elif mode == 'lift':
            # Lift the grasped object
            lift_height = control_commands.get('lift_height', 0.2)
            # Keep gripper closed and move arm up
            # (Implementation depends on arm configuration)
            pass
            
        elif mode in ['idle', 'success', 'failure']:
            # Stop all wheels
            for i in self.wheel_joints:
                p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                       targetVelocity=0, force=1500)
    
    def save_camera_frame(self, rgb, depth, step):
        """Save camera data periodically"""
        if step % 480 == 0:  # Every 2 seconds
            rgb_array = np.reshape(rgb, (240, 320, 4)).astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)
            
            depth_normalized = cv2.normalize(np.array(depth), None, 0, 255, cv2.NORM_MINMAX)
            depth_uint8 = depth_normalized.astype(np.uint8)
            
            cv2.imwrite(f"frame_{step}_rgb.png", rgb_bgr)
            cv2.imwrite(f"frame_{step}_depth.png", depth_uint8)


def main():
    """Main execution loop with Sense-Think-Act cycle"""
    
    print("="*60)
    print("  IIS Cognitive Architecture - Navigate-to-Grasp Mission")
    print("  Integrating all 10 modules in Sense-Think-Act loop")
    print("="*60)
    
    # M2: Build world (hardware initialization)
    robot_id, table_id, room_id, target_id = build_world(gui=False)
    
    # M10: Create cognitive architecture
    cog_arch = CognitiveArchitecture(robot_id, table_id, room_id, target_id)
    
    # Get actual positions for debugging
    robot_pos, _ = p.getBasePositionAndOrientation(robot_id)
    table_pos, _ = p.getBasePositionAndOrientation(table_id)
    target_pos, _ = p.getBasePositionAndOrientation(target_id)
    
    print(f"\n[Init] Robot at: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})")
    print(f"[Init] Table at: ({table_pos[0]:.2f}, {table_pos[1]:.2f})")
    print(f"[Init] Target at: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})")
    print(f"[Init] Mission: Navigate to table and grasp red cylinder\n")
    
    # Main Sense-Think-Act loop
    while p.isConnected():  # DO NOT TOUCH
        # ========== SENSE ==========
        sensor_data = cog_arch.sense()
        
        # ========== THINK ==========
        control_commands = cog_arch.think(sensor_data)
        
        # ========== ACT ==========
        cog_arch.act(control_commands)
        
        # Save camera frames periodically
        if sensor_data['rgb'] is not None:
            cog_arch.save_camera_frame(sensor_data['rgb'], sensor_data['depth'], 
                                      cog_arch.step_counter)
        
        # Status output
        if cog_arch.step_counter % 60 == 0:
            pose = sensor_data['pose']
            print(f"[t={cog_arch.step_counter}] State: {cog_arch.fsm.state.name}, "
                  f"Pose: ({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f})")
        
        cog_arch.step_counter += 1
        
        p.stepSimulation()  # DO NOT TOUCH
        time.sleep(1./240.)  # DO NOT TOUCH


if __name__ == "__main__":
    main()
