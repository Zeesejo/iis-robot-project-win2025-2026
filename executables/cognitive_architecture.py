"""
This file give you a broad overview on the pybullet ecosystem
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2
import sys
import os
from src.modules.state_estimation import state_estimate
from src.modules import sensor_preprocessing  # M3: Cleaned sensor data
from src.environment.world_builder import build_world

####################### Function Signature #################################
# [Keep all your existing helper functions unchanged - save_camera_data, pid_to_target, etc.]
def save_camera_data(rgb, depth, filename_prefix="frame"):
    rgb_array = np.reshape(rgb, (240, 320, 4)).astype(np.uint8)
    rgb_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)
    cv2.imwrite(f"{filename_prefix}_rgb.png", rgb_bgr)
    cv2.imwrite(f"{filename_prefix}_depth.png", depth_uint8)

def pid_to_target(robot_id, target_pos):
    for i in [2, 4]:  # Left wheels
        p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, targetVelocity=-1.0, force=1500.)
    for i in [3, 5]:  # Right wheels
        p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, targetVelocity=1.0, force=1500.)
    dist_error = 1.0
    return dist_error

def setup_simulation():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    room_id = p.loadURDF("../src/environment/room.urdf", [0, 0, 0], useFixedBase=True)
    table_id = p.loadURDF("table/table.urdf", basePosition=[2, 2, 0.0], useFixedBase=True)
    p.loadURDF("block.urdf", basePosition=[1, 0, 0.1], globalScaling=5.0)
    target_id = p.loadURDF("block.urdf", basePosition=[1.8, 1.8, 0.8], globalScaling=2.0)
    robot_id = p.loadURDF("husky/husky.urdf", basePosition=[-3, -3, 0.2])
    arm_id = p.loadURDF("kuka_iiwa/model.urdf", [2, 2, 0.625], useFixedBase=True)
    return robot_id, table_id, room_id, arm_id, target_id

############################################ The Main Function ###########################################
def main():
    robot_id, table_id, room_id, target_id = build_world()
    print("Room initialized. Husky is at (-3, -3). Table is at (2, 2).")
    
    target = [2, 2, 0]  # The table position
    
    # === NEW: Initialize sensor preprocessing IDs ===
    camera_id, lidar_id = sensor_preprocessing.get_sensor_id(robot_id)
    
    step_counter = 0
    
    ##################### MAIN SIMULATION LOOP ############################################
    while p.isConnected():  # DO NOT TOUCH
        
        # === M3: Get CLEANED sensor data ===
        sensor_data = sensor_preprocessing.get_sensor_data(robot_id, camera_id, lidar_id)
        
        # Extract wheel velocities for control inputs
        joint_states = sensor_data['joint_states']
        wheel_left_vel = joint_states['fl_wheel_joint']['velocity']
        wheel_right_vel = joint_states['fr_wheel_joint']['velocity']
        
        # Debug print
        print(f"🔍 WHEELS FL:{wheel_left_vel:.2f} FR:{wheel_right_vel:.2f}")
        
        # === M5: Package for particle filter ===
        sensors = {
            'camera_depth': sensor_data['camera_depth'],      # Filtered 2D depth image
            'lidar': np.array(sensor_data['lidar']),          # Filtered 36-ray lidar
            'imu': np.array(sensor_data['imu']['gyroscope_data'])  # Clean gyro
        }
        control_inputs = {
            'wheel_left': wheel_left_vel,
            'wheel_right': wheel_right_vel
        }
        
        # === M5: State estimation ===
        robot_pose = state_estimate(sensors, control_inputs)
        print(f"🧭 LIVE M5 POSE: {robot_pose}")
        
        # === Camera logging (M3/M4 support) ===
        if step_counter % 240 == 0:  # Save once per second
            save_camera_data(sensor_data['camera_rgb'], sensor_data['camera_depth'], 
                           filename_prefix=f"frame_{step_counter}")
        
        step_counter = step_counter + 1
        
        # === Navigation / PID ===
        dist = pid_to_target(robot_id, target)
        print('Distance: ', dist)
        if dist < 2:
            print("Target Reached!")
            # Apply braking
            for i in range(2, 6):
                p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, targetVelocity=0, force=1000)
        
        p.stepSimulation()  # DO NOT TOUCH
        time.sleep(1./240.)  # DO NOT TOUCH

if __name__ == "__main__":
    main()
