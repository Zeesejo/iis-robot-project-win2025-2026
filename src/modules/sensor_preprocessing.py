import numpy as np
from src.robot.sensor_wrapper import *
import cv2

def get_sensor_data(robot_id, camera_id, lidar_id):
    rgb, depth, mask = get_camera_image(robot_id, camera_id)
    lidar_data = get_lidar_data(robot_id, lidar_id)

    depth = np.clip(depth, 0.0, 10.0)  # Clip to sensor range
    depth = cv2.GaussianBlur(depth, (5, 5), 0)  # Smooth out noise

    lidar_data = np.clip(lidar_data, 0.0, 10.0)  # Clip to sensor range
    lidar_data = np.convolve(lidar_data, np.ones(3)/3, mode='same')  # Simple moving average filter

    joint_states = get_joint_states(robot_id)
    imu_data = get_imu_data(robot_id)

    sensor_data = {
        'camera_rgb': rgb,
        'camera_depth': depth,
        'camera_mask': mask,
        'lidar': lidar_data,
        'joint_states': joint_states,
        'imu': imu_data
    }

    return sensor_data