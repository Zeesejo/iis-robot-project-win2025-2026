"""
Module 3: Sensor Preprocessing
Provides functions to retrieve and preprocess sensor data from the robot.
"""

import numpy as np
import cv2
import pybullet as p
from src.robot.sensor_wrapper import *

def get_link_id_by_name(body_id, link_name):
    """
    Find the link ID for a given link name.
    
    Args:
        body_id: PyBullet body ID
        link_name: Name of the link to find
        
    Returns:
        int: Link ID
        
    Raises:
        ValueError: If link not found
    """
    for i in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, i)
        name = info[12].decode("utf-8")
        if name == link_name:
            return i
    raise ValueError(f"Link {link_name} not found")

def get_sensor_data(robot_id, camera_id, lidar_id):
    """
    Get preprocessed sensor data from the robot.
    
    Args:
        robot_id: PyBullet robot body ID
        camera_id: Link ID for camera
        lidar_id: Link ID for lidar
        
    Returns:
        dict: Preprocessed sensor data including:
            - camera_rgb: RGB image
            - camera_depth: Preprocessed depth image
            - camera_mask: Segmentation mask
            - lidar: Preprocessed lidar data
            - joint_states: Robot joint states
            - imu: IMU data
    """
    # Get raw sensor data
    rgb, depth, mask = get_camera_image(robot_id, camera_id)
    lidar_data = get_lidar_data(robot_id, lidar_id)
    joint_states = get_joint_states(robot_id)
    imu_data = get_imu_data(robot_id)

    # Preprocess depth data
    depth = np.clip(depth, 0.0, 10.0)  # Clip to sensor range
    depth = cv2.GaussianBlur(depth, (5, 5), 0)  # Smooth out noise

    # Preprocess lidar data
    lidar_data = np.clip(lidar_data, 0.0, 10.0)  # Clip to sensor range
    lidar_data = np.convolve(lidar_data, np.ones(3)/3, mode='same')  # Simple moving average filter

    # Package sensor data
    sensor_data = {
        'camera_rgb': rgb,
        'camera_depth': depth,
        'camera_mask': mask,
        'lidar': lidar_data,
        'joint_states': joint_states,
        'imu': imu_data
    }

    return sensor_data

def get_sensor_id(body_id):
    """
    Get sensor link IDs for the robot.
    
    Args:
        body_id: PyBullet robot body ID
        
    Returns:
        tuple: (camera_id, lidar_id)
    """
    camera_id = get_link_id_by_name(body_id, "rgbd_camera_link")
    lidar_id = get_link_id_by_name(body_id, "lidar_link")
    
    return camera_id, lidar_id
