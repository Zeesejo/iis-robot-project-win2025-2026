"""
M3 - Sensor Preprocessing
Acquires sensor data via sensor_wrapper and applies noise handling
using the Law of Large Numbers (averaging N samples to reduce sigma by sqrt(N)).
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robot'))
from sensor_wrapper import get_camera_image, get_lidar_data, get_joint_states, get_imu_data


N_SAMPLES = 5  # Number of readings to average (LLN: sigma -> sigma/sqrt(N))


def get_averaged_lidar(robot_id, sensor_link_id=-1, n=N_SAMPLES):
    """
    Average N LIDAR readings (Law of Large Numbers) to reduce noise.
    Returns a cleaner distance array.
    """
    readings = [np.array(get_lidar_data(robot_id, sensor_link_id)) for _ in range(n)]
    avg = np.mean(readings, axis=0)
    return avg.tolist()


def get_averaged_imu(robot_id, n=N_SAMPLES):
    """
    Average N IMU readings to reduce noise.
    Returns dict with averaged gyroscope and accelerometer data.
    """
    gyro_samples, accel_samples = [], []
    for _ in range(n):
        d = get_imu_data(robot_id)
        gyro_samples.append(d['gyroscope_data'])
        accel_samples.append(d['accelerometer_data'])
    return {
        'gyroscope_data': np.mean(gyro_samples, axis=0).tolist(),
        'accelerometer_data': np.mean(accel_samples, axis=0).tolist()
    }


def get_averaged_joints(robot_id, n=N_SAMPLES):
    """
    Average N joint readings to reduce encoder noise.
    Returns dict of joint_name -> averaged position & velocity.
    """
    all_readings = [get_joint_states(robot_id) for _ in range(n)]
    averaged = {}
    for joint_name in all_readings[0]:
        positions = [r[joint_name]['position'] for r in all_readings]
        velocities = [r[joint_name]['velocity'] for r in all_readings]
        averaged[joint_name] = {
            'index': all_readings[0][joint_name]['index'],
            'position': float(np.mean(positions)),
            'velocity': float(np.mean(velocities)),
            'applied_torque': all_readings[0][joint_name]['applied_torque']
        }
    return averaged


def get_preprocessed_camera(robot_id, sensor_link_id=-1):
    """
    Get camera image. Depth is already noise-reduced in wrapper.
    Additional: clip depth to valid range and normalize.
    """
    rgb, depth, mask = get_camera_image(robot_id, sensor_link_id)
    # Clip depth to sensor range [0.1, 10.0] and normalize
    depth_array = np.array(depth)
    depth_clipped = np.clip(depth_array, 0.0, 1.0)  # PyBullet returns 0-1 normalized
    return rgb, depth_clipped, mask


def preprocess_all(robot_id, lidar_link_id=-1, camera_link_id=-1):
    """Preprocess all sensors and return a unified sensor bundle."""
    return {
        'lidar': get_averaged_lidar(robot_id, lidar_link_id),
        'imu': get_averaged_imu(robot_id),
        'joints': get_averaged_joints(robot_id),
        'camera': get_preprocessed_camera(robot_id, camera_link_id)
    }
