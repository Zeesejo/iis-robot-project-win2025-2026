"""
DO NOT MODIFY THIS FILE !!!
This file essentially contains noisy sensor wrappers
"""

import pybullet as p
import numpy as np

# Global noise settings - can be adjusted by the instructor
NOISE_MU = 0.0
NOISE_SIGMA = 0.01  # 1% noise standard deviation

def add_noise(data, mu=NOISE_MU, sigma=NOISE_SIGMA):
    """Helper to add Gaussian noise to scalars or arrays."""
    noise = np.random.normal(mu, sigma, np.shape(data))
    return data + noise

################# RGBD Camera Sensor ###################################### 

def get_camera_image(robot_id, sensor_link_id=-1):
    if sensor_link_id == -1:
        pos, orn = p.getBasePositionAndOrientation(robot_id)
    else:
        state = p.getLinkState(robot_id, sensor_link_id)
        pos, orn = state[0], state[1]

    rot_matrix = p.getMatrixFromQuaternion(orn)
    forward_vec = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
    up_vec = [rot_matrix[2], rot_matrix[5], rot_matrix[8]]
    
    target_pos = [pos[0] + forward_vec[0], pos[1] + forward_vec[1], pos[2] + forward_vec[2]]
    
    view_matrix = p.computeViewMatrix(pos, target_pos, up_vec)
    proj_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 10.0)

    width, height, rgb, depth, mask = p.getCameraImage(320, 240, view_matrix, proj_matrix)
    
    # Adding noise to the depth buffer (very common in RealSense/Kinect sensors)
    noisy_depth = add_noise(np.array(depth), sigma=0.005) 
    
    return rgb, noisy_depth, mask

################################### LIDAR ####################################################

def get_lidar_data(robot_id, sensor_link_id=-1, num_rays=36):
    if sensor_link_id == -1:
        pos, orn = p.getBasePositionAndOrientation(robot_id)
    else:
        state = p.getLinkState(robot_id, sensor_link_id)
        pos, orn = state[0], state[1]

    ray_start, ray_end = [], []
    ray_len = 5.0 
    _, _, yaw = p.getEulerFromQuaternion(orn)

    for i in range(num_rays):
        angle = yaw + (2.0 * np.pi * i) / num_rays
        ray_start.append(pos)
        ray_end.append([pos[0] + ray_len * np.cos(angle), pos[1] + ray_len * np.sin(angle), pos[2]])

    results = p.rayTestBatch(ray_start, ray_end)
    # Add noise to the hit fractions
    raw_distances = np.array([res[2] * ray_len for res in results])
    return add_noise(raw_distances, sigma=0.02).tolist()

###################################### JOINT Sensor ##################################################

def get_joint_states(robot_id):
    joint_data = {}
    num_joints = p.getNumJoints(robot_id)
    
    for i in range(num_joints):
        state = p.getJointState(robot_id, i)
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode('utf-8')
        
        # Adding noise to position (encoder slip/error) and velocity
        joint_data[joint_name] = {
            "index": i,
            "position": add_noise(state[0], sigma=0.002), 
            "velocity": add_noise(state[1], sigma=0.005),
            "applied_torque": state[3]
        }
    return joint_data

################################# IMU Sensor ##########################################################################

def get_imu_data(robot_id):
    lin_vel, ang_vel = p.getBaseVelocity(robot_id)
    _, orn = p.getBasePositionAndOrientation(robot_id)
    _, inv_orn = p.invertTransform([0,0,0], orn)
    
    local_lin_vel, _ = p.multiplyTransforms([0,0,0], inv_orn, lin_vel, [0,0,0,1])
    local_ang_vel, _ = p.multiplyTransforms([0,0,0], inv_orn, ang_vel, [0,0,0,1])

    # IMUs are notoriously noisy, especially the accelerometer
    return {
        "gyroscope_data": add_noise(np.array(local_ang_vel), sigma=0.01).tolist(),
        "accelerometer_data": add_noise(np.array(local_lin_vel), sigma=0.05).tolist(),
    }
