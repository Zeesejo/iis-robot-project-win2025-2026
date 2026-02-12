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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.robot.sensor_wrapper import *
from src.environment.world_builder import build_world

####################### Function Signature #################################

"""
p.setJointMotorControl2(
    bodyUniqueId,
    jointIndex,
    controlMode,
    targetPosition=0,
    targetVelocity=0,
    force=None,
    positionGain=0.1,
    velocityGain=1.0,
    maxVelocity=100,
    physicsClientId=0
)

p.changeDynamics(
    bodyUniqueId,
    linkIndex,
    mass=None,
    lateralFriction=None,
    spinningFriction=None,
    rollingFriction=None,
    restitution=None,
    linearDamping=None,
    angularDamping=None,
    contactStiffness=None,
    contactDamping=None,
    frictionAnchor=None,
    localInertiaDiagonal=None,
    ccdSweptSphereRadius=None,
    contactProcessingThreshold=None,
    activationState=None,
    jointLowerLimit=None,
    jointUpperLimit=None,
    jointLimitForce=None
)

bodyUniqueId = p.loadURDF(
    fileName,
    basePosition=[0, 0, 0],
    baseOrientation=[0, 0, 0, 1],
    useMaximalCoordinates=0,
    useFixedBase=0,
    flags=0,
    globalScaling=1.0,
    physicsClientId=0
)
"""


################################### Saving an rgbd image ########################

def save_camera_data(rgb, depth, filename_prefix="frame"):
    # 1. Process RGB (PyBullet returns RGBA)
    rgb_array = np.reshape(rgb, (240, 320, 4)).astype(np.uint8)
    rgb_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)
    
    # 2. Process Depth (Convert 0.0-1.0 float to 0-255 grayscale)
    # Note: Depth is already a NumPy array from your add_noise function
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)

    # 3. Save to disk
    cv2.imwrite(f"{filename_prefix}_rgb.png", rgb_bgr)
    cv2.imwrite(f"{filename_prefix}_depth.png", depth_uint8)

############################### Inverse Kinematics ##############################

def move_arm_to_coordinate(arm_id, target_id):
    # Joint 6 is the last joint (lbr_iiwa_joint_7)
    end_effector_index = 6 
    
    #0. get target pose
    target_pos, target_orn = p.getBasePositionAndOrientation(target_id)
    #target_orientation = p.getQuaternionFromEuler(target_orn)
    
    # 1. Compute Inverse Kinematics
    joint_poses = p.calculateInverseKinematics(
                  bodyUniqueId=arm_id,
                  endEffectorLinkIndex=6,
                  targetPosition=target_pos
    )
    
    # 2. Command all 7 joints
    for i in range(7):
        p.setJointMotorControl2(
            bodyIndex=arm_id, 
            jointIndex=i, 
            controlMode=p.POSITION_CONTROL, 
            targetPosition=joint_poses[i],
            force=200 # Newtons
        )


######################## Map between joint names and joint ids ####################
def get_joint_map(object_id):
    """
    Creates a dictionary mapping joint names to their integer indices.
    """
    joint_map = {}
    for i in range(p.getNumJoints(object_id)):
        info = p.getJointInfo(object_id, i)
        joint_name = info[1].decode('utf-8')
        joint_map[joint_name] = i
    return joint_map
    
    
###################### Getting the state of a specific link #######################    
def get_link_pos(robot_id, link_name, joint_map):
    """
    Returns the world position [x, y, z] of a specific link by name.
    """
    link_id = joint_map[link_name]
    # p.getLinkState returns a lot of data; index 0 is the world position
    state = p.getLinkState(robot_id, link_id)
    return state[0]

####################### Checking Collision #################################
def detect_collision(robot_id, target_list):
    """
    Checks if the robot is touching any object in the target_list.
    Returns (True, object_id) if a collision exists, else (False, None).
    """
    contacts = p.getContactPoints(bodyA=robot_id)
    
    for contact in contacts:
        hit_id = contact[2] # ID of the object collided with
        if hit_id in target_list:
            return True, hit_id
            
    return False, None
##############################################################################


############################### PID Controller ################################
# PID Constants (These are the "tuning knobs" for your students)
Kp_dist = 2.5   # Proportional gain for forward motion
Kp_angle = 5.0  # Proportional gain for turning
base_speed = 5.0  # Base forward speed

def pid_to_target(robot_id, target_pos):
    """
    Navigate robot to target using differential drive with PID control
    Returns the distance to target
    """
    # 1. SENSE: Get robot's current position and orientation
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    robot_x, robot_y, robot_z = pos
    
    # Get robot's yaw angle (rotation around z-axis)
    euler = p.getEulerFromQuaternion(orn)
    robot_yaw = euler[2]  # Z-axis rotation
    
    # 2. THINK: Calculate errors
    # Distance error
    dx = target_pos[0] - robot_x
    dy = target_pos[1] - robot_y
    dist_error = np.sqrt(dx**2 + dy**2)
    
    # Angle to target
    angle_to_target = np.arctan2(dy, dx)
    
    # Heading error (normalized to [-pi, pi])
    heading_error = angle_to_target - robot_yaw
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
    
    # 3. ACT: Compute wheel velocities using differential drive
    # Forward velocity based on distance
    forward_vel = Kp_dist * dist_error
    forward_vel = np.clip(forward_vel, -base_speed, base_speed)
    
    # Angular velocity based on heading error
    angular_vel = Kp_angle * heading_error
    
    # Differential drive: convert forward + angular to left/right wheel velocities
    left_vel = forward_vel - angular_vel
    right_vel = forward_vel + angular_vel
    
    # 4. Execute: Apply velocities to wheels  
    # Based on robot.urdf: joints 0-3 are the four wheels (fl, fr, bl, br)
    # Left wheels (fl=0, bl=2), Right wheels (fr=1, br=3)
    for i in [0, 2]:  # Left wheels
        p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, 
                                targetVelocity=left_vel, force=1500.)
    for i in [1, 3]:  # Right wheels
        p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, 
                                targetVelocity=right_vel, force=1500.)
    
    return dist_error
#############################################################################################################

########################################### Setting Up the Environment ######################################


def setup_simulation():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # 1. Spawn the Room
    room_id=p.loadURDF("../src/environment/room.urdf", [0, 0, 0], useFixedBase=True)

    # 2. Spawn the Target Table
    table_id = p.loadURDF("table/table.urdf", basePosition=[2, 2, 0.0], useFixedBase=True)
    #Overwrite table mass
    #p.changeDynamics(table_id, -1, mass=10.0)

    # 3. Spawn Obstacles (Random Blocks)
    # A heavy crate
    p.loadURDF("block.urdf", basePosition=[1, 0, 0.1], globalScaling=5.0) 
    # A simple block obstacle
    target_id=p.loadURDF("block.urdf", basePosition=[1.8, 1.8, 0.8], globalScaling=2.0)

    # 4. Spawn the Husky Robot
    robot_id = p.loadURDF("husky/husky.urdf", basePosition=[-3, -3, 0.2])
    
    # 5. Spawn the Gripper on the Table
    arm_id = p.loadURDF("kuka_iiwa/model.urdf", [2, 2, 0.625], useFixedBase=True)
    
    return robot_id, table_id, room_id, arm_id, target_id
##########################################################################################################

############################################ The Main Function ###########################################
def main():
    robot_id, table_id, room_id, target_id = build_world(gui=False)  # Use DIRECT mode for better performance in Docker
    
    # Debug: Print joint structure
    print(f"\n=== Robot Joint Structure ===")
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        print(f"Joint {i}: {info[1].decode('utf-8')} (Type: {info[2]})")
    print(f"=============================\n")
    
    # Get actual table position (randomized by build_world)
    table_pos, _ = p.getBasePositionAndOrientation(table_id)
    target = [table_pos[0], table_pos[1], 0]  # Target the table position
    
    robot_pos, _ = p.getBasePositionAndOrientation(robot_id)
    print(f"Room initialized. Robot at {robot_pos[:2]}, Table at {table_pos[:2]}")
    
    """
    # Run this ONCE before your simulation loop in p.TORQUE_CONTROL
    for i in [2, 3, 4, 5]:
      p.setJointMotorControl2(
        bodyUniqueId=robot_id, 
        jointIndex=i, 
        controlMode=p.VELOCITY_CONTROL, 
        targetVelocity=0, 
        force=0  # This "disables" the internal motor
      )
     """
    
    step_counter=0
    ##################### LOOP STRUCTURE ############################################
    while p.isConnected(): # DO NOT TOUCH
       
       # Save camera images every 2 seconds (reduce frequency to avoid lag)
       if step_counter % 480 == 0:
           rgb, depth, mask = get_camera_image(robot_id)
           save_camera_data(rgb, depth, filename_prefix=f"frame_{step_counter}")
       step_counter=step_counter+1
    #    move_arm_to_coordinate(arm_id, target_id)  
       dist = pid_to_target(robot_id, target)
       
       if step_counter % 60 == 0:  # Print every 60 steps (~0.25 seconds)
           print(f'Distance to target: {dist:.3f} meters')
       
       if dist < 0.5:  # Stop when within 0.5 meters of target
             if step_counter % 60 == 0:
                 print("Target Reached!")
             # Apply braking to all wheels (indices 0-3)
             for i in range(0, 4):
                 p.setJointMotorControl2(
                     bodyUniqueId=robot_id, 
                     jointIndex=i, 
                     controlMode=p.VELOCITY_CONTROL, 
                     targetVelocity=0, 
                     force=1000
                 )  
                 
       p.stepSimulation()  # DO NOT TOUCH
       time.sleep(1./240.) # DO NOT TOUCH
####################################################################################################


################ The Main Thread ###################################################################

if __name__ == "__main__":
    main()
