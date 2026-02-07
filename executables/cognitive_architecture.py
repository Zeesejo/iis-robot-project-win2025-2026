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
Kp_dist = 1.0  # Force to move forward
Kp_angle = 120.0 # Force to turn
Kd = 1.0        # Damping to prevent oscillation
posi=0.3

def pid_to_target(robot_id, target_pos):
    
    # 4. ACT: Apply raw torque to wheels
    """
    # Torque-based Control
    for i in [2, 4]: # Left
        p.setJointMotorControl2(robot_id, i, p.TORQUE_CONTROL, force=left_torque)
    for i in [3, 5]: # Righ
        p.setJointMotorControl2(robot_id, i, p.TORQUE_CONTROL, force=right_torque)
    """
    """
    # Position-based Control
    posi=posi+0.1
    for i in [2, 4]: # Left
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=-posi, maxVelocity=20.0, force=1500.)
    for i in [3, 5]: # Righ
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=posi, maxVelocity=20.0, force=1500.)
    """
    # Velocity-based Control
    for i in [2, 4]: # Left
        p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, targetVelocity=-1.0, force=1500.)
    for i in [3, 5]: # Righ
        p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, targetVelocity=1.0, force=1500.)
    dist_error=1.0
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
    robot_id, table_id, room_id, target_id = build_world()
    
    print("Room initialized. Husky is at (-3, -3). Table is at (2, 2).")

    target = [2, 2, 0] # The table position
    
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
       
       # Inside your while loop:
       if step_counter % 240 == 0:  # Save once per second
           rgb, depth, mask = get_camera_image(robot_id)
           save_camera_data(rgb, depth, filename_prefix=f"frame_{step_counter}")
       step_counter=step_counter+1
    #    move_arm_to_coordinate(arm_id, target_id)  
       dist = pid_to_target(robot_id, target)
       print ('Distance: ', dist)
       if dist < 2:
             print("Target Reached!")
             # Apply braking torque
             for i in range(2, 6):
                 p.setJointMotorControl2(
                     bodyUniqueId=robot_id, 
                     jointIndex=i, 
                     controlMode=p.VELOCITY_CONTROL, 
                     targetVelocity=0, 
                     force=1000  # This "disables" the internal motor
                 )  
                 
       p.stepSimulation()  # DO NOT TOUCH
       time.sleep(1./240.) # DO NOT TOUCH
####################################################################################################


################ The Main Thread ###################################################################

if __name__ == "__main__":
    main()
