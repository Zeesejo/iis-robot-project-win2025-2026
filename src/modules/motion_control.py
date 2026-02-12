"""
Module 6: Motion Control & Planning
Provides PID controllers, navigation, manipulation, and path planning
"""

import pybullet as p
import pybullet_data
import time
import math

# --------------------------
# PID Controller Class
# --------------------------
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, current_value, dt):
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# --------------------------
# Prolog Path Planner (Optional)
# --------------------------
try:
    from pyswip import Prolog
    PROLOG_AVAILABLE = True
    prolog = Prolog()
    try:
        import os
        prolog_path = os.path.join(os.path.dirname(__file__), "path_planning.pl")
        prolog.consult(prolog_path)
        print("[Motion Control] Prolog path planner loaded successfully")
    except Exception as e:
        print(f"[Motion Control] Warning: Could not load path_planning.pl: {e}")
        PROLOG_AVAILABLE = False
except ImportError:
    print("[Motion Control] PySwip not installed. Prolog planning disabled.")
    PROLOG_AVAILABLE = False

def plan_path(start, goal):
    """
    Returns a list of waypoints from Prolog path planner (if available)
    Falls back to simple straight-line path
    """
    if PROLOG_AVAILABLE:
        try:
            query = f"plan_path({start}, {goal}, Path)."
            for sol in prolog.query(query):
                return sol["Path"]
        except:
            pass
    # Fallback: simple straight-line path
    return [goal]

# --------------------------
# Motion Control Functions
# --------------------------
def move_to_goal(robot_id, goal_pos, wheel_joints=[2, 3, 4, 5], dt=1./240.):
    """
    Move robot base using PID controller to goal position (x, y)
    
    Args:
        robot_id: PyBullet robot body ID
        goal_pos: Target position [x, y]
        wheel_joints: List of wheel joint indices (default: [2,3,4,5] for 4-wheel robot)
        dt: Time step
    
    Returns:
        Current distance to goal
    """
    # Get current position
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    x, y, _ = pos
    
    distance = math.hypot(goal_pos[0]-x, goal_pos[1]-y)
    
    # Simple proportional control for now (can upgrade to full PID)
    Kp = 1.0
    if distance > 0.05:
        # Calculate velocity towards goal
        vx = Kp * (goal_pos[0] - x)
        vy = Kp * (goal_pos[1] - y)
        
        # Convert to wheel velocities (differential drive)
        # Left wheels: indices 2, 4
        # Right wheels: indices 3, 5
        left_wheel_vel = -vx  # Negative for forward
        right_wheel_vel = vx
        
        # Apply velocities
        for i in [2, 4]:  # Left wheels
            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, 
                                   targetVelocity=left_wheel_vel, force=1500.)
        for i in [3, 5]:  # Right wheels
            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL,
                                   targetVelocity=right_wheel_vel, force=1500.)
    else:
        # Stop at goal
        for i in wheel_joints:
            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL,
                                   targetVelocity=0.0, force=1500.)
    
    return distance

def grasp_object(robot_id, target_pos, target_orient, arm_joints=None, close_gripper=True):
    """
    Move the robot arm to grasp an object using Inverse Kinematics
    
    Args:
        robot_id: PyBullet robot body ID
        target_pos: Target position [x, y, z]
        target_orient: Target orientation (quaternion)
        arm_joints: List of arm joint indices (auto-detect if None)
        close_gripper: Whether to close gripper after reaching target
    
    Returns:
        True if IK solution found, False otherwise
    """
    # Auto-detect arm joints from robot URDF structure
    if arm_joints is None:
        # Based on robot.urdf analysis:
        # - arm_base_joint, shoulder_joint, elbow_joint, wrist_pitch_joint, wrist_roll_joint
        # These are the 5 revolute joints for the arm
        num_joints = p.getNumJoints(robot_id)
        arm_joints = []
        gripper_link_idx = -1
        finger_joints = []
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            
            # Collect arm joints in order
            if 'arm_base_joint' in joint_name:
                arm_joints.append(i)
            elif 'shoulder_joint' in joint_name:
                arm_joints.append(i)
            elif 'elbow_joint' in joint_name:
                arm_joints.append(i)
            elif 'wrist_pitch_joint' in joint_name:
                arm_joints.append(i)
            elif 'wrist_roll_joint' in joint_name:
                arm_joints.append(i)
            
            # Find gripper base link (end effector for IK)
            link_name = joint_info[12].decode('utf-8')
            if 'gripper_base' in link_name:
                gripper_link_idx = i
            
            # Find finger joints for grasping
            if 'left_finger_joint' in joint_name or 'right_finger_joint' in joint_name:
                finger_joints.append(i)
        
        if len(arm_joints) < 5:
            print(f"[Motion Control] Warning: Only found {len(arm_joints)} arm joints, expected 5")
            return False
        
        if gripper_link_idx == -1:
            print("[Motion Control] Warning: Could not find gripper_base link")
            # Use last arm joint as fallback
            gripper_link_idx = arm_joints[-1]
    else:
        # If arm_joints provided, assume gripper is the link after last joint
        gripper_link_idx = arm_joints[-1]
        # Try to find finger joints
        num_joints = p.getNumJoints(robot_id)
        finger_joints = []
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            if 'finger' in joint_name:
                finger_joints.append(i)
    
    # Calculate IK solution
    ik_solution = p.calculateInverseKinematics(
        robot_id, 
        gripper_link_idx,  # End effector link
        target_pos, 
        targetOrientation=target_orient,
        maxNumIterations=100,
        residualThreshold=0.001
    )
    
    if ik_solution is None or len(ik_solution) == 0:
        print("[Motion Control] IK solution failed")
        return False
    
    # Apply IK solution to arm joints
    for i, joint_idx in enumerate(arm_joints):
        if i < len(ik_solution):
            p.setJointMotorControl2(
                robot_id,
                joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=ik_solution[i],
                force=500,
                maxVelocity=1.0
            )
    
    # Control gripper
    if close_gripper and len(finger_joints) > 0:
        # Close gripper (prismatic joints move inward)
        for finger_joint in finger_joints:
            p.setJointMotorControl2(
                robot_id,
                finger_joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.02,  # Close to 2cm
                force=50
            )
    elif len(finger_joints) > 0:
        # Open gripper
        for finger_joint in finger_joints:
            p.setJointMotorControl2(
                robot_id,
                finger_joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0,  # Fully open
                force=50
            )
    
    return True

# --------------------------
# Standalone Test Mode
# --------------------------
if __name__ == "__main__":
    """
    Test mode: Run this file directly to test motion control functions
    Usage: python src/modules/motion_control.py
    """
    print("[Motion Control] Running in test mode...")
    
    # Initialize simulation
    physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load floor
    p.loadURDF("plane.urdf")
    
    # Load robot URDF
    robot_id = p.loadURDF("../robot/robot.urdf", basePosition=[0, 0, 0.2])
    
    print(f"[Motion Control] Robot loaded. ID: {robot_id}")
    print("[Motion Control] Testing navigation...")
    
    # Test navigation
    goal = [2.0, 2.0]
    print(f"[Motion Control] Moving to goal: {goal}")
    
    for _ in range(2400):  # 10 seconds at 240 Hz
        dist = move_to_goal(robot_id, goal)
        if _ % 240 == 0:  # Print every second
            print(f"[Motion Control] Distance to goal: {dist:.2f}m")
        p.stepSimulation()
        time.sleep(1./240.)
        
        if dist < 0.1:
            print("[Motion Control] Goal reached!")
            break
    
    print("[Motion Control] Test complete. Press Ctrl+C to exit.")
    
    # Keep simulation running
    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        print("[Motion Control] Shutting down...")
        p.disconnect()

