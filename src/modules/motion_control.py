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
def move_to_goal(robot_id, goal_pos, current_pose, wheel_joints=[0, 1, 2, 3], dt=1./240.):
    """
    Move robot base using PID controller to goal position (x, y)
    Uses differential drive: Left wheels [0,2], Right wheels [1,3]
    
    Args:
        robot_id: PyBullet robot body ID
        goal_pos: Target position [x, y]
        current_pose: Current estimated pose [x, y, theta] from state estimation
        wheel_joints: List of wheel joint indices [FL=0, FR=1, BL=2, BR=3]
        dt: Time step
    
    Returns:
        Current distance to goal
    """
    x, y = current_pose[0], current_pose[1]
    theta = current_pose[2] if len(current_pose) > 2 else 0.0
    
    distance = math.hypot(goal_pos[0] - x, goal_pos[1] - y)
    
    if distance > 0.05:
        # Calculate heading to target
        angle_to_target = math.atan2(goal_pos[1] - y, goal_pos[0] - x)
        heading_error = angle_to_target - theta
        # Normalize to [-pi, pi]
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        
        # PID-style differential drive control
        Kp_lin = 3.0
        Kp_ang = 4.0
        forward_vel = min(Kp_lin * distance, 5.0)
        angular_vel = Kp_ang * heading_error
        
        left_vel = forward_vel - angular_vel
        right_vel = forward_vel + angular_vel
        
        # Apply to left wheels (FL=0, BL=2) and right wheels (FR=1, BR=3)
        for i in [0, 2]:  # Left wheels
            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL,
                                   targetVelocity=left_vel, force=5000.)
        for i in [1, 3]:  # Right wheels
            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL,
                                   targetVelocity=right_vel, force=5000.)
    else:
        # Stop at goal
        for i in wheel_joints:
            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL,
                                   targetVelocity=0.0, force=5000.)
    
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
    
    # Build mapping from joint index to IK solution index.
    # IK returns one value per non-fixed joint, in joint-index order.
    num_joints_total = p.getNumJoints(robot_id)
    non_fixed_joints = []
    for j in range(num_joints_total):
        if p.getJointInfo(robot_id, j)[2] != p.JOINT_FIXED:
            non_fixed_joints.append(j)
    
    # Apply IK solution to arm joints using correct index mapping
    for joint_idx in arm_joints:
        if joint_idx in non_fixed_joints:
            ik_idx = non_fixed_joints.index(joint_idx)
            if ik_idx < len(ik_solution):
                p.setJointMotorControl2(
                    robot_id,
                    joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=ik_solution[ik_idx],
                    force=500,
                    maxVelocity=1.0
                )
    
    # Control gripper
    if close_gripper and len(finger_joints) > 0:
        # Close gripper: move fingers toward center
        for finger_joint in finger_joints:
            joint_info = p.getJointInfo(robot_id, finger_joint)
            joint_name = joint_info[1].decode('utf-8')
            # Left finger: limits [-0.04, 0], close at -0.04 (moves toward center)
            # Right finger: limits [0, 0.04], close at 0.04 (moves toward center)
            if 'left' in joint_name:
                target = -0.04
            else:
                target = 0.04
            p.setJointMotorControl2(
                robot_id,
                finger_joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=50
            )
    elif len(finger_joints) > 0:
        # Open gripper: move fingers away from center
        for finger_joint in finger_joints:
            joint_info = p.getJointInfo(robot_id, finger_joint)
            joint_name = joint_info[1].decode('utf-8')
            # Left finger: open at 0, Right finger: open at 0
            p.setJointMotorControl2(
                robot_id,
                finger_joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0,
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
    
    # Note: In real usage, current_pose should come from particle filter
    # For this test, we'll use a dummy pose that gets updated
    test_pose = [0.0, 0.0, 0.0]
    
    for _ in range(2400):  # 10 seconds at 240 Hz
        dist = move_to_goal(robot_id, goal, test_pose)
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

