"""
Module 6: Motion Control & Planning
Provides PID controllers, navigation, manipulation, and path planning

FIXES:
  [F26] grasp_object: target_pos is in WORLD frame.
        Converts to robot-body frame, clamps reach to MAX_REACH,
        converts back to world frame for IK.
        Joint force reduced 500 -> 50 N to prevent joint separation.
  [F27] Pre-IK stow phase resets arm config.
  [F31] CRITICAL: Remove double Z-offset bug.
        Previously body_z was computed as (world_z - base_z) and then
        IK target was reconstructed as base_z + body_z = world_z -- OK
        in theory, but the clamping clip(body_z, 0.40, 0.85) was
        preventing the arm from reaching table height when base_z=0.1,
        because body_z = 0.695 - 0.1 = 0.595, clipped OK, but ik_wz
        was base_z(0.1) + body_z(0.595) = 0.695 -- actually correct.
        THE REAL BUG was in teleport_grasp_test.py passing
        base_pos[2] + ARM_GRASP_Z_BODY as the world-Z input, which
        then got further offset here.  Fix: accept pure world-Z in
        target_pos[2] and pass it directly to IK without re-offsetting.
        The body-frame conversion now only applies to XY (horizontal
        reach clamping). Z is passed through as-is (world frame).
        Z clamp updated to world-frame range [0.50, 1.20] m.
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import math

# Maximum horizontal reach of arm from robot centre (m)
MAX_REACH = 0.55


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
# Prolog Path Planner
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
    if PROLOG_AVAILABLE:
        try:
            query = f"plan_path({start}, {goal}, Path)."
            for sol in prolog.query(query):
                return sol["Path"]
        except:
            pass
    return [goal]


def move_to_goal(robot_id, goal_pos, current_pose,
                 wheel_joints=[0, 1, 2, 3], dt=1./240.):
    x, y  = current_pose[0], current_pose[1]
    theta = current_pose[2] if len(current_pose) > 2 else 0.0
    distance = math.hypot(goal_pos[0] - x, goal_pos[1] - y)

    if distance > 0.05:
        angle_to_target = math.atan2(goal_pos[1] - y, goal_pos[0] - x)
        heading_error   = math.atan2(
            math.sin(angle_to_target - theta),
            math.cos(angle_to_target - theta))
        Kp_lin    = 3.0
        Kp_ang    = 4.0
        fwd_vel   = min(Kp_lin * distance, 5.0)
        ang_vel   = Kp_ang * heading_error
        left_vel  = fwd_vel - ang_vel
        right_vel = fwd_vel + ang_vel
        for i in [0, 2]:
            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL,
                                    targetVelocity=left_vel,  force=5000.)
        for i in [1, 3]:
            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL,
                                    targetVelocity=right_vel, force=5000.)
    else:
        for i in wheel_joints:
            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL,
                                    targetVelocity=0.0, force=5000.)
    return distance


def grasp_object(robot_id, target_pos, target_orient,
                 arm_joints=None, close_gripper=True, phase='reach_target'):
    """
    Move the robot arm to grasp an object.

    target_pos: [x, y, z] in WORLD frame.
      - XY is converted to robot-body frame, clamped to MAX_REACH,
        then converted back to world frame for IK.
      - Z is used AS-IS (world frame). Pass the actual world Z of
        the grasp point (e.g. cylinder centre height). Do NOT add
        robot base height - this function does not offset Z.

    phase:
      'stow'          - drive all arm joints to 0 (reset config)
      'reach_above'   - IK to target_pos (caller adds +0.15 to Z before calling)
      'reach_target'  - IK to target_pos
      'close_gripper' - IK hold + close fingers
    """
    # ------------------------------------------------------------------ #
    # 1. Discover joints
    # ------------------------------------------------------------------ #
    num_joints_total = p.getNumJoints(robot_id)
    if arm_joints is None:
        arm_joints    = []
        finger_joints = []
        gripper_link_idx = -1
        for i in range(num_joints_total):
            info  = p.getJointInfo(robot_id, i)
            jname = info[1].decode('utf-8')
            lname = info[12].decode('utf-8')
            jtype = info[2]
            if jname in ('arm_base_joint', 'shoulder_joint', 'elbow_joint',
                         'wrist_pitch_joint', 'wrist_roll_joint'):
                if jtype != p.JOINT_FIXED:
                    arm_joints.append(i)
            if 'left_finger_joint' in jname or 'right_finger_joint' in jname:
                finger_joints.append(i)
            if 'gripper_base' in lname:
                gripper_link_idx = i
        if gripper_link_idx == -1 and arm_joints:
            gripper_link_idx = arm_joints[-1]
    else:
        gripper_link_idx = arm_joints[-1]
        finger_joints    = []
        for i in range(num_joints_total):
            info  = p.getJointInfo(robot_id, i)
            jname = info[1].decode('utf-8')
            if 'finger' in jname:
                finger_joints.append(i)

    # ------------------------------------------------------------------ #
    # [F27] Stow phase
    # ------------------------------------------------------------------ #
    if phase == 'stow':
        for j in arm_joints:
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=50,
                                    maxVelocity=2.0)
        return True

    # ------------------------------------------------------------------ #
    # 2. Get robot base pose
    # ------------------------------------------------------------------ #
    base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
    base_yaw = p.getEulerFromQuaternion(base_orn)[2]

    # ------------------------------------------------------------------ #
    # 3. [F31] XY-only body-frame conversion + reach clamp
    #    Z is world-frame and passed through unchanged.
    # ------------------------------------------------------------------ #
    dx_w = target_pos[0] - base_pos[0]
    dy_w = target_pos[1] - base_pos[1]
    world_z = float(target_pos[2])   # pure world Z - no offset added

    # Rotate XY into body frame
    cos_y =  math.cos(-base_yaw)
    sin_y =  math.sin(-base_yaw)
    body_x = dx_w * cos_y - dy_w * sin_y   # forward
    body_y = dx_w * sin_y + dy_w * cos_y   # lateral

    # Clamp horizontal reach
    horiz = math.hypot(body_x, body_y)
    if horiz > MAX_REACH:
        scale  = MAX_REACH / horiz
        body_x *= scale
        body_y *= scale

    # [F31] Clamp Z to world-frame plausible range
    # Table top = 0.625, cylinder centre ~0.695, above = ~0.845
    world_z = float(np.clip(world_z, 0.50, 1.20))

    # Convert clamped XY back to world frame
    cos_y2 = math.cos(base_yaw)
    sin_y2 = math.sin(base_yaw)
    ik_wx  = base_pos[0] + body_x * cos_y2 - body_y * sin_y2
    ik_wy  = base_pos[1] + body_x * sin_y2 + body_y * cos_y2
    ik_wz  = world_z     # [F31] world Z passed straight through

    ik_target = [ik_wx, ik_wy, ik_wz]

    # ------------------------------------------------------------------ #
    # 4. IK with joint limits
    # ------------------------------------------------------------------ #
    lower_limits = [-3.14, -1.00, -0.50, -1.57, -3.14]
    upper_limits = [ 3.14,  1.57,  2.00,  1.57,  3.14]
    rest_poses   = [ 0.00,  0.50,  1.00,  0.00,  0.00]
    joint_ranges = [u - l for u, l in zip(upper_limits, lower_limits)]

    ik_solution = p.calculateInverseKinematics(
        robot_id,
        gripper_link_idx,
        ik_target,
        targetOrientation=target_orient,
        lowerLimits=lower_limits,
        upperLimits=upper_limits,
        jointRanges=joint_ranges,
        restPoses=rest_poses,
        maxNumIterations=200,
        residualThreshold=0.005
    )

    if ik_solution is None or len(ik_solution) == 0:
        print("[Motion Control] IK solution failed")
        return False

    # ------------------------------------------------------------------ #
    # 5. Apply with index mapping + joint-limit clamp
    # ------------------------------------------------------------------ #
    non_fixed = [j for j in range(num_joints_total)
                 if p.getJointInfo(robot_id, j)[2] != p.JOINT_FIXED]

    for joint_idx in arm_joints:
        if joint_idx in non_fixed:
            ik_idx = non_fixed.index(joint_idx)
            if ik_idx < len(ik_solution):
                info   = p.getJointInfo(robot_id, joint_idx)
                lo, hi = info[8], info[9]
                target = float(np.clip(ik_solution[ik_idx], lo, hi))
                p.setJointMotorControl2(
                    robot_id, joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target,
                    force=50,
                    maxVelocity=1.0
                )

    # ------------------------------------------------------------------ #
    # 6. Gripper
    # ------------------------------------------------------------------ #
    for fi in finger_joints:
        jname = p.getJointInfo(robot_id, fi)[1].decode('utf-8')
        if close_gripper:
            tp = -0.04 if 'left' in jname else 0.04
        else:
            tp = 0.0
        p.setJointMotorControl2(robot_id, fi, p.POSITION_CONTROL,
                                targetPosition=tp, force=50)

    return True


# --------------------------
# Standalone Test Mode
# --------------------------
if __name__ == "__main__":
    import pybullet_data
    print("[Motion Control] Running in test mode...")
    physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("../robot/robot.urdf", basePosition=[0, 0, 0.2])
    print(f"[Motion Control] Robot loaded. ID: {robot_id}")
    goal      = [2.0, 2.0]
    test_pose = [0.0, 0.0, 0.0]
    for _ in range(2400):
        dist = move_to_goal(robot_id, goal, test_pose)
        if _ % 240 == 0:
            print(f"[Motion Control] Distance to goal: {dist:.2f}m")
        p.stepSimulation()
        time.sleep(1./240.)
        if dist < 0.1:
            print("[Motion Control] Goal reached!")
            break
    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        p.disconnect()
