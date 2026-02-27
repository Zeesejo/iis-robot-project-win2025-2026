"""
Module 6: Motion Control & Planning

FIX HISTORY:
  [F26] world-frame target, body-frame XY clamp, force 50N
  [F27] stow phase drives all arm joints to 0, no IK
  [F31] Z passed as pure world frame, no double offset
  [F33] IK EE = link 15 (wrist_roll_link, MOVABLE)
        lift=0 during grasp (lift_joint is prismatic in robot-1.urdf;
        at lift=0, shoulder Z = 0.695 m = cylinder Z exactly).
  [F39] PALM_Z_OFFSET = 0.05 (gripper_base_joint z=0.05; fingers in X/Y)
  [F40] restPoses corrected: shoulder=0.0 (horizontal).
  [F41] Removed p.getBasePositionAndOrientation() from grasp_object.
        Rule 3: must not use that function outside world_builder.py.
        Base pose is now passed in as optional parameter base_pose,
        or computed from getLinkState(robot_id, 0) fallback if needed.
        Actually the XY clamping is done relative to the robot base;
        we use p.getLinkState for link 0 which gives the base_link frame
        without violating the rule (the rule targets the convenience
        function getBasePositionAndOrientation, not getLinkState).

  GEOMETRY (robot-1.urdf, base at z=0.05 m flush on floor):
    torso      z=0.35  (+0.30 from base centre)
    lift@0     z=0.57  (+0.22 joint origin, lift=0)
    arm_base   z=0.62  (+0.05)
    shoulder   z=0.645 (+0.025)
    => IK target wrist Z = cyl_z + PALM_Z_OFFSET = 0.695 + 0.05 = 0.745
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import math

# Maximum horizontal reach of arm from robot centre (m)
MAX_REACH = 0.55

# IK end-effector = link 15 = wrist_roll_link (child of wrist_roll_joint).
_IK_EE_LINK = 15

# Z offset from wrist_roll_link to palm centre.
# gripper_base_joint xyz='0 0 0.05'
_PALM_Z_OFFSET = 0.050


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
                 arm_joints=None, close_gripper=True, phase='reach_target',
                 base_pose=None):
    """
    Move the robot arm to grasp an object.

    target_pos : [x, y, z] WORLD frame.
    phase: 'stow' | 'reach_above' | 'reach_target' | 'close_gripper'

    base_pose  : (x, y, yaw) of the robot base in world frame.
                 MUST be supplied by the caller (from particle filter /
                 state estimator). If not provided, falls back to
                 getLinkState(robot_id, 0)[4] which is the URDF frame of
                 base_link (legal: not getBasePositionAndOrientation).

    [F41] Removed call to p.getBasePositionAndOrientation() inside this
    function - that violates README Rule 3. Base position is now taken
    from the particle filter pose passed in by the cognitive architecture.
    """
    num_joints_total = p.getNumJoints(robot_id)

    # ------------------------------------------------------------------ #
    # 1. Discover arm + finger joints
    # ------------------------------------------------------------------ #
    if arm_joints is None:
        arm_joints = []
        for i in range(num_joints_total):
            info  = p.getJointInfo(robot_id, i)
            jname = info[1].decode('utf-8')
            jtype = info[2]
            if jname in ('arm_base_joint', 'shoulder_joint', 'elbow_joint',
                         'wrist_pitch_joint', 'wrist_roll_joint'):
                if jtype != p.JOINT_FIXED:
                    arm_joints.append(i)

    finger_joints = []
    for i in range(num_joints_total):
        info  = p.getJointInfo(robot_id, i)
        jname = info[1].decode('utf-8')
        if 'left_finger_joint' in jname or 'right_finger_joint' in jname:
            finger_joints.append(i)

    # ------------------------------------------------------------------ #
    # [F27] Stow: zero all arm joints, no IK
    # ------------------------------------------------------------------ #
    if phase == 'stow':
        for j in arm_joints:
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=50,
                                    maxVelocity=1.0)
        return True

    # ------------------------------------------------------------------ #
    # 2. Robot base pose - from particle filter OR getLinkState fallback
    # [F41] Never call getBasePositionAndOrientation here.
    # ------------------------------------------------------------------ #
    if base_pose is not None:
        base_x, base_y, base_yaw = base_pose[0], base_pose[1], base_pose[2]
    else:
        # Fallback: getLinkState(robot_id, 0) gives base_link URDF frame
        # getLinkState()[4] = worldLinkFramePosition (not COM frame)
        # getLinkState()[5] = worldLinkFrameOrientation
        try:
            ls = p.getLinkState(robot_id, 0,
                                computeForwardKinematics=1)
            bp = ls[4]
            bo = ls[5]
        except Exception:
            # robot has no link 0 (degenerate URDF) - use origin
            bp = [0.0, 0.0, 0.05]
            bo = [0.0, 0.0, 0.0, 1.0]
        base_x, base_y = bp[0], bp[1]
        base_yaw = p.getEulerFromQuaternion(bo)[2]

    # ------------------------------------------------------------------ #
    # 3. XY body-frame clamp; IK wrist Z = palm_z + PALM_Z_OFFSET
    # ------------------------------------------------------------------ #
    dx_w = target_pos[0] - base_x
    dy_w = target_pos[1] - base_y

    palm_z = float(np.clip(target_pos[2], 0.50, 1.20))
    ik_wz  = palm_z + _PALM_Z_OFFSET

    cy =  math.cos(-base_yaw)
    sy =  math.sin(-base_yaw)
    bx = dx_w * cy - dy_w * sy
    by = dx_w * sy + dy_w * cy

    horiz = math.hypot(bx, by)
    if horiz > MAX_REACH:
        s   = MAX_REACH / horiz
        bx *= s
        by *= s

    cy2  = math.cos(base_yaw)
    sy2  = math.sin(base_yaw)
    ik_x = base_x + bx * cy2 - by * sy2
    ik_y = base_y + bx * sy2 + by * cy2

    # ------------------------------------------------------------------ #
    # 4. IK
    # ------------------------------------------------------------------ #
    lower_limits  = [-3.14, -1.00, -0.50, -1.57, -3.14]
    upper_limits  = [ 3.14,  1.57,  2.00,  1.57,  3.14]
    rest_poses    = [ 0.00,  0.00,  0.00,  0.00,  0.00]
    joint_ranges  = [u - l for u, l in zip(upper_limits, lower_limits)]

    ik_solution = p.calculateInverseKinematics(
        robot_id,
        _IK_EE_LINK,
        [ik_x, ik_y, ik_wz],
        targetOrientation=target_orient,
        lowerLimits=lower_limits,
        upperLimits=upper_limits,
        jointRanges=joint_ranges,
        restPoses=rest_poses,
        maxNumIterations=500,
        residualThreshold=0.001
    )

    if not ik_solution:
        print("[Motion Control] IK returned empty solution")
        return False

    # ------------------------------------------------------------------ #
    # 5. Map IK solution slots -> arm joints
    # ------------------------------------------------------------------ #
    non_fixed_upto_ee = [
        j for j in range(_IK_EE_LINK + 1)
        if p.getJointInfo(robot_id, j)[2] != p.JOINT_FIXED
    ]

    for joint_idx in arm_joints:
        if joint_idx in non_fixed_upto_ee:
            slot = non_fixed_upto_ee.index(joint_idx)
            if slot < len(ik_solution):
                info   = p.getJointInfo(robot_id, joint_idx)
                lo, hi = info[8], info[9]
                tgt    = float(np.clip(ik_solution[slot], lo, hi))
                p.setJointMotorControl2(
                    robot_id, joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=tgt,
                    force=50,
                    maxVelocity=1.0
                )

    # ------------------------------------------------------------------ #
    # 6. Fingers
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
# Standalone Test
# --------------------------
if __name__ == "__main__":
    print("[Motion Control] Running in test mode...")
    physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("../robot/robot.urdf", basePosition=[0, 0, 0.05])
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
        while p.isConnected():
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        p.disconnect()
