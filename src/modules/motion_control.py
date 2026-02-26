"""
Module 6: Motion Control & Planning
Provides PID controllers, navigation, manipulation, and path planning

FIX HISTORY:
  [F26] grasp_object: world-frame target, body-frame XY clamp, force 50N
  [F27] stow phase: drive all arm joints to 0 before IK
  [F31] remove double-Z offset bug; Z passed as pure world frame
  [F32] CRITICAL audit fixes:
        - IK end-effector link = 16 (gripper_base_joint, FIXED joint whose
          child is gripper_base). Using arm_joints[-1]=15 (wrist_roll_joint)
          placed IK target at the wrist, 5-8 cm short of the fingers.
        - Grasp orientation = [0,0,0] (identity). The URDF finger joints are
          prismatic along Y with origin xyz="0.04 ±0.04 0", meaning fingers
          close laterally (Y-axis) and the gripper opening faces +X (forward).
          [0,pi/2,0] rotated the gripper to point down, which is wrong.
          With [0,0,0] the gripper faces forward and wraps around the cylinder.
        - IK joint mapping: calculateInverseKinematics returns one angle per
          non-fixed DOF in the entire chain (lift + arm). We now slice the
          solution by counting non-fixed joints UP TO gripper_base_joint(16),
          then pick only the slots corresponding to arm_joints [8,9,11,12,15].
        - Stow returns immediately after zeroing joints - no IK called.
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import math

# Maximum horizontal reach of arm from robot centre (m)
MAX_REACH = 0.55

# IK end-effector: gripper_base_joint (index 16, FIXED) -
# PyBullet uses the child link of this joint as the IK target frame.
# This places the IK target at the gripper palm, not the wrist.
_GRIPPER_BASE_JOINT_IDX = 16


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

    target_pos : [x, y, z] in WORLD frame.
        XY  - converted to body frame, clamped to MAX_REACH, back to world.
        Z   - passed AS-IS (world frame, e.g. cylinder centre ~0.695 m).

    target_orient : quaternion.  Use [0,0,0,1] (identity / [0,0,0] euler).
        The URDF gripper opens along +X (finger prismatic joints in Y with
        origin offset in X).  Identity orientation keeps gripper facing
        forward, which is correct for a horizontal approach grasp.

    phase:
        'stow'          - zero all arm joints, return immediately (no IK)
        'reach_above'   - IK to target_pos (caller sets Z = cyl_z + 0.15)
        'reach_target'  - IK to target_pos (caller sets Z = cyl_z)
        'close_gripper' - IK hold + close fingers
    """
    num_joints_total = p.getNumJoints(robot_id)

    # ------------------------------------------------------------------ #
    # 1. Discover joints
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

    # [F32] IK end-effector = gripper_base_joint (idx 16, FIXED).
    # PyBullet IK targets the child link of the given joint index.
    gripper_link_idx = _GRIPPER_BASE_JOINT_IDX

    # ------------------------------------------------------------------ #
    # [F27] Stow: zero all arm joints, NO IK
    # ------------------------------------------------------------------ #
    if phase == 'stow':
        for j in arm_joints:
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=50,
                                    maxVelocity=1.0)
        return True

    # ------------------------------------------------------------------ #
    # 2. Robot base pose
    # ------------------------------------------------------------------ #
    base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
    base_yaw = p.getEulerFromQuaternion(base_orn)[2]

    # ------------------------------------------------------------------ #
    # 3. XY clamp to reachable range; Z straight through (world frame)
    # ------------------------------------------------------------------ #
    dx_w  = target_pos[0] - base_pos[0]
    dy_w  = target_pos[1] - base_pos[1]
    world_z = float(np.clip(target_pos[2], 0.50, 1.20))

    # rotate to body frame
    cy =  math.cos(-base_yaw)
    sy =  math.sin(-base_yaw)
    bx = dx_w * cy - dy_w * sy   # forward
    by = dx_w * sy + dy_w * cy   # lateral

    horiz = math.hypot(bx, by)
    if horiz > MAX_REACH:
        s   = MAX_REACH / horiz
        bx *= s
        by *= s

    # back to world frame
    cy2  = math.cos(base_yaw)
    sy2  = math.sin(base_yaw)
    ik_x = base_pos[0] + bx * cy2 - by * sy2
    ik_y = base_pos[1] + bx * sy2 + by * cy2
    ik_z = world_z

    # ------------------------------------------------------------------ #
    # 4. IK
    # [F32] Joint limits cover the 5 revolute arm DOFs only.
    #       The IK solution vector covers ALL non-fixed joints up to
    #       gripper_link_idx. We build a mapping: for each joint in
    #       arm_joints, find its position in the non-fixed list that
    #       IK returns.
    # ------------------------------------------------------------------ #
    lower_limits = [-3.14, -1.00, -0.50, -1.57, -3.14]
    upper_limits = [ 3.14,  1.57,  2.00,  1.57,  3.14]
    rest_poses   = [ 0.00,  0.50,  1.00,  0.00,  0.00]
    joint_ranges = [u - l for u, l in zip(upper_limits, lower_limits)]

    ik_solution = p.calculateInverseKinematics(
        robot_id,
        gripper_link_idx,
        [ik_x, ik_y, ik_z],
        targetOrientation=target_orient,
        lowerLimits=lower_limits,
        upperLimits=upper_limits,
        jointRanges=joint_ranges,
        restPoses=rest_poses,
        maxNumIterations=300,
        residualThreshold=0.003
    )

    if not ik_solution:
        return False

    # ------------------------------------------------------------------ #
    # 5. Map IK solution slots -> arm joints
    # [F32] calculateInverseKinematics returns one value per non-fixed
    #       joint in the ENTIRE chain up to gripper_link_idx.
    #       Build the non-fixed list up to (not including) gripper_base_joint
    #       to get the correct slot index for each arm joint.
    # ------------------------------------------------------------------ #
    non_fixed_upto_gripper = [
        j for j in range(gripper_link_idx)
        if p.getJointInfo(robot_id, j)[2] != p.JOINT_FIXED
    ]

    for joint_idx in arm_joints:
        if joint_idx in non_fixed_upto_gripper:
            slot = non_fixed_upto_gripper.index(joint_idx)
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
# Standalone Test Mode
# --------------------------
if __name__ == "__main__":
    print("[Motion Control] Running in test mode...")
    physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("../robot/robot.urdf", basePosition=[0, 0, 0.2])
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
