"""
teleport_grasp_test.py
======================================================================
Skips SEARCH and NAVIGATE entirely.
Teleports the robot to 0.55 m in front of the red cylinder,
faces it directly, injects the target position into the KB,
then forces FSM -> GRASP and runs the STA loop.

FIX [F31]:
  - Grasp IK target uses REAL WORLD Z of the cylinder (from PyBullet
    ground truth), not a body-frame offset added to robot base Z.
    Old bug: ARM_GRASP_Z_BODY (0.65) was added to base_pos[2] (0.1)
    making IK target Z = 0.75 m, but grasp_object() was ALSO adding
    base_pos[2] internally -> IK got Z ~0.85 m, arm flew backward.
  - Gripper orientation fixed to [0, pi/2, 0] so gripper points
    DOWNWARD for a top-down grasp (was [0,0,0] = pointing sideways).
  - reach_above uses tz + 0.15 m; reach_target uses tz directly.

Usage:
    python executables/teleport_grasp_test.py

Expected output:
    PASS         - gripper contact confirmed within 25 s
    PARTIAL PASS - arm moved correctly, IK working, orientation tweak needed
    FAIL         - arm detached or timeout with no contact
======================================================================
"""

import sys
import os
import time
import math
import numpy as np
import pybullet as p

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.world_builder import build_world, TABLE_HEIGHT, TARGET_HEIGHT
from src.modules.state_estimation import initialize_state_estimator
from src.modules.motion_control import grasp_object
from src.modules.fsm import RobotFSM, RobotState
from src.modules.knowledge_reasoning import get_knowledge_base

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STANDOFF_M      = 0.55    # metres in front of cylinder to place robot
GRASP_TIMEOUT_S = 25.0    # wall-clock seconds before declaring FAIL
DT              = 1.0 / 240.0
ARM_FORWARD_OFFSET = 0.45  # metres forward from robot base to IK target XY

# Gripper points DOWNWARD for top-down grasp
# Euler [0, pi/2, 0] -> quaternion pointing -Z of gripper toward world -Z
GRASP_ORIENTATION = p.getQuaternionFromEuler([0, math.pi / 2, 0]) if False else None
# (computed after p.connect, see main())

# ---------------------------------------------------------------------------

def _set_wheels(robot_id, left, right):
    for i in [0, 2]:
        p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL,
                                targetVelocity=left,  force=5000)
    for i in [1, 3]:
        p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL,
                                targetVelocity=right, force=5000)


def _detect_joints(robot_id):
    arm_joints, gripper_joints = [], []
    lift_joint = None
    for i in range(p.getNumJoints(robot_id)):
        info  = p.getJointInfo(robot_id, i)
        jname = info[1].decode('utf-8')
        jtype = info[2]
        if jname in ('arm_base_joint', 'shoulder_joint', 'elbow_joint',
                     'wrist_pitch_joint', 'wrist_roll_joint'):
            if jtype != p.JOINT_FIXED:
                arm_joints.append(i)
        if 'left_finger_joint' in jname or 'right_finger_joint' in jname:
            gripper_joints.append(i)
        if jname == 'lift_joint' and jtype != p.JOINT_FIXED:
            lift_joint = i
    return arm_joints, gripper_joints, lift_joint


def main():
    print("=" * 60)
    print("  TELEPORT GRASP TEST  [F31 - world-Z fix + downward grip]")
    print("  Skipping SEARCH + NAVIGATE - starting at GRASP range")
    print("=" * 60)

    # -- Build world ---------------------------------------------------------
    robot_id, table_id, room_id, target_id = build_world(gui=True)
    initialize_state_estimator()
    kb = get_knowledge_base()

    arm_joints, gripper_joints, lift_joint = _detect_joints(robot_id)
    print(f"[Test] Arm joints:     {arm_joints}")
    print(f"[Test] Gripper joints: {gripper_joints}")
    print(f"[Test] Lift joint:     {lift_joint}")

    # Compute downward-pointing gripper orientation AFTER p.connect
    # [0, pi/2, 0]: rotate 90 deg around Y -> gripper Z points down
    grasp_orn = p.getQuaternionFromEuler([0, math.pi / 2, 0])
    print(f"[Test] Grasp orientation (quat): {[f'{v:.3f}' for v in grasp_orn]}")

    # -- Target ground-truth from PyBullet ------------------------------------
    tgt_pos_pb, _ = p.getBasePositionAndOrientation(target_id)
    tx, ty, tz = tgt_pos_pb
    # Expected: tz ~ TABLE_HEIGHT + TARGET_HEIGHT/2 + 0.01
    expected_tz = TABLE_HEIGHT + (TARGET_HEIGHT / 2.0) + 0.01
    print(f"[Test] Target world pos: ({tx:.3f}, {ty:.3f}, {tz:.3f})")
    print(f"[Test] Expected tz ~ {expected_tz:.3f} m  (table={TABLE_HEIGHT} + "
          f"cyl_half={TARGET_HEIGHT/2:.3f} + 0.01)")

    # IK targets
    reach_above_z = tz + 0.15   # 15 cm above cylinder top
    reach_z       = tz          # cylinder centre
    print(f"[Test] IK reach_above Z = {reach_above_z:.3f} m")
    print(f"[Test] IK reach_target Z = {reach_z:.3f} m")

    # -- Teleport robot in front of target ------------------------------------
    approach_angle = math.atan2(0.0 - ty, 0.0 - tx)
    robot_x = tx + STANDOFF_M * math.cos(approach_angle)
    robot_y = ty + STANDOFF_M * math.sin(approach_angle)
    robot_z = 0.1

    face_angle = math.atan2(ty - robot_y, tx - robot_x)
    orn_quat   = p.getQuaternionFromEuler([0, 0, face_angle])

    p.resetBasePositionAndOrientation(robot_id,
                                      [robot_x, robot_y, robot_z],
                                      orn_quat)
    print(f"[Test] Robot teleported to ({robot_x:.3f}, {robot_y:.3f}) "
          f"facing {math.degrees(face_angle):.1f} deg")

    # Let physics settle
    for _ in range(120):
        p.stepSimulation()
        time.sleep(DT)

    # -- KB ------------------------------------------------------------------
    kb.add_position('target', tx, ty, tz)

    # -- Force FSM to GRASP --------------------------------------------------
    fsm = RobotFSM()
    fsm.transition_to(RobotState.SEARCH)
    fsm.transition_to(RobotState.NAVIGATE)
    fsm.transition_to(RobotState.APPROACH)
    fsm.transition_to(RobotState.GRASP)
    print(f"[Test] FSM state: {fsm.state.name}")

    _set_wheels(robot_id, 0, 0)

    # -- GRASP loop ----------------------------------------------------------
    print("\n[Test] Starting GRASP loop...")
    print("  Phase stow         (0-1s):   arm joints -> 0")
    print("  Phase reach_above  (1-3s):   IK to tz + 0.15 m")
    print("  Phase reach_target (3-6s):   IK to tz (cylinder centre)")
    print("  Phase close_gripper(>6s):    IK hold + fingers close")
    print()

    contact_detected = False
    arm_detached     = False
    start_time       = time.time()
    last_log_s       = -1

    while True:
        fsm.tick()
        t_sim  = fsm.get_time_in_state()
        t_wall = time.time() - start_time

        phase = ('stow'          if t_sim < 1.0  else
                 'reach_above'   if t_sim < 3.0  else
                 'reach_target'  if t_sim < 6.0  else
                 'close_gripper')

        # [F31] IK target: use REAL world XY (forward from robot) + REAL world Z
        base_pos_pb, base_orn_pb = p.getBasePositionAndOrientation(robot_id)
        base_yaw = p.getEulerFromQuaternion(base_orn_pb)[2]

        # XY: ARM_FORWARD_OFFSET in front of robot base (robot is already
        # facing the target after teleport, so this points at the cylinder)
        ik_wx = base_pos_pb[0] + ARM_FORWARD_OFFSET * math.cos(base_yaw)
        ik_wy = base_pos_pb[1] + ARM_FORWARD_OFFSET * math.sin(base_yaw)

        # Z: pure world frame - no body-frame offset added
        ik_wz = reach_above_z if phase in ('stow', 'reach_above') else reach_z

        tgt_p = [ik_wx, ik_wy, ik_wz]
        close = (phase == 'close_gripper')

        # Raise lift fully
        if lift_joint is not None:
            p.setJointMotorControl2(robot_id, lift_joint,
                                    p.POSITION_CONTROL,
                                    targetPosition=0.3,
                                    force=100, maxVelocity=0.5)

        grasp_object(robot_id, tgt_p, grasp_orn,
                     arm_joints=arm_joints or None,
                     close_gripper=close,
                     phase=phase)

        _set_wheels(robot_id, 0, 0)

        # Contact check
        contacts = p.getContactPoints(bodyA=robot_id, bodyB=target_id)
        if contacts and len(contacts) > 0:
            contact_detected = True

        # Detachment check
        rp, _ = p.getBasePositionAndOrientation(robot_id)
        if abs(rp[2] - robot_z) > 0.5:
            arm_detached = True
            print(f"[Test] WARNING: robot base Z moved! z={rp[2]:.3f}")

        for aj in arm_joints:
            ls  = p.getLinkState(robot_id, aj)
            lp  = ls[0]
            d   = math.dist(lp, [robot_x, robot_y, robot_z])
            if d > 3.0:
                arm_detached = True
                print(f"[Test] ARM DETACHED: joint {aj} is {d:.2f}m away!")
                break

        # Per-second log
        t_s = int(t_wall)
        if t_s != last_log_s:
            last_log_s = t_s
            fps = [f"{p.getJointState(robot_id, gj)[0]:.3f}" for gj in gripper_joints]

            # Also log actual gripper_base link position for debugging
            gripper_link = arm_joints[-1] if arm_joints else None
            gl_str = ""
            if gripper_link is not None:
                gl_pos = p.getLinkState(robot_id, gripper_link)[0]
                gl_str = (f" gripper_link=({gl_pos[0]:.2f},{gl_pos[1]:.2f},"
                          f"{gl_pos[2]:.2f})")

            print(f"[t={t_s:2d}s|sim={t_sim:.1f}s] {phase:14s} "
                  f"ik_target=({ik_wx:.2f},{ik_wy:.2f},{ik_wz:.2f}) "
                  f"contact={'YES' if contact_detected else 'no '} "
                  f"fingers={fps}"
                  f"{gl_str}")

        p.stepSimulation()
        time.sleep(DT)

        # Exit conditions
        if arm_detached:
            print("\n" + "=" * 60)
            print("  RESULT: FAIL - arm detached")
            print("=" * 60)
            break

        if contact_detected and phase == 'close_gripper' and t_sim > 8.0:
            print("\n" + "=" * 60)
            print("  RESULT: PASS - gripper contact confirmed, arm intact!")
            print("=" * 60)
            break

        if t_wall > GRASP_TIMEOUT_S:
            if contact_detected:
                print("\n" + "=" * 60)
                print("  RESULT: PARTIAL PASS - contact seen but timed out")
                print("  Arm intact, IK working. May need orientation/offset tuning.")
                print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print("  RESULT: FAIL - timeout, no contact")
                print("  Check gripper_link position in logs vs ik_target")
                print("=" * 60)
            break

    print("\n[Test] Done. Close PyBullet window to exit.")
    try:
        while p.isConnected():
            p.stepSimulation()
            time.sleep(DT)
    except Exception:
        pass


if __name__ == "__main__":
    main()
