"""
teleport_grasp_test.py
======================================================================
Skips SEARCH and NAVIGATE entirely.
Teleports the robot to STANDOFF_M in front of the red cylinder,
faces it directly, then forces FSM -> GRASP.

FIX HISTORY:
  F31 - use real world Z; gripper orientation [0,pi/2,0] (later reverted)
  F32 - IK XY uses ACTUAL cylinder (tx,ty) from PyBullet ground truth,
        clamped to MAX_REACH from robot base (not robot-forward estimate).
        Orientation corrected to [0,0,0] (identity) matching URDF finger
        geometry (fingers close in Y, gripper opens in +X / forward).
        IK end-effector = gripper_base_joint (idx 16).
        Logs gripper_base link world position every second.
        Settle loop extended to 240 steps (1 sim-second).

Usage:
    python executables/teleport_grasp_test.py

Expected output:
    PASS         - gripper contact confirmed
    PARTIAL PASS - arm moved, IK working, no contact (offset tweak needed)
    FAIL         - timeout with no contact
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
from src.modules.motion_control import grasp_object, MAX_REACH
from src.modules.fsm import RobotFSM, RobotState
from src.modules.knowledge_reasoning import get_knowledge_base

# ---------------------------------------------------------------------------
STANDOFF_M      = 0.55
GRASP_TIMEOUT_S = 30.0
DT              = 1.0 / 240.0

# Real world Z from world_builder constants
_CYL_Z       = TABLE_HEIGHT + (TARGET_HEIGHT / 2.0) + 0.01   # ~0.695 m
_ABOVE_Z     = _CYL_Z + 0.15                                  # ~0.845 m

# [F32] Grasp orientation = identity: gripper faces +X (forward),
# fingers close in Y. This matches the URDF prismatic finger geometry.
_GRASP_ORN_EULER = [0.0, 0.0, 0.0]

# IK end-effector joint index (gripper_base_joint = 16, FIXED)
_GRIPPER_BASE_JOINT = 16
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
    print("  TELEPORT GRASP TEST  [F32 - full audit fix]")
    print(f"  Cylinder world Z = {_CYL_Z:.3f} m  above Z = {_ABOVE_Z:.3f} m")
    print("=" * 60)

    robot_id, table_id, room_id, target_id = build_world(gui=True)
    initialize_state_estimator()
    kb = get_knowledge_base()

    arm_joints, gripper_joints, lift_joint = _detect_joints(robot_id)
    print(f"[Test] Arm joints:       {arm_joints}")
    print(f"[Test] Gripper joints:   {gripper_joints}")
    print(f"[Test] Lift joint:       {lift_joint}")
    print(f"[Test] IK end-effector:  joint {_GRIPPER_BASE_JOINT} (gripper_base_joint)")

    # [F32] identity orientation - gripper faces forward (+X), fingers close in Y
    grasp_orn = p.getQuaternionFromEuler(_GRASP_ORN_EULER)
    print(f"[Test] Grasp orientation (euler {_GRASP_ORN_EULER}): quat={[f'{v:.3f}' for v in grasp_orn]}")

    # Ground-truth target position
    tgt_pos_pb, _ = p.getBasePositionAndOrientation(target_id)
    tx, ty, tz = tgt_pos_pb
    print(f"[Test] Cylinder world pos: ({tx:.3f}, {ty:.3f}, {tz:.3f})")
    print(f"[Test] Expected Z ~ {_CYL_Z:.3f}  actual Z = {tz:.3f}")

    # Teleport robot STANDOFF_M in front of cylinder, facing it
    approach_angle = math.atan2(-ty, -tx)
    robot_x = tx + STANDOFF_M * math.cos(approach_angle)
    robot_y = ty + STANDOFF_M * math.sin(approach_angle)
    robot_z = 0.1
    face_yaw = math.atan2(ty - robot_y, tx - robot_x)
    orn_q    = p.getQuaternionFromEuler([0, 0, face_yaw])

    p.resetBasePositionAndOrientation(robot_id, [robot_x, robot_y, robot_z], orn_q)
    print(f"[Test] Robot at ({robot_x:.3f}, {robot_y:.3f}) facing {math.degrees(face_yaw):.1f} deg")
    print(f"[Test] Dist robot->cylinder: "
          f"{math.hypot(tx-robot_x, ty-robot_y):.3f} m")

    # Settle physics for 1 full sim-second
    for _ in range(240):
        p.stepSimulation()
        time.sleep(DT)

    kb.add_position('target', tx, ty, tz)

    # Force FSM to GRASP
    fsm = RobotFSM()
    fsm.transition_to(RobotState.SEARCH)
    fsm.transition_to(RobotState.NAVIGATE)
    fsm.transition_to(RobotState.APPROACH)
    fsm.transition_to(RobotState.GRASP)
    print(f"[Test] FSM: {fsm.state.name}")

    _set_wheels(robot_id, 0, 0)

    print("\n[Test] GRASP phases:")
    print("  stow         (0-1s):   arm joints -> 0, no IK")
    print("  reach_above  (1-3s):   IK to (tx, ty, cyl_z+0.15)")
    print("  reach_target (3-6s):   IK to (tx, ty, cyl_z)")
    print("  close_gripper(>6s):    IK hold + fingers close")
    print()

    contact_detected = False
    start_time       = time.time()
    last_log_s       = -1

    while True:
        fsm.tick()
        t_sim  = fsm.get_time_in_state()
        t_wall = time.time() - start_time

        phase = ('stow'          if t_sim < 1.0 else
                 'reach_above'   if t_sim < 3.0 else
                 'reach_target'  if t_sim < 6.0 else
                 'close_gripper')

        # [F32] IK XY = actual cylinder (tx,ty), clamped by grasp_object
        # IK Z = real world Z from world_builder constants
        ik_z   = _ABOVE_Z if phase in ('stow', 'reach_above') else _CYL_Z
        tgt_p  = [tx, ty, ik_z]
        close  = (phase == 'close_gripper')

        # Raise lift
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

        # Per-second log
        t_s = int(t_wall)
        if t_s != last_log_s:
            last_log_s = t_s

            fps = [f"{p.getJointState(robot_id, gj)[0]:.3f}" for gj in gripper_joints]

            # Log gripper_base link world position
            try:
                gb_pos = p.getLinkState(robot_id, _GRIPPER_BASE_JOINT)[0]
                gb_str = f"gripper_base=({gb_pos[0]:.3f},{gb_pos[1]:.3f},{gb_pos[2]:.3f})"
            except Exception:
                gb_str = "gripper_base=N/A"

            print(f"[t={t_s:2d}s|sim={t_sim:.1f}s] {phase:14s}  "
                  f"ik=({tx:.3f},{ty:.3f},{ik_z:.3f})  "
                  f"contact={'YES' if contact_detected else 'no '}  "
                  f"fingers={fps}  "
                  f"{gb_str}")

        p.stepSimulation()
        time.sleep(DT)

        # Exit conditions
        if contact_detected and phase == 'close_gripper' and t_sim > 8.0:
            print("\n" + "=" * 60)
            print("  RESULT: PASS - gripper contact confirmed!")
            print("=" * 60)
            break

        if t_wall > GRASP_TIMEOUT_S:
            if contact_detected:
                print("\n" + "=" * 60)
                print("  RESULT: PARTIAL PASS - contact detected, timed out")
                print("  IK is working. Check gripper_base Z vs cylinder Z in logs.")
                print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print("  RESULT: FAIL - no contact after 30s")
                print("  Check: gripper_base Z converging to ik Z?")
                print("  Check: gripper_base XY converging to cylinder XY?")
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
