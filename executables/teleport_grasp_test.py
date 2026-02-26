"""
teleport_grasp_test.py
======================================================================
Skips SEARCH and NAVIGATE entirely.
Teleports the robot to 0.55 m in front of the red cylinder,
faces it directly, injects the target position into the KB,
then forces FSM -> GRASP and runs the STA loop.

Use this to verify the arm IK / grasp logic in isolation
before testing the full pipeline.

Usage:
    python executables/teleport_grasp_test.py

Expected output:
    PASS  - gripper contact detected within 20 s
    PARTIAL PASS - arm moved correctly, no collision / detachment
    FAIL  - arm detached, IK garbage, or timeout
======================================================================
"""

import sys
import os
import time
import math
import numpy as np
import pybullet as p

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.world_builder import build_world
from src.modules.sensor_preprocessing import get_sensor_id
from src.modules.perception import PerceptionModule
from src.modules.state_estimation import state_estimate, initialize_state_estimator
from src.modules.motion_control import grasp_object
from src.modules.fsm import RobotFSM, RobotState
from src.modules.knowledge_reasoning import get_knowledge_base

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STANDOFF_M      = 0.55    # how far in front of cylinder to place robot
GRASP_TIMEOUT_S = 25.0    # seconds before declaring FAIL
DT              = 1.0 / 240.0
ARM_FORWARD_OFFSET = 0.45  # metres forward from robot base for grasp target
ARM_GRASP_Z_BODY   = 0.65  # metres above robot base

# ---------------------------------------------------------------------------

def _set_wheels(robot_id, left, right):
    for i in [0, 2]:
        p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL,
                                targetVelocity=left,  force=5000)
    for i in [1, 3]:
        p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL,
                                targetVelocity=right, force=5000)


def _stow_arm(robot_id, arm_joints):
    for j in arm_joints:
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL,
                                targetPosition=0.0, force=50, maxVelocity=2.0)


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
    print("  TELEPORT GRASP TEST")
    print("  Skipping SEARCH + NAVIGATE - starting at GRASP range")
    print("=" * 60)

    # -- Build world ---------------------------------------------------------
    robot_id, table_id, room_id, target_id = build_world(gui=True)
    initialize_state_estimator()
    kb = get_knowledge_base()

    arm_joints, gripper_joints, lift_joint = _detect_joints(robot_id)
    print(f"[Test] Arm joints: {arm_joints}")
    print(f"[Test] Gripper joints: {gripper_joints}")
    print(f"[Test] Lift joint: {lift_joint}")

    # -- Find target world position from PyBullet ground truth ---------------
    tgt_pos_pb, tgt_orn_pb = p.getBasePositionAndOrientation(target_id)
    tx, ty, tz = tgt_pos_pb
    print(f"[Test] Target ground-truth position: ({tx:.3f}, {ty:.3f}, {tz:.3f})")

    # -- Teleport robot to STANDOFF_M in front of target, facing it ----------
    # Compute approach angle: from target back toward origin (robot starts at 0,0)
    approach_angle = math.atan2(0.0 - ty, 0.0 - tx)   # direction from target toward origin
    robot_x = tx + STANDOFF_M * math.cos(approach_angle)
    robot_y = ty + STANDOFF_M * math.sin(approach_angle)
    robot_z = 0.1   # standard spawn height

    # Robot should face the target (opposite of approach_angle)
    face_angle = math.atan2(ty - robot_y, tx - robot_x)
    orn_quat   = p.getQuaternionFromEuler([0, 0, face_angle])

    p.resetBasePositionAndOrientation(robot_id,
                                      [robot_x, robot_y, robot_z],
                                      orn_quat)
    print(f"[Test] Robot teleported to ({robot_x:.3f}, {robot_y:.3f}) "
          f"facing {math.degrees(face_angle):.1f} deg")

    # Let physics settle for 0.5 s
    for _ in range(120):
        p.stepSimulation()
        time.sleep(DT)

    # -- Inject target into KB -----------------------------------------------
    kb.add_position('target', tx, ty, tz)

    # -- Set up a minimal FSM and force GRASP state --------------------------
    fsm = RobotFSM()
    # Manually transition: IDLE -> SEARCH -> NAVIGATE -> APPROACH -> GRASP
    # (each transition clears failure count)
    fsm.transition_to(RobotState.SEARCH)
    fsm.transition_to(RobotState.NAVIGATE)
    fsm.transition_to(RobotState.APPROACH)
    fsm.transition_to(RobotState.GRASP)
    print(f"[Test] FSM forced to: {fsm.state.name}")

    # -- Wheels off ----------------------------------------------------------
    _set_wheels(robot_id, 0, 0)

    # -- Main GRASP loop -----------------------------------------------------
    print("\n[Test] Starting GRASP loop...")
    print("  Phase 0 (0-1s):  STOW arm")
    print("  Phase 1 (1-3s):  REACH_ABOVE")
    print("  Phase 2 (3-6s):  REACH_TARGET")
    print("  Phase 3 (>6s):   CLOSE_GRIPPER")
    print()

    contact_detected = False
    arm_detached     = False
    start_time       = time.time()
    step             = 0
    last_log_s       = -1

    while True:
        fsm.tick()
        t_sim = fsm.get_time_in_state()   # seconds in GRASP state
        t_wall = time.time() - start_time

        # Phase logic (mirrors cognitive_architecture.py)
        phase = ('stow'          if t_sim < 1.0  else
                 'reach_above'   if t_sim < 3.0  else
                 'reach_target'  if t_sim < 6.0  else
                 'close_gripper')

        # Compute grasp target in world frame from live robot pose
        base_pos_pb, base_orn_pb = p.getBasePositionAndOrientation(robot_id)
        base_yaw = p.getEulerFromQuaternion(base_orn_pb)[2]

        grasp_wx = base_pos_pb[0] + ARM_FORWARD_OFFSET * math.cos(base_yaw)
        grasp_wy = base_pos_pb[1] + ARM_FORWARD_OFFSET * math.sin(base_yaw)
        grasp_wz = base_pos_pb[2] + ARM_GRASP_Z_BODY

        above_wz = grasp_wz + 0.15
        orn      = p.getQuaternionFromEuler([0, 0, 0])

        tgt_p = ([grasp_wx, grasp_wy, above_wz]
                 if phase in ('stow', 'reach_above') else
                 [grasp_wx, grasp_wy, grasp_wz])
        close = (phase == 'close_gripper')

        # Raise lift
        if lift_joint is not None:
            p.setJointMotorControl2(robot_id, lift_joint,
                                    p.POSITION_CONTROL,
                                    targetPosition=0.3,
                                    force=100, maxVelocity=0.5)

        grasp_object(robot_id, tgt_p, orn,
                     arm_joints=arm_joints or None,
                     close_gripper=close, phase=phase)

        # Keep wheels locked
        _set_wheels(robot_id, 0, 0)

        # Check gripper contact
        contacts = p.getContactPoints(bodyA=robot_id, bodyB=target_id)
        if contacts and len(contacts) > 0:
            contact_detected = True

        # Check arm detachment (base link should stay at ~robot_z)
        rp, _ = p.getBasePositionAndOrientation(robot_id)
        if abs(rp[2] - robot_z) > 0.5:
            arm_detached = True
            print(f"[Test] WARNING: Robot base moved unexpectedly! z={rp[2]:.3f}")

        # Check if any arm link has flown off
        for aj in arm_joints:
            ls = p.getLinkState(robot_id, aj)
            link_world_pos = ls[0]
            dist_from_base = math.dist(link_world_pos, [robot_x, robot_y, robot_z])
            if dist_from_base > 3.0:
                arm_detached = True
                print(f"[Test] ARM DETACHED: joint {aj} is {dist_from_base:.2f}m from base!")
                break

        # Per-second logging
        t_s = int(t_wall)
        if t_s != last_log_s:
            last_log_s = t_s
            gripper_pos = []
            for gj in gripper_joints:
                st = p.getJointState(robot_id, gj)
                gripper_pos.append(f"{st[0]:.3f}")
            print(f"[t={t_s:2d}s | sim={t_sim:.1f}s] phase={phase:14s} "
                  f"contact={'YES' if contact_detected else 'no '} "
                  f"fingers={gripper_pos} "
                  f"grasp_target=({grasp_wx:.2f},{grasp_wy:.2f},{grasp_wz:.2f})")

        p.stepSimulation()
        time.sleep(DT)
        step += 1

        # -- Exit conditions -------------------------------------------------
        if arm_detached:
            print("\n" + "=" * 60)
            print("  RESULT: FAIL - arm detached during grasp")
            print("=" * 60)
            break

        if contact_detected and phase == 'close_gripper' and t_sim > 8.0:
            print("\n" + "=" * 60)
            print("  RESULT: PASS - gripper contact confirmed, arm intact")
            print("=" * 60)
            break

        if t_wall > GRASP_TIMEOUT_S:
            if contact_detected:
                print("\n" + "=" * 60)
                print("  RESULT: PARTIAL PASS - contact detected but grasp timed out")
                print("  (arm did not detach, IK is working, may need orientation tuning)")
                print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print("  RESULT: FAIL - timeout, no gripper contact")
                print("  Check: arm joints moving? IK reaching target?")
                print("=" * 60)
            break

    # Keep window open
    print("\n[Test] Keeping simulation open. Close the PyBullet window to exit.")
    try:
        while p.isConnected():
            p.stepSimulation()
            time.sleep(DT)
    except Exception:
        pass


if __name__ == "__main__":
    main()
