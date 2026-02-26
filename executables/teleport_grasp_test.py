"""
teleport_grasp_test.py
======================================================================
Skips SEARCH and NAVIGATE.
Teleports robot to STANDOFF_M in front of the red cylinder,
faces it directly, forces FSM -> GRASP.

FIX HISTORY:
  F31 - real world Z; orientation [0,pi/2,0]
  F32 - IK XY from ground-truth (tx,ty); orientation [0,0,0]
  F33 - lift=0.0 during grasp (NOT 0.3).
        At lift=0: arm_base_z=0.670, cylinder Z=0.695 -> arm nearly
        horizontal, horizontal reach=0.82m, clamped to MAX_REACH=0.55. OK.
        At lift=0.3: arm_base_z=0.970, must angle 27cm down -> IK fails
        -> joints snap to limits -> arm points up (the screenshot bug).
        IK EE = link 15 (wrist_roll_link, MOVABLE). Z target offset by
        +0.075m (palm length beyond wrist) inside grasp_object().
        Logs: wrist_roll link Z + gripper_base link Z every second.
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

_CYL_Z   = TABLE_HEIGHT + (TARGET_HEIGHT / 2.0) + 0.01   # ~0.695 m
_ABOVE_Z = _CYL_Z + 0.15                                  # ~0.845 m

# Identity orientation: gripper faces +X (forward), fingers close in Y
_GRASP_ORN_EULER = [0.0, 0.0, 0.0]

# Link indices for logging
_WRIST_ROLL_LINK    = 15   # IK end-effector (last movable arm joint)
_GRIPPER_BASE_LINK  = 16   # FIXED joint child - for logging only
# ---------------------------------------------------------------------------


def _set_wheels(robot_id, v=0):
    for i in [0, 1, 2, 3]:
        p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL,
                                targetVelocity=v, force=5000)


def _set_lift(robot_id, lift_joint, pos):
    if lift_joint is not None:
        p.setJointMotorControl2(robot_id, lift_joint,
                                p.POSITION_CONTROL,
                                targetPosition=pos,
                                force=100, maxVelocity=0.5)


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


def _link_pos(robot_id, link_idx):
    try:
        return p.getLinkState(robot_id, link_idx)[0]
    except Exception:
        return None


def main():
    print("=" * 60)
    print("  TELEPORT GRASP TEST  [F33]")
    print(f"  Cylinder Z = {_CYL_Z:.3f} m   Above Z = {_ABOVE_Z:.3f} m")
    print(f"  Lift = 0.0 (arm_base_z ~ 0.670, nearly level with cylinder)")
    print("=" * 60)

    robot_id, table_id, room_id, target_id = build_world(gui=True)
    initialize_state_estimator()
    kb = get_knowledge_base()

    arm_joints, gripper_joints, lift_joint = _detect_joints(robot_id)
    print(f"[Test] Arm joints:    {arm_joints}")
    print(f"[Test] Gripper joints:{gripper_joints}")
    print(f"[Test] Lift joint:    {lift_joint}")
    print(f"[Test] IK EE link:    {_WRIST_ROLL_LINK} (wrist_roll_link, movable)")

    grasp_orn = p.getQuaternionFromEuler(_GRASP_ORN_EULER)
    print(f"[Test] Grasp quat:    {[f'{v:.3f}' for v in grasp_orn]}")

    # Ground-truth target
    tgt_pos_pb, _ = p.getBasePositionAndOrientation(target_id)
    tx, ty, tz = tgt_pos_pb
    print(f"[Test] Cylinder:      ({tx:.3f}, {ty:.3f}, {tz:.3f})")

    # Teleport robot STANDOFF_M directly in front of cylinder
    approach_angle = math.atan2(-ty, -tx)
    robot_x = tx + STANDOFF_M * math.cos(approach_angle)
    robot_y = ty + STANDOFF_M * math.sin(approach_angle)
    face_yaw = math.atan2(ty - robot_y, tx - robot_x)
    orn_q    = p.getQuaternionFromEuler([0, 0, face_yaw])
    p.resetBasePositionAndOrientation(robot_id,
                                      [robot_x, robot_y, 0.1], orn_q)
    print(f"[Test] Robot:         ({robot_x:.3f}, {robot_y:.3f}) "
          f"yaw={math.degrees(face_yaw):.1f} deg")
    print(f"[Test] Dist to cyl:   {math.hypot(tx-robot_x, ty-robot_y):.3f} m")

    # [F33] Set lift to 0 BEFORE settle so physics starts correct
    _set_lift(robot_id, lift_joint, 0.0)
    _set_wheels(robot_id, 0)

    # Stow arm joints to 0 before settle
    for j in arm_joints:
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL,
                                targetPosition=0.0, force=50, maxVelocity=1.0)

    # Settle 1 sim-second
    for _ in range(240):
        _set_lift(robot_id, lift_joint, 0.0)
        p.stepSimulation()
        time.sleep(DT)

    # Log arm_base Z after settle
    bp, _ = p.getBasePositionAndOrientation(robot_id)
    print(f"[Test] After settle - robot base Z: {bp[2]:.3f}")
    ab_pos = _link_pos(robot_id, 8)   # arm_base_joint child link
    if ab_pos:
        print(f"[Test] arm_base link Z: {ab_pos[2]:.3f}  "
              f"(cylinder Z={tz:.3f}, delta={ab_pos[2]-tz:.3f})")

    kb.add_position('target', tx, ty, tz)

    fsm = RobotFSM()
    fsm.transition_to(RobotState.SEARCH)
    fsm.transition_to(RobotState.NAVIGATE)
    fsm.transition_to(RobotState.APPROACH)
    fsm.transition_to(RobotState.GRASP)
    print(f"[Test] FSM: {fsm.state.name}")

    print("\n[Test] Phases:")
    print("  stow         (0-1s):  joints->0, lift=0, no IK")
    print("  reach_above  (1-3s):  IK to (tx,ty, cyl_z+0.15), lift=0")
    print("  reach_target (3-6s):  IK to (tx,ty, cyl_z),      lift=0")
    print("  close_gripper(>6s):   IK hold + fingers close,   lift=0")
    print()

    contact_detected = False
    start_time = time.time()
    last_log_s = -1

    while True:
        fsm.tick()
        t_sim  = fsm.get_time_in_state()
        t_wall = time.time() - start_time

        phase = ('stow'          if t_sim < 1.0 else
                 'reach_above'   if t_sim < 3.0 else
                 'reach_target'  if t_sim < 6.0 else
                 'close_gripper')

        ik_z  = _ABOVE_Z if phase in ('stow', 'reach_above') else _CYL_Z
        tgt_p = [tx, ty, ik_z]
        close = (phase == 'close_gripper')

        # [F33] LIFT = 0.0 always during grasp
        _set_lift(robot_id, lift_joint, 0.0)
        _set_wheels(robot_id, 0)

        grasp_object(robot_id, tgt_p, grasp_orn,
                     arm_joints=arm_joints or None,
                     close_gripper=close,
                     phase=phase)

        contacts = p.getContactPoints(bodyA=robot_id, bodyB=target_id)
        if contacts and len(contacts) > 0:
            contact_detected = True

        t_s = int(t_wall)
        if t_s != last_log_s:
            last_log_s = t_s
            fps = [f"{p.getJointState(robot_id, gj)[0]:.3f}" for gj in gripper_joints]

            wr_pos = _link_pos(robot_id, _WRIST_ROLL_LINK)
            gb_pos = _link_pos(robot_id, _GRIPPER_BASE_LINK)
            wr_str = f"wrist=({wr_pos[0]:.3f},{wr_pos[1]:.3f},{wr_pos[2]:.3f})" if wr_pos else "wrist=N/A"
            gb_str = f"palm=({gb_pos[0]:.3f},{gb_pos[1]:.3f},{gb_pos[2]:.3f})"  if gb_pos else "palm=N/A"

            print(f"[t={t_s:2d}s|sim={t_sim:.1f}s] {phase:13s} "
                  f"ik_z={ik_z:.3f}  contact={'YES' if contact_detected else 'no '}  "
                  f"fingers={fps}  {wr_str}  {gb_str}")

        p.stepSimulation()
        time.sleep(DT)

        if contact_detected and phase == 'close_gripper' and t_sim > 8.0:
            print("\n" + "=" * 60)
            print("  RESULT: PASS - contact confirmed!")
            print("=" * 60)
            break

        if t_wall > GRASP_TIMEOUT_S:
            if contact_detected:
                print("\n" + "=" * 60)
                print("  RESULT: PARTIAL PASS - contact seen, timed out")
                print("  palm Z vs ik_z in logs - adjust _CYL_Z if needed")
                print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print("  RESULT: FAIL - no contact")
                print("  palm Z converging to ik_z? palm XY to cylinder XY?")
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
