"""
teleport_grasp_test.py
======================================================================
Skips SEARCH and NAVIGATE.
Teleports robot to STANDOFF_M in front of the red cylinder,
faces it directly, forces FSM -> GRASP.

FIX HISTORY:
  F31 - real world Z; orientation [0,pi/2,0]
  F32 - IK XY from ground-truth (tx,ty); orientation [0,0,0]
  F33 - lift=0.0 during grasp; IK EE=link15 (wrist_roll, movable);
        Z offset +0.075 for palm length inside grasp_object().
  F34 - STABLE SPAWN FIX:
        p.resetJointState() called on EVERY non-fixed joint immediately
        after teleport, BEFORE any p.stepSimulation(). This is
        instantaneous (no physics) and puts the robot in a tucked,
        stable config so it doesn't tip over on spawn.
        Tuck config: lift=0, shoulder=-0.3 (arm back), elbow=0.5 (folded),
        others=0. CoM stays over wheels.
        Robot spawn Z raised to 0.15 for settle clearance.
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
_GRIPPER_BASE_LINK  = 16   # FIXED joint child - palm, for logging only

# [F34] Tuck joint config: arm folded over torso so CoM stays over wheels.
# shoulder=-0.3 tilts arm backward, elbow=0.5 folds it in.
# wrist_pitch has a 0.4 rad pre-bend in URDF origin - compensate with -0.4.
_TUCK = {
    'lift_joint':        0.0,
    'arm_base_joint':    0.0,
    'shoulder_joint':   -0.3,
    'elbow_joint':       0.5,
    'wrist_pitch_joint': -0.4,  # cancels URDF origin rpy="0 0.4 0" pre-bend
    'wrist_roll_joint':  0.0,
    'left_finger_joint': 0.0,
    'right_finger_joint':0.0,
}
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
    """Returns (arm_joints, gripper_joints, lift_joint, joint_name_to_idx)."""
    arm_joints, gripper_joints = [], []
    lift_joint = None
    name_to_idx = {}
    for i in range(p.getNumJoints(robot_id)):
        info  = p.getJointInfo(robot_id, i)
        jname = info[1].decode('utf-8')
        jtype = info[2]
        name_to_idx[jname] = i
        if jname in ('arm_base_joint', 'shoulder_joint', 'elbow_joint',
                     'wrist_pitch_joint', 'wrist_roll_joint'):
            if jtype != p.JOINT_FIXED:
                arm_joints.append(i)
        if 'left_finger_joint' in jname or 'right_finger_joint' in jname:
            gripper_joints.append(i)
        if jname == 'lift_joint' and jtype != p.JOINT_FIXED:
            lift_joint = i
    return arm_joints, gripper_joints, lift_joint, name_to_idx


def _reset_joints_to_tuck(robot_id, name_to_idx):
    """
    [F34] Instantly set all arm/lift joints to tuck config using
    p.resetJointState (no physics, immediate). Must be called BEFORE
    any p.stepSimulation() after teleporting the robot.
    This prevents CoM shift from flinging the robot over.
    """
    for jname, jpos in _TUCK.items():
        if jname in name_to_idx:
            idx = name_to_idx[jname]
            p.resetJointState(robot_id, idx, jpos, targetVelocity=0.0)


def _link_pos(robot_id, link_idx):
    try:
        return p.getLinkState(robot_id, link_idx)[0]
    except Exception:
        return None


def main():
    print("=" * 60)
    print("  TELEPORT GRASP TEST  [F34]")
    print(f"  Cylinder Z = {_CYL_Z:.3f} m   Above Z = {_ABOVE_Z:.3f} m")
    print(f"  Lift = 0.0 during grasp")
    print(f"  IK EE = link 15 (wrist_roll_link, movable)")
    print("=" * 60)

    robot_id, table_id, room_id, target_id = build_world(gui=True)
    initialize_state_estimator()
    kb = get_knowledge_base()

    arm_joints, gripper_joints, lift_joint, name_to_idx = _detect_joints(robot_id)
    print(f"[Test] Arm joints:    {arm_joints}")
    print(f"[Test] Gripper joints:{gripper_joints}")
    print(f"[Test] Lift joint:    {lift_joint}")
    print(f"[Test] All joints:    {name_to_idx}")

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

    # [F34] Spawn at Z=0.15 for settle clearance
    p.resetBasePositionAndOrientation(robot_id,
                                      [robot_x, robot_y, 0.15], orn_q)
    print(f"[Test] Robot:         ({robot_x:.3f}, {robot_y:.3f}) "
          f"yaw={math.degrees(face_yaw):.1f} deg  Z=0.15")
    print(f"[Test] Dist to cyl:   {math.hypot(tx-robot_x, ty-robot_y):.3f} m")

    # [F34] CRITICAL: reset ALL joints to tuck config BEFORE any stepSimulation.
    # resetJointState is instantaneous - bypasses physics completely.
    # This prevents the arm's default pose from shifting CoM and tipping robot.
    _reset_joints_to_tuck(robot_id, name_to_idx)
    print("[Test] Joints reset to tuck config (instantaneous, pre-physics)")

    # Wheels stopped
    _set_wheels(robot_id, 0)

    # Settle physics for 1 sim-second, holding lift=0
    print("[Test] Settling physics (240 steps)...")
    for _ in range(240):
        _set_lift(robot_id, lift_joint, 0.0)
        _set_wheels(robot_id, 0)
        p.stepSimulation()
        time.sleep(DT)

    # Log actual positions after settle
    bp, _ = p.getBasePositionAndOrientation(robot_id)
    print(f"[Test] After settle - robot base Z: {bp[2]:.3f} (should be ~0.1)")
    wr_pos = _link_pos(robot_id, _WRIST_ROLL_LINK)
    gb_pos = _link_pos(robot_id, _GRIPPER_BASE_LINK)
    if wr_pos:
        print(f"[Test] wrist_roll link Z: {wr_pos[2]:.3f}")
    if gb_pos:
        print(f"[Test] palm (gripper_base) Z: {gb_pos[2]:.3f}")
    print(f"[Test] Cylinder Z: {tz:.3f}")

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

        # [F33/F34] LIFT = 0.0 always during grasp
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
            wr_str = (f"wrist=({wr_pos[0]:.3f},{wr_pos[1]:.3f},{wr_pos[2]:.3f})"
                      if wr_pos else "wrist=N/A")
            gb_str = (f"palm=({gb_pos[0]:.3f},{gb_pos[1]:.3f},{gb_pos[2]:.3f})"
                      if gb_pos else "palm=N/A")
            cyl_str = f"cyl=({tx:.3f},{ty:.3f},{tz:.3f})"

            print(f"[t={t_s:2d}s|sim={t_sim:.1f}s] {phase:13s} "
                  f"ik_z={ik_z:.3f}  contact={'YES' if contact_detected else 'no '}  "
                  f"fingers={fps}  {wr_str}  {gb_str}  {cyl_str}")

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
                print("  Check palm Z vs ik_z in logs above")
                print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print("  RESULT: FAIL - no contact after 30s")
                print("  Check: palm Z converging to ik_z?")
                print("  Check: palm XY converging to cylinder XY?")
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
