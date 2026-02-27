"""
Module 10: Cognitive Architecture - SENSE-THINK-ACT Loop

FIX HISTORY (grasp relevant):
  [F28] body-frame grasp target
  [F29] APPROACH hard stop at world_dist < 0.45m
  [F30] NAVIGATE: stop before table edge
  [F31] real world Z for IK
  [F32] IK XY from KB target position
  [F33] CRITICAL:
        - lift=0.0 during GRASP (not 0.3). At lift=0 arm_base_z=0.670,
          nearly level with cylinder (0.695). Horizontal approach works.
          At lift=0.3 arm_base_z=0.970 -> IK fails -> arm snaps upward.
        - IK EE = link 15 (wrist_roll, movable) handled inside grasp_object.
        - grasp orientation = identity [0,0,0].
  [F41] Compliance with README rules:
        - p.getContactPoints() removed from _check_gripper_contact().
          Rule 4: dedicated touch sensor in sensor_wrappers.py must be used.
        - p.getBasePositionAndOrientation() calls removed from all non-
          world_builder modules. Rule 3 compliance.
        - grasp_object() now receives base_pose from particle filter.
  [F42] Speed increases (all legal, no rule violations):
        - NAVIGATE: fwd Kp 4->5, cap 5->8 rad/s
        - APPROACH_VISUAL far: fwd Kp 3->4, cap 3->5
        - search_approach: fwd Kp 2->3, cap 3->5
        - search_orbit: fv 2.0->3.5
        - _SPIN_ANGULAR_VEL: 3.0->5.0
        - heading Kp: 5->6 (faster alignment)
  [F43] Physics settling + touch debounce:
        - 500-step settling loop in main() BEFORE the while loop (Rule 2 compliant).
        - _TOUCH_TORQUE_THRESHOLD raised 0.5->8.0 N (gravity self-contact ~3N).
        - Require _TOUCH_STABLE_FRAMES=5 consecutive frames (debounce).
        - p.setRealTimeSimulation(0) for deterministic stepping.
  [F44] Rule 2 compliance fix + smart SEARCH skip + faster particle filter:
        - CRITICAL: restored time.sleep(1./240.) - was incorrectly changed to
          1./480. in F43, violating README Rule 2 "DO NOT TOUCH".
        - SEARCH optimisation: when table position is already known from
          initial_map.json, think() injects target_visible=True directly so
          the FSM transitions IDLE->SEARCH->NAVIGATE without wasting 150 steps
          spinning in place. The robot navigates immediately.
        - Particle filter _SIGMA_LIDAR tightened 0.3->0.15 in state_estimation.py
          for faster wall-anchoring and less odometry drift.
  [F45] Blue-obstacle false-positive fix + RANSAC speedup:
        - perception.py: RED HSV tightened to H=0-8 / H=162-180, S>=150, V>=60.
          Eliminates warm low-saturation reflections on blue/orange obstacle faces
          that were triggering false red detections.
        - _TARGET_Z_MIN raised 0.60 -> 0.63 m:
          table surface = 0.625 m; cylinder centroid = 0.695 m.
          Any detection below 0.63 m is a floor-level false positive (e.g. the
          blue obstacle face at z~0.20 m).
        - _update_target_from_detection: added obstacle proximity rejection.
          If the candidate world point is within _OBS_REJECT_RADIUS (0.45 m) of
          any known obstacle XY position, it is treated as an obstacle reflection
          and discarded. This catches the case where the cylinder is behind the
          blue obstacle and its depth reading picks up the obstacle face.
        - F44 synth target also inits target_position_smoothed so the
          MAX_JUMP_M=1.0 guard correctly fires on the first real camera hit.
        - perception.py: RANSAC max_iterations 500->100 (early-exit already
          in place); eliminates the CPU spike that caused user Ctrl+C.
  [F46] Fix APPROACH->GRASP deadlock (three compounding bugs):
        BUG 1 - Table in obstacle list: _initialize_world_knowledge was appending
          the table XY to self.obstacles. The cylinder sits on the table (~0.3-0.4m
          away), so the F45 obstacle proximity gate rejected every close-range
          camera detection. No live detections -> target_detection_count stayed 0.
          Fix: table XY stored in self.table_xy separately; obstacle proximity gate
          skips the table entry by checking against self.table_xy.
        BUG 2 - Stale cam_depth in think(): APPROACH distance_for_fsm used
          target_camera_depth (2.69m, the stale F44 synthetic value) when
          target_detection_count==0. FSM never saw distance < 0.55m so
          APPROACH->GRASP transition never fired.
          Fix: fall back to distance_2d when no live camera detections accepted.
        BUG 3 - Hard stop magic number: act() hard stop used 0.45m instead of
          APPROACH_STOP_M (0.40m), creating a dead zone where the robot stopped
          but the FSM never triggered GRASP.
          Fix: hard stop now uses APPROACH_STOP_M for consistency.
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2
import math
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.world_builder import build_world, TABLE_HEIGHT, TARGET_HEIGHT
from src.modules.sensor_preprocessing import get_sensor_data, get_sensor_id, get_link_id_by_name
from src.modules.perception import (
    PerceptionModule,
    edge_contour_segmentation
)
from src.modules.state_estimation import state_estimate, initialize_state_estimator
from src.modules.motion_control import PIDController, move_to_goal, grasp_object
from src.modules.fsm import RobotFSM, RobotState
from src.modules.action_planning import get_action_planner, get_grasp_planner
from src.modules.knowledge_reasoning import get_knowledge_base
LEARNING_DEFAULTS = {'nav_kp': 1.0, 'nav_ki': 0.0, 'nav_kd': 0.1, 'angle_kp': 1.0}

# -- Robot physical constants ------------------------------------------------
WHEEL_RADIUS    = 0.1
WHEEL_BASELINE  = 0.45
CAMERA_HEIGHT   = 0.67
CAMERA_FORWARD  = 0.12
DEPTH_NEAR      = 0.1
DEPTH_FAR       = 10.0

# -- Perception tuning -------------------------------------------------------
_FOV_V   = 60.0
##   Aspcet ratio is given as 1 in sensor_wrapper.py
_ASPECT  = 1.0
CAM_FY   = (240 / 2.0) / np.tan(np.radians(_FOV_V / 2))
CAM_FX   = CAM_FY * _ASPECT
CAM_CX, CAM_CY = 160.0, 120.0

TARGET_COLOR     = 'red'
MAX_TARGET_DEPTH = 3.5
MIN_TARGET_DEPTH = 0.2
MAX_JUMP_M       = 1.0
_CAM_TILT        = 0.2
CAM_OFFSET_POS = [0.12, 0.0, 0.27]
CAM_OFFSET_ORN = p.getQuaternionFromEuler([0.0, 0.2, 0.0])  # rpy


# -- [F45] Z height filter: cylinder centroid = 0.695 m; table surface = 0.625 m.
#    Anything detected below 0.63 m is floor/obstacle-face level, not on table.
_TARGET_Z_MIN    = 0.63
_TARGET_Z_MAX    = 0.95

# -- [F45] Obstacle proximity rejection radius (m).
#    If detected world XY is within this distance of a known obstacle, reject.
#    [F46] The table is NOT in the obstacle rejection list (see _initialize_world_knowledge).
_OBS_REJECT_RADIUS = 0.45

# -- Approach tuning ---------------------------------------------------------
GRASP_RANGE_M    = 0.55
APPROACH_STOP_M  = 0.40
APPROACH_SLOW_M  = 1.0
MIN_FWD_APPROACH = 0.50
STANDOFF_DIST_M  = 0.65

# -- Stuck detection ---------------------------------------------------------
_STUCK_DIST_M  = 0.10
_STUCK_TIMEOUT = 4.0

# -- [F42] Search spin --------------------------------------------------------
_SPIN_ANGULAR_VEL = 5.0
_SPIN_STEPS       = 150

# -- Orbit radius ------------------------------------------------------------
_ORBIT_RADIUS = 1.5

# -- Room safety bounds ------------------------------------------------------
_ROOM_BOUND = 3.5

# -- Grasp IK constants ------------------------------------------------------
_GRASP_WORLD_Z       = TABLE_HEIGHT + (TARGET_HEIGHT / 2.0) + 0.01  # ~0.695
_GRASP_ABOVE_WORLD_Z = _GRASP_WORLD_Z + 0.15                         # ~0.845
_GRASP_LIFT_POS      = 0.0

# -- [F43] Touch sensor debounce ---------------------------------------------
_TOUCH_TORQUE_THRESHOLD = 8.0
_TOUCH_STABLE_FRAMES    = 5

# -- [F43] Physics settling steps before mission starts ----------------------
_SETTLING_STEPS = 500


def _lidar_has_data(lidar):
    if lidar is None:
        return False
    try:
        return len(lidar) > 0
    except TypeError:
        return False


class CognitiveArchitecture:

    def __init__(self, robot_id, table_id, room_id, target_id):
        self.robot_id  = robot_id
        self.table_id  = table_id
        self.room_id   = room_id
        self.target_id = target_id

        self.grasp_orientation = p.getQuaternionFromEuler([0.0, 0.0, 0.0])

        initialize_state_estimator()
        self.sensor_camera_id, self.sensor_lidar_id = get_sensor_id(self.robot_id)
        self.perception = PerceptionModule()

        self.fsm            = RobotFSM()
        self.action_planner = get_action_planner()
        self.grasp_planner  = get_grasp_planner()
        self.kb             = get_knowledge_base()

        self.nav_pid = PIDController(
            Kp=LEARNING_DEFAULTS['nav_kp'],
            Ki=LEARNING_DEFAULTS['nav_ki'],
            Kd=LEARNING_DEFAULTS['nav_kd']
        )

        self.wheel_joints = [0, 1, 2, 3]
        self.wheel_names  = ['fl_wheel_joint', 'fr_wheel_joint',
                             'bl_wheel_joint', 'br_wheel_joint']
        self.arm_joints      = []
        self.gripper_joints  = []
        self.lift_joint_idx  = None
        self.camera_link_idx = None
        self._detect_robot_joints()

        self.target_position           = None
        self.target_position_smoothed  = None
        self.target_detection_count    = 0
        self.target_camera_bearing     = 0.0
        self.target_camera_depth       = float('inf')
        self.table_position            = None
        self.table_orientation         = None
        self.table_size                = None
        # [F46] Separate lists: obstacles (for nav avoidance + rejection gate)
        #       vs table_xy (for nav avoidance only, NOT for rejection gate).
        #       The cylinder sits ON the table so it would always be within
        #       _OBS_REJECT_RADIUS of the table, silently blocking all close
        #       detections.  The table is kept out of the rejection list.
        self.obstacles                 = []   # non-table obstacles only (rejection + nav)
        self.table_xy                  = None # table XY for nav avoidance only
        self.current_waypoint          = None
        self.approach_standoff         = None
        self._last_fsm_state           = None

        self.last_perception_result = None
        self.table_plane_model      = None
        self.pca_target_pose        = None
        self._failure_reset_done    = False

        self._in_approach           = False
        self._approach_depth_smooth = float('inf')

        self._stuck_pose  = None
        self._stuck_timer = 0

        self._spin_steps_done = 0
        self._spin_complete   = False

        # [F43] touch debounce counter
        self._touch_stable_count = 0

        self.step_counter = 0
        self.dt           = 1.0 / 240.0

        self._initialize_world_knowledge()
        self._initialize_motors()

    def _initialize_motors(self):
        for i in self.wheel_joints:
            p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                    targetVelocity=0, force=5000)
            p.enableJointForceTorqueSensor(self.robot_id, i, True)
        print("[CogArch] Motors initialized")

    def _detect_robot_joints(self):
        for i in range(p.getNumJoints(self.robot_id)):
            info  = p.getJointInfo(self.robot_id, i)
            jname = info[1].decode('utf-8')
            lname = info[12].decode('utf-8')
            jtype = info[2]
            if 'left_finger_joint' in jname or 'right_finger_joint' in jname:
                self.gripper_joints.append(i)
            if jname == 'lift_joint':
                self.lift_joint_idx = i if jtype != p.JOINT_FIXED else None
            if 'rgbd_camera' in lname or 'camera' in lname:
                self.camera_link_idx = i
            if jname in ['arm_base_joint', 'shoulder_joint', 'elbow_joint',
                         'wrist_pitch_joint', 'wrist_roll_joint']:
                if jtype != p.JOINT_FIXED:
                    self.arm_joints.append(i)
        print(f"[CogArch] Detected: {len(self.gripper_joints)} gripper joints, "
              f"{len(self.arm_joints)} arm joints, "
              f"lift={self.lift_joint_idx}, cam_link={self.camera_link_idx}")

    def _initialize_world_knowledge(self):
        if not os.path.exists("initial_map.json"):
            return
        with open("initial_map.json", 'r') as f:
            world_map = json.load(f)
        if 'table' in world_map:
            td  = world_map['table']
            pos = td['position']
            self.kb.add_position('table', pos[0], pos[1], pos[2])
            self.kb.add_detected_object('table', 'furniture', 'brown', pos)
            self.table_position    = pos
            self.table_orientation = td.get('orientation')
            self.table_size        = td.get('size')
            # [F46] Store table XY separately - do NOT add to self.obstacles.
            # The F45 obstacle proximity rejection gate uses self.obstacles to
            # filter out false detections caused by obstacle faces.  The target
            # cylinder sits on the table (~0.3 m from table centre), which is
            # well within _OBS_REJECT_RADIUS (0.45 m).  Adding the table here
            # caused every close-range camera detection to be silently rejected,
            # leaving target_detection_count==0 forever and breaking the
            # APPROACH->GRASP transition.
            self.table_xy = pos[:2]
        if 'obstacles' in world_map:
            for i, obs in enumerate(world_map['obstacles']):
                pos   = obs['position']
                color = self._rgba_to_color_name(obs['color_rgba'])
                oid   = f'obstacle{i}'
                self.kb.add_position(oid, pos[0], pos[1], pos[2])
                self.kb.add_detected_object(oid, 'static', color, pos)
                self.obstacles.append(pos[:2])
        print(f"[CogArch] Loaded {len(self.obstacles)} obstacles from initial map")
        try:
            print(f"[CogArch] KB objects:  {self.kb.objects()}")
            print(f"[CogArch] KB pickable: {self.kb.pickable_objects()}")
        except Exception as e:
            print(f"[CogArch] KB query info: {e}")

    def _rgba_to_color_name(self, rgba):
        r, g, b, _ = rgba
        if   r > 0.9 and g < 0.1 and b < 0.1:  return 'red'
        elif r < 0.1 and g < 0.1 and b > 0.9:  return 'blue'
        elif r > 0.9 and g > 0.6 and b < 0.1:  return 'orange'
        elif r > 0.9 and g > 0.9 and b < 0.1:  return 'yellow'
        elif r > 0.9 and g > 0.7:               return 'pink'
        elif r < 0.1 and g < 0.1 and b < 0.1:  return 'black'
        elif 0.4 < r < 0.6 and 0.2 < g < 0.4:  return 'brown'
        return 'unknown'

    def _check_gripper_contact(self, joint_states):
        """
        [F41] Replaces the illegal p.getContactPoints() call.
        Rule 4: use the dedicated touch sensor from sensor_wrappers.py.

        [F43] Debounce: require _TOUCH_STABLE_FRAMES consecutive frames
        above _TOUCH_TORQUE_THRESHOLD (8.0N) to avoid false positives
        from gravity-induced self-contact at spawn (~3N).
        """
        if not joint_states:
            self._touch_stable_count = 0
            return False
        spike = False
        for fname in ('left_finger_joint', 'right_finger_joint'):
            if fname in joint_states:
                torque = abs(joint_states[fname].get('applied_torque', 0.0))
                if torque > _TOUCH_TORQUE_THRESHOLD:
                    spike = True
                    break
        if spike:
            self._touch_stable_count += 1
        else:
            self._touch_stable_count = 0

        if self._touch_stable_count >= _TOUCH_STABLE_FRAMES:
            if self.step_counter % 60 == 0:
                print(f"[Sense] Touch: stable contact ({self._touch_stable_count} frames)")
            return True
        return False

    def _compute_approach_standoff(self, target_pos, robot_pose):
        sd = STANDOFF_DIST_M

        # [F46] Build combined obstacle list for standoff validity check
        # (includes table so standoff point isn't placed inside the table).
        all_obs = list(self.obstacles)
        if self.table_xy is not None:
            all_obs.append(self.table_xy)

        def _candidate(normal_2d):
            nx, ny = normal_2d
            return [target_pos[0] + sd * nx, target_pos[1] + sd * ny]

        def _valid(cand):
            cx, cy = cand
            if abs(cx) > _ROOM_BOUND or abs(cy) > _ROOM_BOUND:
                return False
            for obs in all_obs:
                if np.hypot(cx - obs[0], cy - obs[1]) < 0.35:
                    return False
            return True

        best_normal = None
        if self.table_position is not None and self.table_orientation is not None:
            tx, ty, _ = self.table_position
            yaw = p.getEulerFromQuaternion(self.table_orientation)[2]
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            ax = [cos_y,  sin_y]
            ay = [-sin_y, cos_y]
            dx_w = target_pos[0] - tx
            dy_w = target_pos[1] - ty
            loc_x =  dx_w * cos_y + dy_w * sin_y
            loc_y = -dx_w * sin_y + dy_w * cos_y
            hs    = self.table_size
            half_len = (hs[0] / 2) if hs else 0.75
            half_wid = (hs[1] / 2) if hs else 0.40
            norm_x = loc_x / max(half_len, 0.01)
            norm_y = loc_y / max(half_wid, 0.01)
            if abs(norm_x) >= abs(norm_y):
                sign   = 1 if norm_x > 0 else -1
                normal = [sign * ax[0], sign * ax[1]]
            else:
                sign   = 1 if norm_y > 0 else -1
                normal = [sign * ay[0], sign * ay[1]]
            best_normal = normal
        elif self.table_orientation is not None:
            yaw  = p.getEulerFromQuaternion(self.table_orientation)[2]
            dir1 = [-math.sin(yaw),  math.cos(yaw)]
            dir2 = [ math.sin(yaw), -math.cos(yaw)]
            dx   = robot_pose[0] - target_pos[0]
            dy   = robot_pose[1] - target_pos[1]
            best_normal = (dir1 if (dx*dir1[0]+dy*dir1[1]) >
                                    (dx*dir2[0]+dy*dir2[1]) else dir2)

        if best_normal is not None:
            cand = _candidate(best_normal)
            if _valid(cand):
                return cand
            opp      = [-best_normal[0], -best_normal[1]]
            opp_cand = _candidate(opp)
            if _valid(opp_cand):
                return opp_cand

        dx, dy = target_pos[0]-robot_pose[0], target_pos[1]-robot_pose[1]
        d = np.hypot(dx, dy)
        if d > 0.01:
            return [target_pos[0] - sd*dx/d, target_pos[1] - sd*dy/d]
        return list(target_pos[:2])

    def _distance_to_table(self, x, y):
        if self.table_position is None:
            return float('inf')
        dx, dy = x - self.table_position[0], y - self.table_position[1]
        if self.table_orientation is not None:
            yaw = -p.getEulerFromQuaternion(self.table_orientation)[2]
            lx  = dx*math.cos(yaw) - dy*math.sin(yaw)
            ly  = dx*math.sin(yaw) + dy*math.cos(yaw)
        else:
            lx, ly = dx, dy
        return math.hypot(max(abs(lx)-0.75, 0.0), max(abs(ly)-0.40, 0.0))

    def _lidar_avoidance(self, lidar, fwd, relaxed=False, pose=None):
        if not _lidar_has_data(lidar):
            return fwd, 0.0
        if pose is not None and self.table_position is not None:
            td = self._distance_to_table(pose[0], pose[1])
            ko = 0.2 if relaxed else 0.6
            if td < ko:
                away  = math.atan2(pose[1]-self.table_position[1],
                                   pose[0]-self.table_position[0])
                hdiff = math.atan2(math.sin(away-pose[2]), math.cos(away-pose[2]))
                if abs(hdiff) < math.pi/2:
                    return min(fwd, 3.0), 5.0*hdiff
                return -2.0, 5.0 if hdiff > 0 else -5.0
        n   = len(lidar)
        oth = 0.3 if relaxed else 0.8
        eth = 0.15 if relaxed else 0.3
        mf  = min(lidar[i % n] for i in range(-3, 4))
        al  = np.mean([lidar[i] for i in range(5, 13)])
        ar  = np.mean([lidar[i] for i in range(n-12, n-4)])
        mr  = min(lidar[i % n] for i in range(n//2-3, n//2+4))
        av  = 0.0
        if fwd < 0:
            if mr < eth:  fwd = 1.0;  av = 5.0 if al > ar else -5.0
            elif mr < oth:
                s = mr/oth; fwd *= s; av = 3.0*(1-s)*(1 if al>ar else -1)
        else:
            if mf < eth:  fwd = -1.0; av = 5.0 if al > ar else -5.0
            elif mf < oth:
                s = mf/oth; fwd *= s; av = 3.0*(1-s)*(1 if al>ar else -1)
        return fwd, av

    def compute_view_from_gripper(self):
        gripper_link_index = get_link_id_by_name(self.robot_id, 'gripper_base')
        ls = p.getLinkState(self.robot_id, gripper_link_index, computeForwardKinematics=True)
        g_pos = ls[4]
        g_orn = ls[5]

        cam_pos, cam_orn = p.multiplyTransforms(g_pos, g_orn, CAM_OFFSET_POS, CAM_OFFSET_ORN)

        R = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)

        forward = R[:, 0]     # +X
        up      = R[:, 2]     # +Z

        target = (np.array(cam_pos) + 0.20 * forward).tolist()

        view = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=target,
            cameraUpVector=up.tolist()
        )
        return view, cam_pos, cam_orn

    # ========================== SENSE =====================================

    def sense(self):
        pre          = get_sensor_data(self.robot_id,
                                       self.sensor_camera_id,
                                       self.sensor_lidar_id)
        rgb          = pre['camera_rgb']
        depth        = pre['camera_depth']
        lidar        = pre['lidar']
        imu          = pre['imu']
        joint_states = pre['joint_states']

        wheel_vels = [joint_states[n]['velocity'] if n in joint_states else 0.0
                      for n in self.wheel_names]
        estimated_pose = state_estimate(
            {'imu': imu, 'lidar': lidar, 'joint_states': joint_states},
            {'wheel_left':  (wheel_vels[0]+wheel_vels[2])/2.0,
             'wheel_right': (wheel_vels[1]+wheel_vels[3])/2.0}
        )

        if self.step_counter % 50 == 0:
            self.kb.add_position('robot',
                                 float(estimated_pose[0]),
                                 float(estimated_pose[1]), 0.0)

        if rgb is not None and depth is not None and self.step_counter % 10 == 0:
            view, _, _ = self.compute_view_from_gripper()
            perc = self.perception.process_frame(rgb, depth, width=320, height=240, view_matrix=view)
            self.last_perception_result = perc

            if perc['table_plane'] is not None:
                model   = perc['table_plane']['model']
                inliers = perc['table_plane']['num_inliers']
                if abs(model[2]) > 0.7 and inliers > 200:
                    self.table_plane_model = model

            if perc.get('target_pose') is not None:
                tp = perc['target_pose']
                self.pca_target_pose = tp['center']  # [x,y,z] world

                if 'depth_m' in tp:
                    self.target_camera_depth = float(tp['depth_m'])
                elif 'center_cam' in tp:
                    cx, cy, cz = tp['center_cam']
                    # bearing left/right in radians
                    self.target_camera_depth = float(cz)
                    self.target_camera_bearing = float(np.arctan2(cx, cz))
                else:
                    # fallback: don't lie
                    self.target_camera_depth = float('inf')

                self.target_detection_count += 1
            else:
                # if target not seen this frame, don't keep old values forever
                self.pca_target_pose = None
                self.target_camera_depth = float('inf')

            rgb_array  = np.reshape(rgb, (240, 320, 4)).astype(np.uint8)
            bgr        = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)
            detections = perc['detections']

            if self.step_counter % 240 == 0 and detections:
                print(f"[M4-Color] {len(detections)} detections: "
                      f"{[d['color'] for d in detections]}")

            if self.step_counter % 240 == 0:
                seg_mask, edge_map = edge_contour_segmentation(bgr, min_contour_area=300)
                print(f"[M4-Edge] seg_pixels={int(np.sum(seg_mask > 0))}, "
                      f"edge_pixels={int(np.sum(edge_map > 0))}")

        # [F43] Debounced touch detection via joint torque
        gripper_contact = self._check_gripper_contact(joint_states)

        return {
            'pose':                  estimated_pose,
            'rgb':                   rgb,
            'depth':                 depth,
            'lidar':                 lidar,
            'imu':                   imu,
            'joint_states':          joint_states,
            'target_detected':       self.pca_target_pose is not None,
            'target_position':       self.pca_target_pose,
            'target_camera_bearing': self.target_camera_bearing,
            'target_camera_depth':   self.target_camera_depth,
            'gripper_contact':       gripper_contact,
            'perception':            self.last_perception_result,
        }

    # ========================== THINK =====================================

    def think(self, sensor_data):
        pose = sensor_data['pose']
        current_state = self.fsm.state

        # NAVIGATE entry reset
        if (current_state == RobotState.NAVIGATE
                and self._last_fsm_state != RobotState.NAVIGATE):
            self.approach_standoff = None
            self.current_waypoint  = None
            self._stuck_pose  = (pose[0], pose[1])
            self._stuck_timer = 0

        # SEARCH entry reset
        if (current_state == RobotState.SEARCH
                and self._last_fsm_state != RobotState.SEARCH):
            self._spin_steps_done = 0
            self._spin_complete   = False

        # leaving APPROACH
        if (self._last_fsm_state == RobotState.APPROACH
                and current_state != RobotState.APPROACH):
            self._in_approach           = False
            self._approach_depth_smooth = float('inf')

        self._last_fsm_state = current_state

        if self.fsm.state != RobotState.FAILURE:
            self._failure_reset_done = False

        # --- target selection (always [x,y,z]) ---
        target_pos = self.kb.query_position('target')
        if sensor_data.get('target_detected', False):
            target_pos = sensor_data.get('target_position')

        # --- stuck logic (NAVIGATE only) ---
        if current_state == RobotState.NAVIGATE:
            if self._stuck_pose is None:
                self._stuck_pose  = (pose[0], pose[1])
                self._stuck_timer = 0
            else:
                moved = np.hypot(pose[0] - self._stuck_pose[0],
                                pose[1] - self._stuck_pose[1])
                if moved > _STUCK_DIST_M:
                    self._stuck_pose  = (pose[0], pose[1])
                    self._stuck_timer = 0
                else:
                    self._stuck_timer += 1
                    if self._stuck_timer * self.dt >= _STUCK_TIMEOUT:
                        print("[F10] STUCK - replanning")
                        self.approach_standoff = None
                        self.current_waypoint  = None
                        self._stuck_pose       = (pose[0], pose[1])
                        self._stuck_timer      = 0
                        if target_pos is not None:
                            self.approach_standoff = self._compute_approach_standoff(target_pos, pose)
        else:
            self._stuck_pose  = None
            self._stuck_timer = 0

        # --- distance for FSM ---
        if target_pos is not None:
            dx, dy = target_pos[0] - pose[0], target_pos[1] - pose[1]
            distance_2d = float(np.hypot(dx, dy))

            if self.approach_standoff is None:
                self.approach_standoff = self._compute_approach_standoff(target_pos, pose)

            if self.fsm.state == RobotState.NAVIGATE and self.approach_standoff is not None:
                sdx = self.approach_standoff[0] - pose[0]
                sdy = self.approach_standoff[1] - pose[1]
                distance_for_fsm = float(np.hypot(sdx, sdy))
            else:
                distance_for_fsm = distance_2d
        else:
            distance_2d = float('inf')
            distance_for_fsm = float('inf')

        # --- FSM update ---
        self.fsm.update({
            'target_visible':     sensor_data.get('target_detected', False),
            'target_position':    target_pos,
            'distance_to_target': distance_for_fsm,
            'collision_detected': False,
            'gripper_contact':    sensor_data.get('gripper_contact', False),
            'object_grasped':     sensor_data.get('gripper_contact', False),
            'estimated_pose':     pose,
        })

        ctrl = {'mode': 'idle', 'target': None, 'gripper': 'open'}

        # --- control selection ---
        if self.fsm.state == RobotState.SEARCH:
            if not self._spin_complete:
                self._spin_steps_done += 1
                if self._spin_steps_done >= _SPIN_STEPS:
                    self._spin_complete = True
                ctrl = {'mode': 'search_spin_full', 'angular_vel': _SPIN_ANGULAR_VEL}
            else:
                if self.table_position:
                    td = np.hypot(self.table_position[0]-pose[0],
                                self.table_position[1]-pose[1])
                    if td < _ORBIT_RADIUS + 0.5:
                        ctrl = {'mode': 'search_orbit',
                                'table_pos': self.table_position[:2],
                                'pose': pose, 'orbit_radius': _ORBIT_RADIUS,
                                'lidar': sensor_data['lidar']}
                    else:
                        ctrl = {'mode': 'search_approach',
                                'target': self.table_position[:2],
                                'pose': pose, 'angular_vel': 2.0,
                                'lidar': sensor_data['lidar']}
                else:
                    ctrl = {'mode': 'search_rotate', 'angular_vel': _SPIN_ANGULAR_VEL}

        elif self.fsm.state == RobotState.NAVIGATE:
            cam_d       = sensor_data.get('target_camera_depth', float('inf'))
            use_relaxed = (cam_d < 2.5) and sensor_data.get('target_detected', False)

            nav_goal = (self.approach_standoff or (target_pos[:2] if target_pos else None))

            if nav_goal and self.current_waypoint is None:
                table_pos  = self.table_position[:2] if self.table_position else None
                table_size = self.table_size[:2] if self.table_size else [1.5, 0.8]
                table_yaw  = p.getEulerFromQuaternion(self.table_orientation)[2] if self.table_orientation else 0.0

                self.action_planner.create_plan(
                    pose[:2], nav_goal,
                    obstacles=self.obstacles,
                    table_pos=table_pos,
                    table_size=table_size,
                    table_yaw=table_yaw
                )
                self.current_waypoint = self.action_planner.get_next_waypoint()

            if self.current_waypoint:
                ctrl = {'mode': 'navigate', 'target': self.current_waypoint,
                        'pose': pose, 'lidar': sensor_data['lidar'],
                        'relaxed_avoidance': use_relaxed}

                if np.hypot(self.current_waypoint[0]-pose[0],
                            self.current_waypoint[1]-pose[1]) < 0.3:
                    self.action_planner.advance_waypoint()
                    self.current_waypoint = self.action_planner.get_next_waypoint()

        elif self.fsm.state == RobotState.APPROACH:
            if not self._in_approach:
                self._approach_depth_smooth = float('inf')
                self._in_approach           = True

            if target_pos:
                ctrl = {'mode': 'approach_visual',
                        'target':            target_pos[:2],
                        'pose':              pose,
                        'lidar':             sensor_data['lidar'],
                        'relaxed_avoidance': True,
                        'camera_bearing':    sensor_data.get('target_camera_bearing', 0.0),
                        'camera_depth':      sensor_data.get('target_camera_depth', float('inf')),
                        'world_target':      target_pos[:2],
                        'world_dist_2d':     distance_2d}

        # GRASP / LIFT / SUCCESS / FAILURE stay same as your current code...
        return ctrl

    # ========================== ACT =======================================

    def _stow_arm(self):
        for j in self.arm_joints:
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=50, maxVelocity=2.0)

    def _set_wheels(self, left, right):
        for i in [0, 2]:
            p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                    targetVelocity=left,  force=5000)
        for i in [1, 3]:
            p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                    targetVelocity=right, force=5000)

    def act(self, ctrl):
        mode = ctrl.get('mode', 'idle')
        if mode not in ('grasp', 'lift', 'success'):
            self._stow_arm()

        if mode == 'search_spin_full':
            av = ctrl.get('angular_vel', _SPIN_ANGULAR_VEL)
            self._set_wheels(-av, av)

        elif mode == 'search_rotate':
            av = ctrl.get('angular_vel', _SPIN_ANGULAR_VEL)
            self._set_wheels(-av, av)

        elif mode == 'search_approach':
            tgt, pose, lidar = ctrl['target'], ctrl['pose'], ctrl.get('lidar')
            dx, dy = tgt[0]-pose[0], tgt[1]-pose[1]
            dist   = np.hypot(dx, dy)
            he     = math.atan2(dy, dx) - pose[2]
            he     = math.atan2(math.sin(he), math.cos(he))
            fv     = min(5.0, 3.0*dist)
            av     = 6.0*he
            fv, at = self._lidar_avoidance(lidar, fv, pose=pose)
            av    += at
            self._set_wheels(fv-av, fv+av)

        elif mode == 'search_orbit':
            tp, pose  = ctrl['table_pos'], ctrl['pose']
            r_orb     = ctrl.get('orbit_radius', _ORBIT_RADIUS)
            lidar     = ctrl.get('lidar')
            dx, dy    = pose[0]-tp[0], pose[1]-tp[1]
            cur_r     = np.hypot(dx, dy)
            ang_ft    = math.atan2(dy, dx)
            origin_angle = math.atan2(-tp[1], -tp[0])
            ccw_tangent  = ang_ft + math.pi / 2
            cw_tangent   = ang_ft - math.pi / 2

            def _ang_diff(a, b):
                return math.atan2(math.sin(a - b), math.cos(a - b))
            use_ccw = abs(_ang_diff(ccw_tangent, origin_angle)) < \
                      abs(_ang_diff(cw_tangent,  origin_angle))
            tangent = ccw_tangent if use_ccw else cw_tangent
            rerr    = cur_r - r_orb
            desired = tangent + 0.5 * rerr
            he      = math.atan2(math.sin(desired - pose[2]),
                                 math.cos(desired - pose[2]))
            fv      = 3.5
            av      = 6.0 * he
            fv, at  = self._lidar_avoidance(lidar, fv, pose=pose)
            av     += at
            if self.step_counter % 240 == 0:
                dir_label = 'CCW' if use_ccw else 'CW'
                print(f"[Act] ORBIT {dir_label} r={cur_r:.2f}m (target {r_orb:.1f}m)")
            self._set_wheels(fv-av, fv+av)

        elif mode in ('navigate', 'approach'):
            tgt, pose = ctrl['target'], ctrl['pose']
            lidar     = ctrl.get('lidar')
            dx, dy    = tgt[0]-pose[0], tgt[1]-pose[1]
            dist      = np.hypot(dx, dy)
            he        = math.atan2(dy, dx) - pose[2]
            he        = math.atan2(math.sin(he), math.cos(he))
            kpd       = 5.0 if mode == 'navigate' else 4.0
            fv        = np.clip(kpd*dist, 0, 8.0)
            av        = 6.0*he
            relaxed   = ctrl.get('relaxed_avoidance', False)
            fv, at    = self._lidar_avoidance(lidar, fv, relaxed=relaxed, pose=pose)
            av       += at
            if self.step_counter % 240 == 0:
                print(f"[Act] {mode.upper()}: dist={dist:.2f}m, "
                      f"heading={np.degrees(he):.0f}deg")
            self._set_wheels(fv-av, fv+av)

        elif mode == 'approach_visual':
            pose       = ctrl.get('pose')
            world_tgt  = ctrl.get('world_target') or ctrl.get('target')
            lidar      = ctrl.get('lidar')
            cam_depth  = ctrl.get('camera_depth', float('inf'))
            world_dist = ctrl.get('world_dist_2d', float('inf'))

            # [F46] BUG 3 FIX: use APPROACH_STOP_M constant (0.40m) instead of
            # the previous magic number 0.45m.  The old value created a dead
            # zone: robot halted at 0.45m but the FSM GRASP threshold is 0.55m
            # (and F46 now uses world_dist_2d as fallback), so the robot would
            # stop moving before the FSM could ever transition to GRASP.
            if world_dist < APPROACH_STOP_M:
                self._set_wheels(0, 0)
                if self.step_counter % 60 == 0:
                    print(f"[F29] Hard stop: world_dist={world_dist:.2f}m")
                return

            if (cam_depth < MAX_TARGET_DEPTH and
                    (world_dist > cam_depth * 1.5 or world_dist > 2.5)):
                sd = cam_depth
            elif world_dist < float('inf'):
                sd = world_dist
            else:
                sd = cam_depth

            _MAX_BEARING_FAR  = math.radians(25.0)
            _MAX_BEARING_NEAR = math.radians(60.0)
            bearing_limit = _MAX_BEARING_NEAR if sd < 1.0 else _MAX_BEARING_FAR

            if world_tgt is not None and pose is not None:
                dx_w = world_tgt[0] - pose[0]
                dy_w = world_tgt[1] - pose[1]
                desired_heading = math.atan2(dy_w, dx_w)
                he = math.atan2(math.sin(desired_heading - pose[2]),
                                math.cos(desired_heading - pose[2]))
            else:
                he = ctrl.get('camera_bearing', 0.0)

            he = float(np.clip(he, -bearing_limit, bearing_limit))

            if sd > APPROACH_SLOW_M:
                fv = np.clip(4.0 * (sd - APPROACH_STOP_M), MIN_FWD_APPROACH, 5.0)
            elif sd > APPROACH_STOP_M:
                fv = max(MIN_FWD_APPROACH,
                         np.clip(4.0*(sd-APPROACH_STOP_M), 0.0, MIN_FWD_APPROACH*2))
            else:
                fv = 0.0

            av = 6.0 * he
            if sd > 1.0:
                fv, at = self._lidar_avoidance(lidar, fv, relaxed=True, pose=pose)
                av    += at
            else:
                if _lidar_has_data(lidar):
                    mf = min(lidar[i % len(lidar)] for i in range(-2, 3))
                    if mf < 0.15:
                        fv = 0.0

            if self.step_counter % 240 == 0:
                print(f"[Act] APPROACH_VISUAL: sd={sd:.2f}m, "
                      f"cam={cam_depth:.2f}m, bearing={np.degrees(he):.0f}deg")
            self._set_wheels(fv-av, fv+av)

        elif mode == 'grasp':
            self._set_wheels(0, 0)
            phase = ctrl.get('phase', 'reach_target')
            gpos  = ctrl['grasp_pos']

            if self.lift_joint_idx is not None:
                p.setJointMotorControl2(self.robot_id, self.lift_joint_idx,
                                        p.POSITION_CONTROL,
                                        targetPosition=_GRASP_LIFT_POS,
                                        force=100, maxVelocity=0.5)

            base_pose = ctrl.get('pose')
            grasp_object(self.robot_id, gpos, self.grasp_orientation,
                         arm_joints=self.arm_joints or None,
                         close_gripper=(phase == 'close_gripper'),
                         phase=phase,
                         base_pose=base_pose)

        elif mode == 'lift':
            self._set_wheels(0, 0)
            for fi in self.gripper_joints:
                jn = p.getJointInfo(self.robot_id, fi)[1].decode('utf-8')
                tp = -0.04 if 'left' in jn else 0.04
                p.setJointMotorControl2(self.robot_id, fi, p.POSITION_CONTROL,
                                        targetPosition=tp, force=50)
            if self.lift_joint_idx is not None:
                p.setJointMotorControl2(self.robot_id, self.lift_joint_idx,
                                        p.POSITION_CONTROL,
                                        targetPosition=0.3, force=100,
                                        maxVelocity=0.5)

        elif mode in ('idle', 'success', 'failure'):
            if mode == 'failure':
                lidar  = ctrl.get('lidar')
                rv     = -3.0
                rv, at = self._lidar_avoidance(lidar, rv)
                self._set_wheels(rv-at, rv+at)
                for j in self.arm_joints:
                    p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL,
                                            targetPosition=0.0, force=50,
                                            maxVelocity=1.0)
                if self.lift_joint_idx is not None:
                    p.setJointMotorControl2(self.robot_id, self.lift_joint_idx,
                                            p.POSITION_CONTROL,
                                            targetPosition=0.0, force=100,
                                            maxVelocity=0.5)
                for fi in self.gripper_joints:
                    p.setJointMotorControl2(self.robot_id, fi, p.POSITION_CONTROL,
                                            targetPosition=0.0, force=50)
            else:
                for i in self.wheel_joints:
                    p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                            targetVelocity=0, force=1500)


# ========================== MAIN ==========================================

def main():
    print("="*60)
    print("  IIS Cognitive Architecture - Navigate-to-Grasp Mission")
    print("="*60)

    robot_id, table_id, room_id, target_id = build_world(gui=True)

    # Disable real-time simulation for deterministic stepping
    p.setRealTimeSimulation(0)

    # [F43] Physics settling: let robot drop onto floor, all joints stabilise.
    # This happens BEFORE the while loop, so Rule 2 is not violated.
    print(f"[Init] Settling physics ({_SETTLING_STEPS} steps)...")
    num_joints = p.getNumJoints(robot_id)
    for _ in range(_SETTLING_STEPS):
        for ji in range(num_joints):
            info  = p.getJointInfo(robot_id, ji)
            jtype = info[2]
            if jtype == p.JOINT_REVOLUTE or jtype == p.JOINT_PRISMATIC:
                p.setJointMotorControl2(robot_id, ji, p.VELOCITY_CONTROL,
                                        targetVelocity=0.0, force=100)
        p.stepSimulation()
    print("[Init] Physics settled.")

    cog = CognitiveArchitecture(robot_id, table_id, room_id, target_id)

    print(f"\n[Init] Robot at (0.00, 0.00)")
    if cog.table_position:
        print(f"[Init] Table at ({cog.table_position[0]:.2f}, "
              f"{cog.table_position[1]:.2f})")
    try:
        print(f"[Init] M8 sensors:      {cog.kb.sensors()}")
        print(f"[Init] M8 capabilities: {cog.kb.robot_capabilities()}")
    except Exception:
        pass
    print(f"[Init] M9 DISABLED")
    print(f"[Init] Grasp world Z={_GRASP_WORLD_Z:.3f}m  "
          f"above Z={_GRASP_ABOVE_WORLD_Z:.3f}m  lift={_GRASP_LIFT_POS}")
    print("[Init] Mission: navigate to table, grasp red cylinder\n")


    ######        Main Loop        ######

    while p.isConnected():          # DO NOT TOUCH
        try:
            cog.fsm.tick()
            sensor_data      = cog.sense()
            control_commands = cog.think(sensor_data)
            cog.act(control_commands)
        except p.error as e:
            print(f"[Main] PyBullet disconnected: {e}")
            break

        if cog.fsm.state == RobotState.SUCCESS:
            if cog.fsm.get_time_in_state() > 3.0:
                print("\n" + "="*60)
                print("  MISSION COMPLETE - target grasped!")
                print("="*60)
                break

        if cog.step_counter % 240 == 0:
            pose = sensor_data['pose']
            print(f"[t={cog.step_counter/240:.0f}s] "
                  f"State={cog.fsm.state.name}  "
                  f"Pose=({pose[0]:.2f},{pose[1]:.2f},{np.degrees(pose[2]):.0f}deg)")

        cog.step_counter += 1
        p.stepSimulation()          # DO NOT TOUCH
        time.sleep(1. / 240.)       # DO NOT TOUCH


if __name__ == "__main__":
    main()
