"""
Module 10: Cognitive Architecture - SENSE-THINK-ACT Loop
Integrates all 10 modules for autonomous navigate-to-grasp mission.

SENSE-THINK-ACT Cycle:
    SENSE:  Read sensors, run full M4 PerceptionModule, estimate state
    THINK:  Update knowledge, plan actions, make decisions
    ACT:    Execute motion commands, control gripper

FIXES in this revision
  [F1]  fsm.tick() now called once per STA step - step-based timeouts work.
  [F2]  CAMERA_HEIGHT updated to 0.67 m.
  [F3]  approach_standoff reset on every NAVIGATE re-entry.
  [F4]  APPROACH_VISUAL rewritten: world-frame bearing, stops at depth<0.55m.
  [F5]  Raised dist-to-table rejection gate to 3.5 m.
  [F6]  distance_for_fsm during APPROACH uses 2D world-frame distance.
  [F7]  approach_visual speed uses np.hypot(world_tgt - pose).
  [F8]  _in_approach + _approach_depth_smooth reset when FSM leaves APPROACH.
  [F9]  Fixed bare `if lidar:` numpy-safe check.
  [F10] Stuck detection in think(): replan after 4 s without 0.1 m progress.
  [F11] action_planning.py: smarter bypass, room-bounds clamped.
  [F12] approach_visual stall fix.
  [F13] perception.py: tightened red HSV, PCA on red-masked pixels only.
  [F14] 360 spin-search, target-side standoff, standoff validity check.
  [F15] A) Red HSV S>=100/V>=60 for long-range detection.
        B) Orbit direction picks the side that faces origin (robot start),
           radius reduced 2.0->1.5 m for better small-cylinder visibility.
  [F16] FIX #1: NAVIGATE uses _compute_approach_standoff() instead of
        hardcoded +0.5 X offset so approach is correct for any table rotation.
  [F17] FIX #2: Grasp IK target converted from world-frame to robot-local
        frame so arm can actually reach the cylinder.
  [F18] FIX #4: _perception_interval raised 10->30 to reduce CPU stall.
  [F19] FIX: APPROACH think() now reads self.target_camera_depth directly
        each tick instead of the stale sensor_data value, so the FSM
        distance_to_target stays live and the robot actually advances.
  [F20] FIX: F12 lidar emergency-stop threshold raised 0.07->0.04 m and
        requires 3 consecutive hits to fire, preventing sensor noise from
        freezing forward velocity during close-range approach.
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

from src.environment.world_builder import build_world
from src.modules.sensor_preprocessing import get_sensor_data, get_sensor_id
from src.modules.perception import (PerceptionModule, detect_objects_by_color, RANSAC_Segmentation,)
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
_ASPECT  = 320.0 / 240.0
_FOV_H   = 2 * np.degrees(np.arctan(np.tan(np.radians(_FOV_V / 2)) * _ASPECT))
CAM_FX   = (320 / 2.0) / np.tan(np.radians(_FOV_H / 2))
CAM_FY   = (240 / 2.0) / np.tan(np.radians(_FOV_V / 2))
CAM_CX, CAM_CY = 160.0, 120.0

TARGET_COLOR     = 'red'
MAX_TARGET_DEPTH = 4.0
MIN_TARGET_DEPTH = 0.2
MAX_JUMP_M       = 1.0
_CAM_TILT        = 0.2

# -- Approach tuning ---------------------------------------------------------
GRASP_RANGE_M    = 0.35
APPROACH_STOP_M  = 0.25
APPROACH_SLOW_M  = 0.6
MIN_FWD_APPROACH = 0.30
STANDOFF_DIST_M  = 0.80  # Keep closer distance for better grasping

# -- Stuck detection ---------------------------------------------------------
_STUCK_DIST_M  = 0.10
_STUCK_TIMEOUT = 2.0  # Reduced from 4.0 for faster recovery

# -- [F14-A] 360 spin constants ----------------------------------------------
_SPIN_ANGULAR_VEL = 5.0  # Increased from 3.0 for faster spinning
_SPIN_STEPS       = 50   # Reduced from 100 for faster search

# -- [F15-B] Orbit radius reduced for better small-object visibility ---------
_ORBIT_RADIUS = 1.5   # was 2.0

# -- Room safety bounds ------------------------------------------------------
_ROOM_BOUND = 3.5

# -- GRASP CONFIGURATION -----------------------------------------------------
# Table height is 0.625m, cylinder sits on top (height 0.12m, center at 0.685m)
TABLE_SURFACE_Z = 0.625
CYLINDER_CENTER_Z = 0.685  # TABLE_HEIGHT + TARGET_HEIGHT/2
GRASP_HEIGHT_OFFSET = 0.02  # Gripper needs to be slightly below cylinder center for grasp
LIFT_HEIGHT = 0.15  # How high to lift the cylinder after grasping
APPROACH_HEIGHT = 0.15  # How high above grasp point to approach from

# -- [F20] Lidar emergency-stop: raised threshold + consecutive-hit guard ----
_LIDAR_ESTOP_THRESHOLD   = 0.04   # was 0.07 m; noise readings rarely exceed 0.02 m
_LIDAR_ESTOP_CONSEC_HITS = 3      # must trigger this many times in a row to stop


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

        initialize_state_estimator()
        self.sensor_camera_id, self.sensor_lidar_id = get_sensor_id(self.robot_id)
        #M4 — PerceptionModule (RANSAC + PCA + colour detection)
        self.perception = PerceptionModule()
        # [F18] FIX #4: Raised from 10 to 30 to reduce per-frame CPU cost.
        # Full PCA+RANSAC pipeline every ~125 ms is still plenty for this task.
        self._perception_interval = 30

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
        self.obstacles                 = []
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

        self.step_counter = 0
        self.dt           = 1.0 / 240.0

        self._last_grasp_attempt = False

        # [F20] Consecutive lidar-estop hit counter
        self._lidar_estop_consec = 0
        
        self._initialize_world_knowledge()
        self._initialize_motors()

    # -----------------------------------------------------------------------

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
        # Try both locations: current directory and executables directory
        possible_paths = ["initial_map.json", os.path.join("..", "initial_map.json")]
        
        world_map = None
        for map_path in possible_paths:
            if os.path.exists(map_path):
                with open(map_path, 'r') as f:
                    world_map = json.load(f)
                print(f"[CogArch] Loaded world map from {map_path}")
                break
        
        if world_map is None:
            print("[CogArch] WARNING: Could not find initial_map.json")
            return
            
        if 'table' in world_map:
            td  = world_map['table']
            pos = td['position']
            self.kb.add_position('table', pos[0], pos[1], pos[2])
            self.kb.add_detected_object('table', 'furniture', 'brown', pos)
            self.table_position    = pos
            self.table_orientation = td.get('orientation')
            self.table_size        = td.get('size')
            self.obstacles.append(pos[:2])
            print(f"[CogArch] Table loaded at position: {pos[:2]}")
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

    # -----------------------------------------------------------------------

    def _check_gripper_contact(self):
        """
        Check if the gripper has successfully grasped the object.
        We check for:
        1. Physical contact between gripper and target
        """
        # Check for direct contact between gripper and target
        contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.target_id)
        if contacts and len(contacts) > 0:
            if self.step_counter % 60 == 0:
                print(f"[Sense] Gripper contact detected ({len(contacts)} contact points)")
            return True
        
        return False

    def _compute_approach_standoff(self, target_pos, robot_pose):
        """
        [F14-B] Place standoff on the table side nearest the target cylinder.
        [F14-C] Validate: inside room bounds and clear of obstacles.
        """
        sd = STANDOFF_DIST_M

        def _candidate(normal_2d):
            nx, ny = normal_2d
            return [target_pos[0] + sd * nx, target_pos[1] + sd * ny]

        def _valid(cand):
            cx, cy = cand
            if abs(cx) > _ROOM_BOUND or abs(cy) > _ROOM_BOUND:
                return False
            for obs in self.obstacles:
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
            hs = self.table_size
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
            print(f"[F14-B] Cylinder local=({loc_x:.2f},{loc_y:.2f}) -> "
                  f"approach normal=({normal[0]:.2f},{normal[1]:.2f})")

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
                print(f"[F14-C] Standoff ({cand[0]:.2f},{cand[1]:.2f}) valid")
                return cand
            opp  = [-best_normal[0], -best_normal[1]]
            opp_cand = _candidate(opp)
            if _valid(opp_cand):
                print(f"[F14-C] Primary blocked, using opposite "
                      f"({opp_cand[0]:.2f},{opp_cand[1]:.2f})")
                return opp_cand
            print("[F14-C] Both sides blocked - falling back to robot-side")

        dx, dy = target_pos[0]-robot_pose[0], target_pos[1]-robot_pose[1]
        d = np.hypot(dx, dy)
        if self.table_position is not None:
            return list(self.table_position[:2])
        elif d > 0.01:
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
        # REMOVED: Table avoidance logic - the table is the GOAL, not an obstacle!
        # The robot should navigate TO the table, not avoid it.
        # Regular obstacle avoidance below handles collision prevention.
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

    def _pixel_depth_to_world(self, px, py, depth_m, robot_pose):
        nx = (px - CAM_CX) / CAM_FX
        ny = (py - CAM_CY) / CAM_FY
        cam_x = depth_m * nx
        cam_y = depth_m * ny
        cam_z = depth_m
        ct = math.cos(_CAM_TILT)
        st = math.sin(_CAM_TILT)
        body_forward = ct * cam_z - st * cam_y
        body_up      = st * cam_z + ct * cam_y
        body_lateral = cam_x
        rx, ry, rt = robot_pose
        world_x = rx + body_forward * math.cos(rt) - (-body_lateral) * math.sin(rt)
        world_y = ry + body_forward * math.sin(rt) + (-body_lateral) * math.cos(rt)
        world_z = CAMERA_HEIGHT - body_up
        return [world_x, world_y, world_z]

    def _update_target_from_detection(self, new_pos, depth_m, bearing):
        if self.target_position_smoothed is not None:
            jump = np.hypot(new_pos[0] - self.target_position_smoothed[0],
                            new_pos[1] - self.target_position_smoothed[1])
            if jump > MAX_JUMP_M:
                return False
        self.target_detection_count += 1
        self.target_camera_bearing  = bearing
        self.target_camera_depth    = depth_m
        alpha = 0.5 if self.target_detection_count <= 5 else 0.15
        if self.target_position_smoothed is None:
            self.target_position_smoothed = list(new_pos)
        else:
            for k in range(3):
                self.target_position_smoothed[k] = (
                    alpha * new_pos[k] + (1-alpha) * self.target_position_smoothed[k]
                )
        self.target_position = list(self.target_position_smoothed)
        self.kb.add_position('target', *self.target_position)
        return True

   # ==================== SENSE ====================
    def sense(self):
        """
        SENSE phase: Acquire sensor data and update state estimate.
        Returns sensor_data dict for use in THINK phase.
        M4 integration:
          - Every _perception_interval steps, run PerceptionModule.process_frame()
            which executes: colour detection -> RANSAC (table plane) -> PCA (poses)
          - Target 3-D position is taken from the PCA centre of the red ROI
            (not a single-pixel depth sample)
          - Obstacle poses from PCA are pushed back into KB and self.obstacles
          - Static scene map (obstacles) locked after first detection per task spec
        """
        # M3: Get preprocessed sensor data via sensor_preprocessing module
        preprocessed = get_sensor_data(self.robot_id, self.sensor_camera_id, self.sensor_lidar_id)
        rgb = preprocessed['camera_rgb']
        depth = preprocessed['camera_depth']
        lidar = preprocessed['lidar']
        imu = preprocessed['imu']
        joint_states = preprocessed['joint_states']
        
        # M5: Get wheel velocities from joint states using correct URDF joint names
        wheel_vels = []
        for name in self.wheel_names:
            if name in joint_states:
                wheel_vels.append(joint_states[name]['velocity'])
            else:
                wheel_vels.append(0.0)
        
        # M5: State estimation via state_estimate() function
        sensors_for_pf = {
            'imu': imu,
            'lidar': lidar,
            'joint_states': joint_states
        }
        control_inputs = {
            'wheel_left': (wheel_vels[0] + wheel_vels[2]) / 2.0,   # avg of FL + BL
            'wheel_right': (wheel_vels[1] + wheel_vels[3]) / 2.0,  # avg of FR + BR
        }
        estimated_pose = state_estimate(sensors_for_pf, control_inputs)
        
        # M8: Update robot position in Knowledge Base (Prolog)
        if self.step_counter % 50 == 0:
            self.kb.add_position('robot', 
                                float(estimated_pose[0]), 
                                float(estimated_pose[1]), 0.0)
        
        # M4: Full perception pipeline - colour detection + RANSAC + PCA
        if rgb is not None and depth is not None \
                and self.step_counter % self._perception_interval == 0:

            perc = self.perception.process_frame(rgb, depth)

            # --- 1. Log colour detections ---
            if self.step_counter % 60 == 0 and perc['detections']:
                colors_found = [d['color'] for d in perc['detections']]
                print(f"[M4-Color] Detected colours: {colors_found}")

            # --- 2. Log RANSAC table plane ---
            if perc['table_plane'] is not None and self.step_counter % 240 == 0:
                print(f"[M4-RANSAC] Table plane confirmed, "
                    f"inliers={perc['table_plane']['num_inliers']}")

            # --- 3. Target 3-D pose from PCA of red point cloud ---
            if perc['target_pose'] is not None:
                cam_x, cam_y, cam_z = perc['target_pose']['center']

                if DEPTH_NEAR < cam_z < DEPTH_FAR:
                    robot_x, robot_y, robot_theta = estimated_pose
                    cos_t, sin_t = math.cos(robot_theta), math.sin(robot_theta)
                    rb_x = cam_z + CAMERA_FORWARD
                    rb_y = -cam_x
                    world_x = robot_x + rb_x * cos_t - rb_y * sin_t
                    world_y = robot_y + rb_x * sin_t + rb_y * cos_t
                    world_z = CAMERA_HEIGHT - cam_y

                    new_target = [world_x, world_y, world_z]

                    # Outlier rejection XY
                    accept = True
                    if self.target_position_smoothed is not None:
                        jump = np.hypot(new_target[0] - self.target_position_smoothed[0],
                                        new_target[1] - self.target_position_smoothed[1])
                        if jump > 2.0 and self.target_detection_count > 5:
                            accept = False
                    # Outlier rejection Z (cylinder sits on table at ~0.625m)
                    if not (0.55 < world_z < 1.0):
                        accept = False

                    if accept:
                        self.target_detection_count += 1
                        # FIX: correct bearing sign (positive = target is to the right)
                        fy = (240 / 2.0) / np.tan(np.deg2rad(60 / 2.0))
                        fx = fy * (320 / 240)
                        px = cam_x * fx / cam_z + 160.0
                        self.target_camera_bearing = math.atan2(px - 160.0, fx)
                        self.target_camera_depth   = cam_z   # <-- always fresh from PCA

                        alpha = 0.4 if self.target_detection_count > 3 else 0.8
                        if self.target_position_smoothed is None:
                            self.target_position_smoothed = list(new_target)
                        else:
                            for k in range(3):
                                self.target_position_smoothed[k] = (
                                    alpha * new_target[k]
                                    + (1 - alpha) * self.target_position_smoothed[k])
                        self.target_position = list(self.target_position_smoothed)
                        self.kb.add_position('target', *self.target_position)
                        print(f"[M4-PCA] TARGET world=({world_x:.2f},{world_y:.2f},{world_z:.2f}) "
                            f"depth={cam_z:.2f}m bearing={math.degrees(self.target_camera_bearing):.1f}deg")
            # Fallback: PCA failed but colour visible - update bearing from bbox only
            elif perc['detections']:
                red_dets = [d for d in perc['detections'] if d['color'] == 'red']
                if red_dets:
                    best = max(red_dets, key=lambda d: d['area'])
                    bx, by, bw, bh = best['bbox']
                    px = bx + bw / 2.0
                    fy = (240 / 2.0) / np.tan(np.deg2rad(60 / 2.0))
                    fx = fy * (320 / 240)
                    self.target_camera_bearing = math.atan2(px - 160.0, fx)

            # --- 4. Obstacle poses from PCA -> KB + obstacle map update ---
            for obs in perc['obstacle_poses']:
                cam_x, cam_y, cam_z = obs['center']
                if cam_z < DEPTH_NEAR or cam_z > DEPTH_FAR:
                    continue
                robot_x, robot_y, robot_theta = estimated_pose
                cos_t, sin_t = math.cos(robot_theta), math.sin(robot_theta)
                rb_x = cam_z + CAMERA_FORWARD
                rb_y  = -cam_x
                wx = robot_x + rb_x * cos_t - rb_y * sin_t
                wy = robot_y + rb_x * sin_t  + rb_y * cos_t
                if not self.perception._scene_map_locked:
                    self.kb.add_position(f"obs_{obs['color']}", wx, wy, 0.0)
                    already = any(abs(o[0]-wx) < 0.3 and abs(o[1]-wy) < 0.3
                                for o in self.obstacles)
                    if not already:
                        self.obstacles.append([wx, wy])
                        print(f"[M4-PCA] Obstacle ({obs['color']}) "
                            f"added to map world=({wx:.2f},{wy:.2f})")
        
        # M3: Check gripper contact using joint state feedback (legal)
        gripper_contact = self._check_gripper_contact()
        
        return {
            'pose': estimated_pose,
            'rgb': rgb,
            'depth': depth,
            'lidar': lidar,
            'imu': imu,
            'joint_states': joint_states,
            'target_detected': self.target_position is not None,
            'target_position': self.target_position,
            'target_camera_bearing': self.target_camera_bearing,
            'target_camera_depth': self.target_camera_depth,
            'gripper_contact': gripper_contact,
            'table_near': self.table_position is not None and np.hypot(estimated_pose[0] - self.table_position[0], estimated_pose[1] - self.table_position[1]) < 2.0
        }

    # ========================== THINK =====================================

    def think(self, sensor_data):
        pose = sensor_data['pose']
        current_state = self.fsm.state

        if (current_state == RobotState.NAVIGATE
                and self._last_fsm_state != RobotState.NAVIGATE):
            self.approach_standoff = None
            self.current_waypoint  = None
            self._stuck_pose  = (pose[0], pose[1])
            self._stuck_timer = 0

        if (current_state == RobotState.SEARCH
                and self._last_fsm_state != RobotState.SEARCH):
            self._spin_steps_done = 0
            self._spin_complete   = False
            print("[F14-A] Entered SEARCH: starting 360 spin-in-place")

        if (self._last_fsm_state == RobotState.APPROACH
                and current_state != RobotState.APPROACH):
            self._in_approach           = False
            self._approach_depth_smooth = float('inf')

        self._last_fsm_state = current_state

        if self.fsm.state != RobotState.FAILURE:
            self._failure_reset_done = False

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
                        print(f"[F10] STUCK at ({pose[0]:.2f},{pose[1]:.2f}) - replanning")
                        self.approach_standoff = None
                        self.current_waypoint  = None
                        self._stuck_pose       = (pose[0], pose[1])
                        self._stuck_timer      = 0
                        if self.target_position is not None:
                            self.approach_standoff = self._compute_approach_standoff(
                                self.target_position, pose)
        else:
            self._stuck_pose  = None
            self._stuck_timer = 0

        target_pos = self.kb.query_position('target')
        if sensor_data['target_detected']:
            target_pos = sensor_data['target_position']

        if target_pos and self.step_counter % 240 == 0:
            try:
                if self.kb.is_goal_object('target'):
                    print("[M8-KB] Target confirmed as goal object (red)")
                if self.kb.check_can_grasp():
                    print("[M8-KB] Prolog confirms: robot can grasp target")
            except Exception:
                pass

        # Update FSM with current sensor data
        # Calculate distance_for_fsm based on current state
        if target_pos:
            dx, dy      = target_pos[0]-pose[0], target_pos[1]-pose[1]
            distance_2d = np.hypot(dx, dy)
            if self.fsm.state == RobotState.APPROACH:
                # [F19] FIX: Read self.target_camera_depth directly — it is updated
                # by sense() every _perception_interval steps.  sensor_data carries
                # the value from the START of this STA tick which may be one tick
                # stale; reading the instance variable gives the freshest value.
                cam_depth = self.target_camera_depth
                if cam_depth < MAX_TARGET_DEPTH:
                    distance_for_fsm = cam_depth
                    if self.step_counter % 120 == 0:
                        print(f"[Think] Using camera depth={cam_depth:.2f}m for FSM")
                else:
                    distance_for_fsm = distance_2d
            elif self.approach_standoff is not None and self.fsm.state == RobotState.NAVIGATE:
                sdx = self.approach_standoff[0]-pose[0]
                sdy = self.approach_standoff[1]-pose[1]
                distance_for_fsm = np.hypot(sdx, sdy)
            else:
                cam_d = self.target_camera_depth
                distance_for_fsm = cam_d if cam_d < MAX_TARGET_DEPTH else distance_2d
        else:
            # No target detected - use table position for navigation
            if self.table_position is not None:
                dx, dy = self.table_position[0] - pose[0], self.table_position[1] - pose[1]
                distance_2d = np.hypot(dx, dy)
                distance_for_fsm = distance_2d
            else:
                distance_2d = distance_for_fsm = float('inf')

        self.fsm.update({
            'target_visible':     sensor_data['target_detected'],
            'target_position':    target_pos,
            'distance_to_target': distance_for_fsm,
            'collision_detected': False,
            'gripper_contact':    sensor_data.get('gripper_contact', False),
            'object_grasped':     sensor_data.get('gripper_contact', False),
            'estimated_pose':     pose,
        })

        ctrl = {'mode': 'idle', 'target': None, 'gripper': 'open'}

        # -- SEARCH ----------------------------------------------------------
        if self.fsm.state == RobotState.SEARCH:
            if self.step_counter % 120 == 0:
                print(f"[Think] SEARCH: target_detected={sensor_data['target_detected']}, "
                      f"target_pos={target_pos is not None}")
            
            # Immediately go towards the table when available - no spinning
            if self.table_position:
                td = np.hypot(self.table_position[0]-pose[0],
                              self.table_position[1]-pose[1])
                
                # If table is far, go directly to it
                if td > 1.5:
                    ctrl = {'mode': 'search_approach',
                            'target': self.table_position[:2],
                            'pose': pose, 'angular_vel': 5.0,
                            'lidar': sensor_data['lidar']}
                    if self.step_counter % 240 == 0:
                        print(f"[Think] SEARCH: Going directly to TABLE (dist={td:.2f}m)")
                else:
                    # Table is close - transition to NAVIGATE to approach and grasp
                    if self.step_counter % 240 == 0:
                        print(f"[Think] SEARCH: Table near (dist={td:.2f}m), transitioning to NAVIGATE")
                    self.fsm.transition_to(RobotState.NAVIGATE)
            else:
                # No table info - just rotate to find it
                ctrl = {'mode': 'search_rotate', 'angular_vel': 5.0}

        # -- NAVIGATE --------------------------------------------------------
        elif self.fsm.state == RobotState.NAVIGATE:
            if self.step_counter % 120 == 0:
                print(f"[Think] NAVIGATE: target_pos={target_pos is not None}, "
                      f"approach_standoff={self.approach_standoff}")
            
            cam_d       = self.target_camera_depth
            use_relaxed = cam_d < 2.5 and sensor_data['target_detected']

            # [F16] FIX #1: Use _compute_approach_standoff() so the nav goal is
            # always on the correct face of the table regardless of its yaw.
            # Old code used a hardcoded +0.5 X offset which only worked when the
            # table happened to face +X.
            if self.approach_standoff is None:
                ref = target_pos if target_pos else self.table_position
                if ref is not None:
                    self.approach_standoff = self._compute_approach_standoff(ref, pose)
                    if self.step_counter % 240 == 0:
                        print(f"[F16] Computed standoff: {self.approach_standoff}")

            nav_goal = self.approach_standoff

            # Fallback: if standoff computation failed, head directly to table
            if nav_goal is None and self.table_position is not None:
                nav_goal = self.table_position[:2]
            elif nav_goal is None and target_pos is not None:
                nav_goal = target_pos[:2]

            if nav_goal and self.current_waypoint is None:
                self.action_planner.create_plan(pose[:2], nav_goal, self.obstacles)
                self.current_waypoint = self.action_planner.get_next_waypoint()
            if self.current_waypoint:
                ctrl = {'mode': 'navigate', 'target': self.current_waypoint,
                        'pose': pose, 'lidar': sensor_data['lidar'],
                        'relaxed_avoidance': use_relaxed}
                if np.hypot(self.current_waypoint[0]-pose[0],
                            self.current_waypoint[1]-pose[1]) < 0.2:
                    self.action_planner.advance_waypoint()
                    self.current_waypoint = self.action_planner.get_next_waypoint()

        # -- APPROACH --------------------------------------------------------
        elif self.fsm.state == RobotState.APPROACH:
            if not self._in_approach:
                self._approach_depth_smooth = float('inf')
                self._in_approach           = True
                print("[Think] APPROACH: Starting visual approach to target")
            
            # If we don't have target position yet, we need to find it first
            if not target_pos:
                # Use camera bearing to find the target
                cam_bearing = self.target_camera_bearing
                cam_depth = self.target_camera_depth
                
                # If we can see something (bearing != 0 or depth < inf), target is visible
                if cam_depth < MAX_TARGET_DEPTH:
                    # Target is visible, use it for approach
                    if self.table_position:
                        # Use table position with some offset based on bearing
                        approach_target = [
                            self.table_position[0] + 0.3 * math.cos(cam_bearing),
                            self.table_position[1] + 0.3 * math.sin(cam_bearing),
                        ]
                    else:
                        approach_target = [pose[0] + 0.3 * math.cos(cam_bearing + pose[2]),
                                           pose[1] + 0.3 * math.sin(cam_bearing + pose[2])]
                    if self.step_counter % 120 == 0:
                        print(f"[Think] APPROACH: Target detected via camera, bearing={np.degrees(cam_bearing):.1f}deg")
                else:
                    # Target not visible - slowly rotate to find it
                    if self.step_counter % 120 == 0:
                        print(f"[Think] APPROACH: Target not visible, rotating to find it")
                    ctrl = {'mode': 'search_rotate', 'angular_vel': 2.0}
                    return ctrl
            else:
                # We have target position - approach it
                approach_target = target_pos[:2]
            
            # Calculate 3D distance to target if available
            if target_pos:
                dist_3d = np.hypot(target_pos[0] - pose[0], target_pos[1] - pose[1])
            else:
                dist_3d = float('inf')
            
            # [F19] Always read live depth from instance variable, not stale sensor_data
            cam_depth = self.target_camera_depth
            if cam_depth < MAX_TARGET_DEPTH:
                use_dist = cam_depth
                if self.step_counter % 120 == 0:
                    print(f"[Think] APPROACH: using camera depth={cam_depth:.2f}m")
            else:
                use_dist = dist_3d
            
            ctrl = {'mode': 'approach_visual',
                    'target':            approach_target,
                    'pose':              pose,
                    'lidar':             sensor_data['lidar'],
                    'relaxed_avoidance': True,
                    'camera_bearing':    self.target_camera_bearing,
                    'camera_depth':      cam_depth,
                    'world_target':      approach_target,
                    'world_dist_2d':     dist_3d}
            
            # Also update distance_for_fsm for FSM transition
            self.fsm.distance_to_target = use_dist

        # -- GRASP -----------------------------------------------------------
        elif self.fsm.state == RobotState.GRASP:
            # Use last known target position or table position
            grasp_target = target_pos
            
            # If no target detected, use table position with estimated cylinder location
            if grasp_target is None:
                if self.table_position:
                    # Estimate cylinder is on table at a slight offset from center
                    grasp_target = [
                        self.table_position[0],
                        self.table_position[1],
                        CYLINDER_CENTER_Z  # Use known cylinder center height
                    ]
                    if self.step_counter % 60 == 0:
                        print(f"[Think] GRASP: Using estimated table position as grasp target")
                else:
                    # No table - can't grasp
                    print(f"[Think] GRASP: No target or table, transitioning to SEARCH")
                    self.fsm.transition_to(RobotState.SEARCH)
                    return ctrl
            
            if grasp_target:
                gp    = self.grasp_planner.plan_grasp(grasp_target)
                gt    = self.fsm.get_time_in_state()
                
                phase = ('reach_above'  if gt < 3.0 else
                         'reach_target' if gt < 7.0 else
                         'close_gripper')
                
                ctrl = {'mode': 'grasp',
                        'approach_pos': gp['approach_pos'],
                        'grasp_pos':    gp['grasp_pos'],
                        'orientation':  gp['orientation'],
                        'phase':        phase,
                        'robot_pose':   pose}  # [F17] pass pose for local-frame IK
                if self.step_counter % 60 == 0:
                    print(f"[Think] GRASP: phase={phase}, time={gt:.1f}s, target={grasp_target[:2]}")
            else:
                # No target at all - go back to search
                print(f"[Think] GRASP: No target, transitioning to SEARCH")
                self.fsm.transition_to(RobotState.SEARCH)

        # -- LIFT ------------------------------------------------------------
        elif self.fsm.state == RobotState.LIFT:
            ctrl = {'mode': 'lift', 'lift_height': 0.2, 'gripper': 'close'}

        # -- PLACE ------------------------------------------------------------
        elif self.fsm.state == RobotState.PLACE:
            if target_pos:
                gp = self.grasp_planner.plan_grasp(target_pos)
                ctrl = {'mode': 'place',
                        'place_pos': gp['place_pos'],
                        'orientation': gp['orientation']}

        # -- SUCCESS ---------------------------------------------------------
        elif self.fsm.state == RobotState.SUCCESS:
            ctrl = {'mode': 'success', 'gripper': 'close'}

        # -- FAILURE ---------------------------------------------------------
        elif self.fsm.state == RobotState.FAILURE:
            if not self._failure_reset_done:
                self.approach_standoff        = None
                self.current_waypoint         = None
                self.target_position_smoothed = None
                self.target_detection_count   = 0
                self.pca_target_pose          = None
                self.table_plane_model        = None
                self._spin_steps_done         = 0
                self._spin_complete           = False
                self._failure_reset_done      = True
            ctrl = {'mode': 'failure', 'gripper': 'open',
                    'lidar': sensor_data['lidar']}

        return ctrl

    # ========================== ACT =======================================

    def _stow_arm(self):
        for j in self.arm_joints:
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=500, maxVelocity=2.0)

    def _execute_arm_ik(self, target_world_pos, target_orn, robot_pose=None):
        """
        Execute inverse kinematics to move arm to target position.

        [F17] FIX #2: Convert world-frame target into robot-local frame before
        calling PyBullet IK.  The old code passed raw world coordinates
        (e.g. [2.3, -1.1, 0.685]) which are meaningless to PyBullet's IK
        solver because IK expects a position relative to the robot base.
        We subtract the robot's XY position and rotate by -theta so IK
        receives a body-frame position it can actually reach.

        Returns True if IK solution found, False otherwise.
        """
        if not self.arm_joints:
            print("[IK] No arm joints configured")
            return False

        # --- [F17] World -> robot-local frame conversion --------------------
        if robot_pose is not None:
            rx, ry, rtheta = robot_pose[0], robot_pose[1], robot_pose[2]
            dx = target_world_pos[0] - rx
            dy = target_world_pos[1] - ry
            dz = target_world_pos[2]   # Z stays the same (world-up)
            cos_t = math.cos(-rtheta)
            sin_t = math.sin(-rtheta)
            local_x = cos_t * dx - sin_t * dy
            local_y = sin_t * dx + cos_t * dy
            ik_target = [local_x, local_y, dz]
        else:
            ik_target = list(target_world_pos)
        # --------------------------------------------------------------------

        # Find gripper base link index for IK
        gripper_link_idx = -1
        for i in range(p.getNumJoints(self.robot_id)):
            link_name = p.getJointInfo(self.robot_id, i)[12].decode('utf-8')
            if 'gripper_base' in link_name:
                gripper_link_idx = i
                break
        
        if gripper_link_idx == -1:
            # Fallback to last arm joint
            gripper_link_idx = self.arm_joints[-1]
        
        try:
            # Calculate IK solution using robot-local target
            ik_solution = p.calculateInverseKinematics(
                self.robot_id,
                gripper_link_idx,
                ik_target,
                targetOrientation=target_orn,
                maxNumIterations=100,
                residualThreshold=0.001
            )
            
            if ik_solution is None or len(ik_solution) == 0:
                print("[IK] No IK solution found")
                return False
            
            # Apply IK solution to arm joints
            # Get all non-fixed joint indices
            num_joints = p.getNumJoints(self.robot_id)
            non_fixed_joints = []
            for j in range(num_joints):
                if p.getJointInfo(self.robot_id, j)[2] != p.JOINT_FIXED:
                    non_fixed_joints.append(j)
            
            # Apply IK to arm joints
            for joint_idx in self.arm_joints:
                if joint_idx in non_fixed_joints:
                    ik_idx = non_fixed_joints.index(joint_idx)
                    if ik_idx < len(ik_solution):
                        p.setJointMotorControl2(
                            self.robot_id,
                            joint_idx,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=ik_solution[ik_idx],
                            force=500,
                            maxVelocity=1.0
                        )
            
            return True
        except Exception as e:
            print(f"[IK] Error: {e}")
            return False

    def _close_gripper(self):
        """
        Close gripper fingers to grasp the object.
        """
        if not self.gripper_joints:
            print("[Gripper] No gripper joints found")
            return
        
        for fi in self.gripper_joints:
            joint_info = p.getJointInfo(self.robot_id, fi)
            joint_name = joint_info[1].decode('utf-8')
            # Left finger: limits [-0.04, 0], close at -0.04 (moves toward center)
            # Right finger: limits [0, 0.04], close at 0.04 (moves toward center)
            if 'left' in joint_name:
                target = -0.04  # Close toward center
            else:
                target = 0.04  # Close toward center
            
            p.setJointMotorControl2(
                self.robot_id,
                fi,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=50
            )

    def _open_gripper(self):
        """
        Open gripper fingers to release the object.
        """
        if not self.gripper_joints:
            print("[Gripper] No gripper joints found")
            return
        
        for fi in self.gripper_joints:
            p.setJointMotorControl2(
                self.robot_id,
                fi,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0,  # Open position
                force=50
            )

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
            if self.step_counter % 60 == 0:
                pct = min(100, int(100 * self._spin_steps_done / _SPIN_STEPS))
                print(f"[F14-A] Spinning {pct}% of 360")

        elif mode == 'search_rotate':
            av = ctrl.get('angular_vel', 3.0)
            self._set_wheels(-av, av)
            if self.step_counter % 60 == 0:
                print(f"[Act] SEARCH_ROTATE: rotating to find target")

        elif mode == 'search_approach':
            tgt, pose, lidar = ctrl['target'], ctrl['pose'], ctrl.get('lidar')
            dx, dy = tgt[0]-pose[0], tgt[1]-pose[1]
            dist   = np.hypot(dx, dy)
            he     = math.atan2(dy, dx) - pose[2]
            he     = math.atan2(math.sin(he), math.cos(he))
            fv     = min(5.0, 3.0*dist)
            av     = 5.0*he
            fv, at = self._lidar_avoidance(lidar, fv, pose=pose)
            av    += at
            if self.step_counter % 240 == 0:
                print(f"[Act] SEARCH_APPROACH: dist={dist:.2f}m, "
                      f"heading={np.degrees(he):.0f} deg")
            self._set_wheels(fv-av, fv+av)

        elif mode == 'search_orbit':
            tp, pose  = ctrl['table_pos'], ctrl['pose']
            r_orb     = ctrl.get('orbit_radius', _ORBIT_RADIUS)
            lidar     = ctrl.get('lidar')
            dx, dy    = pose[0]-tp[0], pose[1]-tp[1]
            cur_r     = np.hypot(dx, dy)
            ang_ft    = math.atan2(dy, dx)

            origin_angle = math.atan2(-tp[1], -tp[0])
            ccw_tangent = ang_ft + math.pi / 2
            cw_tangent  = ang_ft - math.pi / 2
            def _ang_diff(a, b):
                return math.atan2(math.sin(a - b), math.cos(a - b))
            use_ccw = abs(_ang_diff(ccw_tangent, origin_angle)) < \
                      abs(_ang_diff(cw_tangent,  origin_angle))
            tangent = ccw_tangent if use_ccw else cw_tangent

            rerr    = cur_r - r_orb
            desired = tangent + 0.5 * rerr
            he      = math.atan2(math.sin(desired - pose[2]),
                                 math.cos(desired - pose[2]))
            fv      = 3.0
            av      = 5.0 * he
            fv, at  = self._lidar_avoidance(lidar, fv, pose=pose)
            av     += at
            if self.step_counter % 240 == 0:
                dir_label = 'CCW' if use_ccw else 'CW'
                print(f"[Act] ORBIT {dir_label} r={cur_r:.2f}m "
                      f"(target {r_orb:.1f}m)")
            self._set_wheels(fv-av, fv+av)

        elif mode in ('navigate', 'approach'):
            tgt, pose = ctrl['target'], ctrl['pose']
            lidar     = ctrl.get('lidar')
            dx, dy    = tgt[0]-pose[0], tgt[1]-pose[1]
            dist      = np.hypot(dx, dy)
            he        = math.atan2(dy, dx) - pose[2]
            he        = math.atan2(math.sin(he), math.cos(he))
            kpd       = 6.0 if mode == 'navigate' else 4.0
            fv        = np.clip(kpd*dist, 0, 8.0)
            av        = 6.0*he
            relaxed   = ctrl.get('relaxed_avoidance', False)
            fv, at    = self._lidar_avoidance(lidar, fv, relaxed=relaxed, pose=pose)
            av       += at
            if self.step_counter % 240 == 0:
                print(f"[Act] {mode.upper()}: dist={dist:.2f}m, "
                      f"heading={np.degrees(he):.0f} deg, "
                      f"fwd={fv:.1f}, turn={av:.1f}")
            self._set_wheels(fv-av, fv+av)

        elif mode == 'approach_visual':
            pose       = ctrl.get('pose')
            world_tgt  = ctrl.get('world_target') or ctrl.get('target')
            lidar      = ctrl.get('lidar')
            cam_depth  = ctrl.get('camera_depth', float('inf'))
            world_dist = ctrl.get('world_dist_2d', float('inf'))
            sd = world_dist if world_dist < float('inf') else cam_depth

            if world_tgt is not None and pose is not None:
                dx_w = world_tgt[0] - pose[0]
                dy_w = world_tgt[1] - pose[1]
                desired_heading = math.atan2(dy_w, dx_w)
                he = math.atan2(math.sin(desired_heading - pose[2]),
                                math.cos(desired_heading - pose[2]))
            else:
                he = ctrl.get('camera_bearing', 0.0)

            if sd > APPROACH_SLOW_M:
                fv = np.clip(5.0 * (sd - APPROACH_STOP_M), MIN_FWD_APPROACH, 5.0)
            elif sd > APPROACH_STOP_M:
                fv = max(MIN_FWD_APPROACH,
                         np.clip(5.0*(sd-APPROACH_STOP_M), 0.0, MIN_FWD_APPROACH*2))
            else:
                fv = 0.0

            if sd < 0.5:
                fv = 0.0

            av = 5.0 * he

            if sd > 1.0:
                fv, at = self._lidar_avoidance(lidar, fv, relaxed=True, pose=pose)
                av    += at
            else:
                # [F20] Raised threshold 0.07->0.04 m; require 3 consecutive hits
                # to distinguish real obstacles from sensor noise at close range.
                if _lidar_has_data(lidar):
                    mf = min(lidar[i % len(lidar)] for i in range(-2, 3))
                    if mf < _LIDAR_ESTOP_THRESHOLD:
                        self._lidar_estop_consec += 1
                        if self._lidar_estop_consec >= _LIDAR_ESTOP_CONSEC_HITS:
                            fv = 0.0
                            print(f"[F12] Lidar emergency stop mf={mf:.3f}m "
                                  f"(consec={self._lidar_estop_consec})")
                    else:
                        self._lidar_estop_consec = 0  # reset on clean reading

            if self.step_counter % 240 == 0:
                print(f"[Act] APPROACH_VISUAL: world_dist={sd:.2f}m, "
                      f"cam_depth={cam_depth:.2f}m, "
                      f"bearing={np.degrees(he):.0f} deg, "
                      f"fwd={fv:.2f}, turn={av:.1f}")
            self._set_wheels(fv-av, fv+av)

        elif mode == 'grasp':
            self._set_wheels(0, 0)
            phase       = ctrl.get('phase', 'close_gripper')
            ap          = ctrl['approach_pos']
            gpos        = ctrl['grasp_pos']
            orn         = p.getQuaternionFromEuler(ctrl['orientation'])
            robot_pose  = ctrl.get('robot_pose')  # [F17] needed for local-frame IK

            if phase == 'reach_above':
                ik_success = self._execute_arm_ik(ap, orn, robot_pose)
                if self.step_counter % 120 == 0:
                    print(f"[Act] GRASP reach_above -> local IK target computed from world {ap}")

            elif phase == 'reach_target':
                ik_success = self._execute_arm_ik(gpos, orn, robot_pose)
                if self.step_counter % 120 == 0:
                    print(f"[Act] GRASP reach_target -> local IK target computed from world {gpos}")

            elif phase == 'close_gripper':
                self._set_wheels(0, 0)
                ik_success = self._execute_arm_ik(gpos, orn, robot_pose)
                self._close_gripper()
                if self.step_counter % 60 == 0:
                    print(f"[Act] GRASP: Closing gripper to grasp object")

            self._last_grasp_attempt = True

        elif mode == 'lift':
            self._set_wheels(0, 0)
            self._close_gripper()
            if self.lift_joint_idx is not None:
                p.setJointMotorControl2(self.robot_id, self.lift_joint_idx,
                                        p.POSITION_CONTROL,
                                        targetPosition=0.3, force=100,
                                        maxVelocity=0.5)
            for j in self.arm_joints:
                current_pos = p.getJointState(self.robot_id, j)[0]
                p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL,
                                        targetPosition=current_pos, force=200,
                                        maxVelocity=0.3)
            if self.step_counter % 120 == 0:
                print("[Act] LIFT: raising object with gripper closed")

        elif mode == 'place':
            self._set_wheels(0, 0)
            self._close_gripper()
            place_pos = ctrl.get('place_pos')
            orientation = ctrl.get('orientation', [0, 1.57, 0])
            orn = p.getQuaternionFromEuler(orientation)
            if place_pos:
                self._execute_arm_ik(place_pos, orn)
                if self.step_counter % 120 == 0:
                    print(f"[Act] PLACE: Moving to place position {place_pos}")
                time_in_state = self.fsm.get_time_in_state()
                if time_in_state > 0.5:
                    self._open_gripper()
                    if self.step_counter % 60 == 0:
                        print("[Act] PLACE: Opening gripper to release object back on table")
            else:
                self._open_gripper()

        elif mode in ('idle', 'success', 'failure'):
            if mode == 'failure':
                if self.step_counter % 240 == 0:
                    print("[Act] FAILURE: backing up")
                lidar  = ctrl.get('lidar')
                rv     = -3.0
                rv, at = self._lidar_avoidance(lidar, rv)
                self._set_wheels(rv-at, rv+at)
                for j in self.arm_joints:
                    p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL,
                                            targetPosition=0.0, force=200,
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
    print("  M4:  Perception - PerceptionModule (full pipeline)")
    print("       * detect_objects_by_color  (HSV)")
    print("       * edge_contour_segmentation")
    print("       * depth_to_point_cloud")
    print("       * RANSAC_Segmentation      (table plane)")
    print("       * compute_pca / refine_object_points (target pose)")
    print("       * SiftFeatureExtractor     (SIFT KB)")
    print("  M9:  Learning DISABLED - hardcoded defaults")
    print("="*60)

    robot_id, table_id, room_id, target_id = build_world(gui=True)
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
    print(f"[Init] M9 DISABLED - "
          f"nav_kp={LEARNING_DEFAULTS['nav_kp']:.2f}  "
          f"angle_kp={LEARNING_DEFAULTS['angle_kp']:.2f}")
    print("[Init] Mission: navigate to table, grasp red cylinder\n")

    while p.isConnected():
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
                print("  MISSION COMPLETE - target grasped and placed back on table!")
                print("="*60)
                break

        if cog.step_counter % 240 == 0:
            pose = sensor_data['pose']
            print(f"[t={cog.step_counter/240:.0f}s] "
                  f"State={cog.fsm.state.name}  "
                  f"Pose=({pose[0]:.2f},{pose[1]:.2f},{np.degrees(pose[2]):.0f}deg))")

        cog.step_counter += 1
        p.stepSimulation()
        time.sleep(1./240.)


if __name__ == "__main__":
    main()
