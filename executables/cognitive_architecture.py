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
  [F6]  distance_for_fsm during APPROACH uses cam_depth (not world 2D dist).
  [F7]  approach_visual speed uses world_dist / cam_depth.
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
  [F16] APPROACH->GRASP: use cam_depth (not 2D world dist) for FSM trigger.
  [F17] Removed creep nudge past APPROACH_STOP_M that prevented stopping.
  [F18] Raised lidar emergency stop threshold 0.07->0.15 m.
  [F19] GRASP block uses approach_standoff as arm approach position.
  [F20] Target world_z clamped to [0.60, 0.95].
  [F21] world_builder: force red via changeVisualShape; widen HSV; no fallback PCA.
  [F22] Hard-reset target smoother on NAVIGATE->APPROACH transition so
        accumulated orbit detections don't poison the close-range estimate.
  [F23] FSM: lower NAVIGATE->APPROACH threshold 2.0->1.2 m.
  [F24] approach_visual: clamp |bearing| <= 25 deg during close approach
        to prevent hard turns from stale wide-angle target estimates.
  [F25] approach_visual: prefer cam_depth over world_dist when world_dist
        is unreliable (> 2.5 m while cam_depth < 2.5 m).
  [F26] motion_control: convert world-frame grasp target to robot-body
        frame before IK; clamp reach to 0.55 m; force 500->50 N.
  [F27] motion_control: pre-IK stow (phase='stow') resets arm config.
  [F28] cognitive_architecture GRASP: compute grasp target as a forward
        offset from robot base in robot-frame, not raw world coords.
        Pre-IK stow inserted as phase 0 (t < 1 s).
  [F29] approach_visual: hard stop if world_dist < 0.45 m (under-table).
  [F30] NAVIGATE: stop if robot is within 0.4 m of table edge to prevent
        driving under table.
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
from src.modules.perception import (
    PerceptionModule,
    detect_objects_by_color,
    RANSAC_Segmentation,
    depth_to_point_cloud,
    edge_contour_segmentation,
    compute_pca,
    refine_object_points,
    SCENE_OBJECTS,
    COLOR_RANGES_HSV,
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
_ASPECT  = 320.0 / 240.0
_FOV_H   = 2 * np.degrees(np.arctan(np.tan(np.radians(_FOV_V / 2)) * _ASPECT))
CAM_FX   = (320 / 2.0) / np.tan(np.radians(_FOV_H / 2))
CAM_FY   = (240 / 2.0) / np.tan(np.radians(_FOV_V / 2))
CAM_CX, CAM_CY = 160.0, 120.0

TARGET_COLOR     = 'red'
MAX_TARGET_DEPTH = 3.5
MIN_TARGET_DEPTH = 0.2
MAX_JUMP_M       = 1.0
_CAM_TILT        = 0.2

# -- Approach tuning ---------------------------------------------------------
GRASP_RANGE_M    = 0.55
APPROACH_STOP_M  = 0.40
APPROACH_SLOW_M  = 1.0
MIN_FWD_APPROACH = 0.50
STANDOFF_DIST_M  = 0.65

# -- Stuck detection ---------------------------------------------------------
_STUCK_DIST_M  = 0.10
_STUCK_TIMEOUT = 4.0

# -- [F14-A] 360 spin constants ----------------------------------------------
_SPIN_ANGULAR_VEL = 3.0
_SPIN_STEPS       = 150

# -- [F15-B] Orbit radius reduced for better small-object visibility ---------
_ORBIT_RADIUS = 1.5   # was 2.0

# -- Room safety bounds ------------------------------------------------------
_ROOM_BOUND = 3.5

# -- [F20] Target Z clamp (cylinder sits on table ~0.625 m) ------------------
_TARGET_Z_MIN = 0.60
_TARGET_Z_MAX = 0.95

# -- [F26] Arm reach constants -----------------------------------------------
# Forward offset from robot centre to arm base ~0.25 m; max arm reach ~0.55 m
# So max useful forward reach from robot centre = 0.25 + 0.55 = 0.80 m
# We target 0.45 m forward from base (conservative but reachable)
_ARM_FORWARD_OFFSET = 0.45   # m forward from robot base centre
_ARM_GRASP_Z_BODY   = 0.65   # m above robot base (table top ~0.72 m world -> ~0.62 m body)


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
            self.obstacles.append(pos[:2])
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
        contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.target_id)
        if contacts and len(contacts) > 0:
            if self.step_counter % 60 == 0:
                print("[Sense] Gripper contact detected")
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
            perc = self.perception.process_frame(rgb, depth, width=320, height=240)
            self.last_perception_result = perc

            if perc['table_plane'] is not None:
                model   = perc['table_plane']['model']
                inliers = perc['table_plane']['num_inliers']
                if abs(model[2]) > 0.7 and inliers > 200:
                    self.table_plane_model = model
                    if self.step_counter % 240 == 0:
                        print(f"[M4-RANSAC] Table plane confirmed: "
                              f"A={model[0]:.3f} B={model[1]:.3f} "
                              f"C={model[2]:.3f} D={model[3]:.3f} "
                              f"({inliers} inliers)")

            if perc['target_pose'] is not None:
                self.pca_target_pose = perc['target_pose']
                if self.step_counter % 240 == 0:
                    pca = self.pca_target_pose
                    print(f"[M4-PCA] Object cloud centre (cam-frame): "
                          f"{[f'{v:.3f}' for v in pca['center']]}, "
                          f"dims={[f'{v:.3f}' for v in pca['dimensions']]}")

            if perc['obstacle_poses'] and self.step_counter % 240 == 0:
                for op in perc['obstacle_poses']:
                    print(f"[M4-PCA] Obstacle '{op['color']}' centre: "
                          f"{[f'{v:.3f}' for v in op['center']]}")

            rgb_array  = np.reshape(rgb, (240, 320, 4)).astype(np.uint8)
            bgr        = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)
            detections = perc['detections']

            if self.step_counter % 240 == 0 and detections:
                print(f"[M4-Color] {len(detections)} detections: "
                      f"{[d['color'] for d in detections]}")

            if self.step_counter % 240 == 0:
                depth_arr_log = np.array(depth).reshape(240, 320)
                seg_mask, edge_map = edge_contour_segmentation(bgr, min_contour_area=300)
                print(f"[M4-Edge] seg_pixels={int(np.sum(seg_mask > 0))}, "
                      f"edge_pixels={int(np.sum(edge_map > 0))}")

            depth_arr = np.array(depth).reshape(240, 320)
            red_dets  = sorted(
                [d for d in detections if d['color'] == TARGET_COLOR],
                key=lambda d: d['area'], reverse=True
            )

            for det in red_dets[:1]:
                x, y, w, h = det['bbox']
                cx_px = int(x + w / 2)
                cy_px = int(y + h / 2)
                if not (0 <= cy_px < 240 and 0 <= cx_px < 320):
                    continue
                raw_d = depth_arr[cy_px, cx_px]
                if raw_d <= 0 or raw_d >= 1.0:
                    continue
                true_d = DEPTH_FAR * DEPTH_NEAR / (
                    DEPTH_FAR - (DEPTH_FAR - DEPTH_NEAR) * raw_d)
                if not (MIN_TARGET_DEPTH < true_d < MAX_TARGET_DEPTH):
                    continue
                if np.isnan(true_d) or np.isinf(true_d):
                    continue
                if self.target_position_smoothed is None and true_d > 3.0:
                    continue

                bearing = math.atan2(-(cx_px - CAM_CX), CAM_FX)
                wp      = self._pixel_depth_to_world(cx_px, cy_px,
                                                     true_d, estimated_pose)

                # [F20] Clamp world Z to plausible cylinder-on-table range
                wp[2] = float(np.clip(wp[2], _TARGET_Z_MIN, _TARGET_Z_MAX))

                if self.table_position is not None:
                    dist_to_table = np.hypot(wp[0] - self.table_position[0],
                                             wp[1] - self.table_position[1])
                    if dist_to_table > MAX_TARGET_DEPTH:
                        if self.step_counter % 60 == 0:
                            print(f"[M4] Rejected detection at "
                                  f"({wp[0]:.2f},{wp[1]:.2f}) - "
                                  f"{dist_to_table:.2f}m from table")
                        continue

                if self.pca_target_pose is not None:
                    pca_c      = self.pca_target_pose['center']
                    pca_body_x = pca_c[2]
                    pca_body_y = -pca_c[0]
                    rx, ry, rt = estimated_pose
                    pca_wx = rx + pca_body_x*math.cos(rt) - pca_body_y*math.sin(rt)
                    pca_wy = ry + pca_body_x*math.sin(rt) + pca_body_y*math.cos(rt)
                    pca_wz = float(np.clip(
                        CAMERA_HEIGHT - pca_c[1], _TARGET_Z_MIN, _TARGET_Z_MAX))
                    pca_world = [pca_wx, pca_wy, pca_wz]
                    if np.hypot(pca_world[0]-wp[0], pca_world[1]-wp[1]) < 0.5:
                        wp = pca_world
                        if self.step_counter % 60 == 0:
                            print(f"[M4-PCA] Using PCA target centre: "
                                  f"({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})")

                accepted = self._update_target_from_detection(wp, true_d, bearing)
                if accepted and self.step_counter % 10 == 0:
                    print(f"[CogArch] TARGET DETECTED at "
                          f"({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f}), "
                          f"depth={true_d:.2f}m")
                break

        gripper_contact = self._check_gripper_contact()

        return {
            'pose':                  estimated_pose,
            'rgb':                   rgb,
            'depth':                 depth,
            'lidar':                 lidar,
            'imu':                   imu,
            'joint_states':          joint_states,
            'target_detected':       self.target_position is not None,
            'target_position':       self.target_position,
            'target_camera_bearing': self.target_camera_bearing,
            'target_camera_depth':   self.target_camera_depth,
            'gripper_contact':       gripper_contact,
            'perception':            self.last_perception_result,
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

        # [F22] Hard-reset target smoother on NAVIGATE->APPROACH
        if (current_state == RobotState.APPROACH
                and self._last_fsm_state == RobotState.NAVIGATE):
            print("[F22] APPROACH entry: resetting target position smoother")
            self.target_position_smoothed = None
            self.target_detection_count   = 0
            self.pca_target_pose          = None

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

        if target_pos:
            dx, dy      = target_pos[0]-pose[0], target_pos[1]-pose[1]
            distance_2d = np.hypot(dx, dy)
            if self.approach_standoff is None:
                self.approach_standoff = self._compute_approach_standoff(
                    target_pos, pose)
                print(f"[CogArch] Computed approach standoff: "
                      f"({self.approach_standoff[0]:.2f}, "
                      f"{self.approach_standoff[1]:.2f})")

            # [F16] During APPROACH use camera depth to trigger FSM->GRASP
            if self.fsm.state == RobotState.APPROACH:
                cam_d = sensor_data.get('target_camera_depth', float('inf'))
                distance_for_fsm = (cam_d
                                    if MIN_TARGET_DEPTH < cam_d < MAX_TARGET_DEPTH
                                    else distance_2d)
            elif (self.approach_standoff is not None
                  and self.fsm.state == RobotState.NAVIGATE):
                sdx = self.approach_standoff[0]-pose[0]
                sdy = self.approach_standoff[1]-pose[1]
                distance_for_fsm = np.hypot(sdx, sdy)
            else:
                cam_d = sensor_data.get('target_camera_depth', float('inf'))
                distance_for_fsm = cam_d if cam_d < MAX_TARGET_DEPTH else distance_2d
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
            if not self._spin_complete:
                self._spin_steps_done += 1
                if self._spin_steps_done >= _SPIN_STEPS:
                    self._spin_complete = True
                    print("[F14-A] 360 spin complete - proceeding to search approach")
                ctrl = {'mode': 'search_spin_full',
                        'angular_vel': _SPIN_ANGULAR_VEL}
            else:
                if self.table_position:
                    td = np.hypot(self.table_position[0]-pose[0],
                                  self.table_position[1]-pose[1])
                    if td < _ORBIT_RADIUS + 0.5:
                        ctrl = {'mode': 'search_orbit',
                                'table_pos': self.table_position[:2],
                                'pose': pose,
                                'orbit_radius': _ORBIT_RADIUS,
                                'lidar': sensor_data['lidar']}
                    else:
                        ctrl = {'mode': 'search_approach',
                                'target': self.table_position[:2],
                                'pose': pose, 'angular_vel': 2.0,
                                'lidar': sensor_data['lidar']}
                else:
                    ctrl = {'mode': 'search_rotate', 'angular_vel': 3.0}

        # -- NAVIGATE --------------------------------------------------------
        elif self.fsm.state == RobotState.NAVIGATE:
            cam_d       = sensor_data.get('target_camera_depth', float('inf'))
            use_relaxed = cam_d < 2.5 and sensor_data['target_detected']
            nav_goal    = (self.approach_standoff or
                           (target_pos[:2] if target_pos else None))
            if nav_goal and self.current_waypoint is None:
                self.action_planner.create_plan(pose[:2], nav_goal, self.obstacles)
                self.current_waypoint = self.action_planner.get_next_waypoint()
            if self.current_waypoint:
                ctrl = {'mode': 'navigate', 'target': self.current_waypoint,
                        'pose': pose, 'lidar': sensor_data['lidar'],
                        'relaxed_avoidance': use_relaxed}
                if np.hypot(self.current_waypoint[0]-pose[0],
                            self.current_waypoint[1]-pose[1]) < 0.3:
                    self.action_planner.advance_waypoint()
                    self.current_waypoint = self.action_planner.get_next_waypoint()

        # -- APPROACH --------------------------------------------------------
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

        # -- GRASP -----------------------------------------------------------
        elif self.fsm.state == RobotState.GRASP:
            # [F28] Compute grasp target in body-relative world coords.
            # Instead of using the raw (potentially noisy) world target position,
            # we compute a point that is exactly _ARM_FORWARD_OFFSET metres in
            # front of the robot base and _ARM_GRASP_Z_BODY above the base.
            # This is always within the arm's reach envelope.
            base_pos_pb, base_orn_pb = p.getBasePositionAndOrientation(self.robot_id)
            base_yaw = p.getEulerFromQuaternion(base_orn_pb)[2]

            # World-frame grasp position: forward from robot, at table height
            grasp_wx = base_pos_pb[0] + _ARM_FORWARD_OFFSET * math.cos(base_yaw)
            grasp_wy = base_pos_pb[1] + _ARM_FORWARD_OFFSET * math.sin(base_yaw)
            grasp_wz = base_pos_pb[2] + _ARM_GRASP_Z_BODY

            # Above-grasp position (for reach_above phase)
            above_wx = base_pos_pb[0] + _ARM_FORWARD_OFFSET * math.cos(base_yaw)
            above_wy = base_pos_pb[1] + _ARM_FORWARD_OFFSET * math.sin(base_yaw)
            above_wz = grasp_wz + 0.15

            gt = self.fsm.get_time_in_state()
            # Phase 0 (t < 1 s): stow arm to reset config before IK
            # Phase 1 (1-3 s):   reach above
            # Phase 2 (3-6 s):   reach target
            # Phase 3 (>6 s):    close gripper
            phase = ('stow'          if gt < 1.0  else
                     'reach_above'   if gt < 3.0  else
                     'reach_target'  if gt < 6.0  else
                     'close_gripper')

            approach_pos = [above_wx, above_wy, above_wz]
            grasp_pos    = [grasp_wx, grasp_wy, grasp_wz]

            if self.step_counter % 120 == 0:
                print(f"[Act] GRASP phase={phase}, "
                      f"target=({grasp_wx:.2f},{grasp_wy:.2f},{grasp_wz:.2f})")

            ctrl = {'mode': 'grasp',
                    'approach_pos': approach_pos,
                    'grasp_pos':    grasp_pos,
                    'orientation':  [0, 0, 0],
                    'phase':        phase}

        # -- LIFT ------------------------------------------------------------
        elif self.fsm.state == RobotState.LIFT:
            ctrl = {'mode': 'lift', 'lift_height': 0.2, 'gripper': 'close'}

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
            if self.step_counter % 60 == 0:
                pct = min(100, int(100 * self._spin_steps_done / _SPIN_STEPS))
                print(f"[F14-A] Spinning {pct}% of 360")

        elif mode == 'search_rotate':
            av = ctrl.get('angular_vel', 3.0)
            self._set_wheels(-av, av)

        elif mode == 'search_approach':
            tgt, pose, lidar = ctrl['target'], ctrl['pose'], ctrl.get('lidar')
            dx, dy = tgt[0]-pose[0], tgt[1]-pose[1]
            dist   = np.hypot(dx, dy)
            he     = math.atan2(dy, dx) - pose[2]
            he     = math.atan2(math.sin(he), math.cos(he))
            fv     = min(3.0, 2.0*dist)
            av     = 4.0*he
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
            fv      = 2.0
            av      = 4.0 * he
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
            kpd       = 4.0 if mode == 'navigate' else 3.0
            fv        = np.clip(kpd*dist, 0, 5.0)
            av        = 5.0*he
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

            # [F29] Hard stop if robot is too close - likely under/against table
            if world_dist < 0.45:
                self._set_wheels(0, 0)
                if self.step_counter % 60 == 0:
                    print(f"[F29] Hard stop: world_dist={world_dist:.2f}m < 0.45m")
                return

            # [F25] Prefer cam_depth when world position estimate is stale
            if (cam_depth < MAX_TARGET_DEPTH and
                    (world_dist > cam_depth * 1.5 or world_dist > 2.5)):
                sd = cam_depth
            elif world_dist < float('inf'):
                sd = world_dist
            else:
                sd = cam_depth

            # [F24] Clamp bearing to avoid hard turns from stale estimates
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
                fv = np.clip(3.0 * (sd - APPROACH_STOP_M), MIN_FWD_APPROACH, 3.0)
            elif sd > APPROACH_STOP_M:
                fv = max(MIN_FWD_APPROACH,
                         np.clip(3.0*(sd-APPROACH_STOP_M), 0.0, MIN_FWD_APPROACH*2))
            else:
                fv = 0.0

            av = 5.0 * he

            if sd > 1.0:
                fv, at = self._lidar_avoidance(lidar, fv, relaxed=True, pose=pose)
                av    += at
            else:
                if _lidar_has_data(lidar):
                    mf = min(lidar[i % len(lidar)] for i in range(-2, 3))
                    if mf < 0.15:
                        fv = 0.0
                        print(f"[F18] Lidar emergency stop mf={mf:.3f}m")

            if self.step_counter % 240 == 0:
                print(f"[Act] APPROACH_VISUAL: sd={sd:.2f}m, "
                      f"cam_depth={cam_depth:.2f}m, world_dist={world_dist:.2f}m, "
                      f"bearing={np.degrees(he):.0f} deg (clamped to "
                      f"±{np.degrees(bearing_limit):.0f}), "
                      f"fwd={fv:.2f}, turn={av:.1f}")
            self._set_wheels(fv-av, fv+av)

        elif mode == 'grasp':
            self._set_wheels(0, 0)
            phase = ctrl.get('phase', 'reach_target')
            ap    = ctrl['approach_pos']
            gpos  = ctrl['grasp_pos']
            orn   = p.getQuaternionFromEuler(ctrl['orientation'])

            if self.lift_joint_idx is not None:
                p.setJointMotorControl2(self.robot_id, self.lift_joint_idx,
                                        p.POSITION_CONTROL,
                                        targetPosition=0.3, force=100,
                                        maxVelocity=0.5)

            tgt_p = ap if phase in ('reach_above', 'stow') else gpos
            close = (phase == 'close_gripper')

            grasp_object(self.robot_id, tgt_p, orn,
                         arm_joints=self.arm_joints or None,
                         close_gripper=close,
                         phase=phase)

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
            if self.step_counter % 120 == 0:
                print("[Act] LIFT: raising object")

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
                print("  MISSION COMPLETE - target grasped and lifted!")
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
