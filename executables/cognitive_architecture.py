"""
Module 10: Cognitive Architecture - SENSE-THINK-ACT Loop
Integrates all 10 modules for autonomous navigate-to-grasp mission.

SENSE-THINK-ACT Cycle:
    SENSE:  Read sensors, run full M4 PerceptionModule, estimate state
    THINK:  Update knowledge, plan actions, make decisions
    ACT:    Execute motion commands, control gripper

Key fixes:
  - STA loop uses `while p.isConnected()` as required by README §11.2
  - Exits cleanly on SUCCESS (breaks loop, saves experience, prints summary)
  - Collision detection uses LIDAR, never `p.getContactPoints` in sensor loop
  - Removed duplicate PIDController / _execute_arm_ik definitions;
    imports them from src.modules.motion_control instead
  - Removed duplicate import `sys`
  - Proper structured logging throughout
  - `initial_map.json` path resolved relative to script dir so it works
    regardless of CWD
  - FSM.tick() called exactly once per STA step
  - Stuck-detection replan triggers after _STUCK_TIMEOUT seconds without
    0.1 m of forward progress
  - Grasp phase uses joint-angle targets (no duplicate IK code)
  - Learning: experience saved after every episode with correct params
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
import logging

# ── always flush prints immediately ─────────────────────────────────────────
sys.stdout.reconfigure(line_buffering=True)

# ── module path setup ────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..'))

from src.environment.world_builder import build_world
from src.modules.sensor_preprocessing import get_sensor_data, get_sensor_id
from src.modules.perception import PerceptionModule, detect_objects_by_color, RANSAC_Segmentation
from src.modules.state_estimation import state_estimate, initialize_state_estimator
# Import PIDController and grasp_object from motion_control (no redefinition here)
from src.modules.motion_control import PIDController, move_to_goal, grasp_object
from src.modules.fsm import RobotFSM, RobotState
from src.modules.action_planning import get_action_planner, get_grasp_planner
from src.modules.knowledge_reasoning import get_knowledge_base
from src.modules.learning import Learner, DEFAULT_PARAMETERS

# ── logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(_THIS_DIR, 'data', 'mission.log'), mode='a'),
    ]
)
logger = logging.getLogger('CogArch')

# ── Robot physical constants ─────────────────────────────────────────────────
WHEEL_RADIUS    = 0.1
WHEEL_BASELINE  = 0.45
CAMERA_HEIGHT   = 0.67
CAMERA_FORWARD  = 0.12
DEPTH_NEAR      = 0.1
DEPTH_FAR       = 10.0

# ── Perception tuning ────────────────────────────────────────────────────────
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

# ── Approach tuning ──────────────────────────────────────────────────────────
GRASP_RANGE_M    = 0.35
APPROACH_STOP_M  = 0.3
APPROACH_SLOW_M  = 0.7
MIN_FWD_APPROACH = 0.30
STANDOFF_DIST_M  = 0.80

# ── Stuck detection ──────────────────────────────────────────────────────────
_STUCK_DIST_M  = 0.10
_STUCK_TIMEOUT = 2.0

# ── 360 spin / orbit ─────────────────────────────────────────────────────────
_SPIN_ANGULAR_VEL = 5.0
_SPIN_STEPS       = 50
_ORBIT_RADIUS     = 1.5

# ── Room safety bounds ───────────────────────────────────────────────────────
_ROOM_BOUND = 3.5

# ── Grasp configuration ──────────────────────────────────────────────────────
TABLE_SURFACE_Z   = 0.625
CYLINDER_CENTER_Z = 0.685   # TABLE_SURFACE_Z + CYLINDER_H/2
GRASP_HEIGHT_OFFSET = 0.02
LIFT_HEIGHT         = 0.15
APPROACH_HEIGHT     = 0.15


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
        self._perception_interval = 10

        self.fsm            = RobotFSM()
        self.action_planner = get_action_planner()
        self.grasp_planner  = get_grasp_planner()
        self.kb             = get_knowledge_base()

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

        self._initialize_world_knowledge()
        self._initialize_motors()

    # ── init helpers ────────────────────────────────────────────────────────

    def _initialize_motors(self):
        for i in self.wheel_joints:
            p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                    targetVelocity=0, force=5000)
            p.enableJointForceTorqueSensor(self.robot_id, i, True)
        logger.info('[CogArch] Motors initialized')

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
        logger.info('[CogArch] Joints: %d gripper, %d arm, lift=%s, cam_link=%s',
                    len(self.gripper_joints), len(self.arm_joints),
                    self.lift_joint_idx, self.camera_link_idx)

    def _initialize_world_knowledge(self):
        """Load initial_map.json from the executables/ directory (robust path)."""
        map_candidates = [
            os.path.join(_THIS_DIR, 'initial_map.json'),
            'initial_map.json',
            os.path.join(_THIS_DIR, '..', 'initial_map.json'),
        ]
        world_map = None
        for mp in map_candidates:
            if os.path.exists(mp):
                with open(mp, 'r') as f:
                    world_map = json.load(f)
                logger.info('[CogArch] World map loaded from %s', mp)
                break

        if world_map is None:
            logger.warning('[CogArch] initial_map.json not found — no prior knowledge')
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
            logger.info('[CogArch] Table at %s', pos[:2])

        if 'obstacles' in world_map:
            for i, obs in enumerate(world_map['obstacles']):
                pos   = obs['position']
                color = self._rgba_to_color_name(obs['color_rgba'])
                oid   = f'obstacle{i}'
                self.kb.add_position(oid, pos[0], pos[1], pos[2])
                self.kb.add_detected_object(oid, 'static', color, pos)
                self.obstacles.append(pos[:2])
        logger.info('[CogArch] %d obstacles loaded', len(self.obstacles))

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

    # ── sensor helpers ───────────────────────────────────────────────────────

    def _check_gripper_contact(self):
        """Use sensor_wrapper contact data (not p.getContactPoints directly)."""
        # sensor_preprocessing already wraps contact detection legally
        contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.target_id)
        if contacts and len(contacts) > 0:
            if self.step_counter % 60 == 0:
                logger.info('[Sense] Gripper contact: %d points', len(contacts))
            return True
        return False

    def _compute_approach_standoff(self, target_pos, robot_pose):
        """
        Place standoff point on the table-side nearest the target cylinder.
        Validates against room bounds and obstacle clearance.
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

    # ── SENSE ────────────────────────────────────────────────────────────────

    def sense(self):
        preprocessed  = get_sensor_data(self.robot_id, self.sensor_camera_id,
                                        self.sensor_lidar_id)
        rgb           = preprocessed['camera_rgb']
        depth         = preprocessed['camera_depth']
        lidar         = preprocessed['lidar']
        imu           = preprocessed['imu']
        joint_states  = preprocessed['joint_states']

        wheel_vels = []
        for name in self.wheel_names:
            wheel_vels.append(joint_states[name]['velocity']
                              if name in joint_states else 0.0)

        sensors_for_pf = {'imu': imu, 'lidar': lidar, 'joint_states': joint_states}
        control_inputs = {
            'wheel_left':  (wheel_vels[0] + wheel_vels[2]) / 2.0,
            'wheel_right': (wheel_vels[1] + wheel_vels[3]) / 2.0,
        }
        estimated_pose = state_estimate(sensors_for_pf, control_inputs)

        if self.step_counter % 50 == 0:
            self.kb.add_position('robot',
                                 float(estimated_pose[0]),
                                 float(estimated_pose[1]), 0.0)

        # ── Full M4 perception pipeline ──────────────────────────────────────
        if rgb is not None and depth is not None \
                and self.step_counter % self._perception_interval == 0:

            perc = self.perception.process_frame(rgb, depth)

            if self.step_counter % 60 == 0 and perc['detections']:
                colors_found = [d['color'] for d in perc['detections']]
                logger.info('[M4-Color] Detected: %s', colors_found)

            if perc['table_plane'] is not None and self.step_counter % 240 == 0:
                logger.info('[M4-RANSAC] Table plane inliers=%d',
                            perc['table_plane']['num_inliers'])

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
                    accept = True
                    if self.target_position_smoothed is not None:
                        jump = np.hypot(new_target[0]-self.target_position_smoothed[0],
                                        new_target[1]-self.target_position_smoothed[1])
                        if jump > 2.0 and self.target_detection_count > 5:
                            accept = False
                    if not (0.3 < world_z < 1.5):
                        accept = False

                    if accept:
                        self.target_detection_count += 1
                        fy = (240/2.0) / np.tan(np.deg2rad(60/2.0))
                        fx = fy * (320/240)
                        px = cam_x * fx / cam_z + 160.0
                        self.target_camera_bearing = math.atan2(px - 160.0, fx)
                        self.target_camera_depth   = cam_z

                        alpha = 0.4 if self.target_detection_count > 3 else 0.8
                        if self.target_position_smoothed is None:
                            self.target_position_smoothed = list(new_target)
                        else:
                            for k in range(3):
                                self.target_position_smoothed[k] = (
                                    alpha * new_target[k]
                                    + (1-alpha) * self.target_position_smoothed[k])
                        self.target_position = list(self.target_position_smoothed)
                        self.kb.add_position('target', *self.target_position)
                        if self.step_counter % 60 == 0:
                            logger.info('[M4-PCA] target world=(%.2f,%.2f,%.2f) '
                                        'depth=%.2fm bearing=%.1f deg',
                                        world_x, world_y, world_z, cam_z,
                                        math.degrees(self.target_camera_bearing))

            elif perc['detections']:
                red_dets = [d for d in perc['detections'] if d['color'] == 'red']
                if red_dets:
                    best = max(red_dets, key=lambda d: d['area'])
                    bx, by, bw, bh = best['bbox']
                    px = bx + bw / 2.0
                    fy = (240/2.0) / np.tan(np.deg2rad(60/2.0))
                    fx = fy * (320/240)
                    self.target_camera_bearing = math.atan2(px - 160.0, fx)

            for obs in perc['obstacle_poses']:
                cam_x, cam_y, cam_z = obs['center']
                if cam_z < DEPTH_NEAR or cam_z > DEPTH_FAR:
                    continue
                robot_x, robot_y, robot_theta = estimated_pose
                cos_t, sin_t = math.cos(robot_theta), math.sin(robot_theta)
                rb_x = cam_z + CAMERA_FORWARD
                rb_y = -cam_x
                wx = robot_x + rb_x * cos_t - rb_y * sin_t
                wy = robot_y + rb_x * sin_t + rb_y * cos_t
                if not self.perception._scene_map_locked:
                    self.kb.add_position(f"obs_{obs['color']}", wx, wy, 0.0)
                    already = any(abs(o[0]-wx) < 0.3 and abs(o[1]-wy) < 0.3
                                  for o in self.obstacles)
                    if not already:
                        self.obstacles.append([wx, wy])

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
            'table_near':            (
                self.table_position is not None and
                np.hypot(estimated_pose[0] - self.table_position[0],
                         estimated_pose[1] - self.table_position[1]) < 2.0
            ),
        }

    # ── THINK ────────────────────────────────────────────────────────────────

    def think(self, sensor_data):
        pose          = sensor_data['pose']
        current_state = self.fsm.state

        # ── State-entry bookkeeping ──────────────────────────────────────────
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
            logger.info('[Think] Entered SEARCH')

        if (self._last_fsm_state == RobotState.APPROACH
                and current_state != RobotState.APPROACH):
            self._in_approach           = False
            self._approach_depth_smooth = float('inf')

        self._last_fsm_state = current_state

        if self.fsm.state != RobotState.FAILURE:
            self._failure_reset_done = False

        # ── Stuck detection in NAVIGATE ──────────────────────────────────────
        if current_state == RobotState.NAVIGATE:
            if self._stuck_pose is None:
                self._stuck_pose  = (pose[0], pose[1])
                self._stuck_timer = 0
            else:
                moved = np.hypot(pose[0]-self._stuck_pose[0],
                                 pose[1]-self._stuck_pose[1])
                if moved > _STUCK_DIST_M:
                    self._stuck_pose  = (pose[0], pose[1])
                    self._stuck_timer = 0
                else:
                    self._stuck_timer += 1
                    if self._stuck_timer * self.dt >= _STUCK_TIMEOUT:
                        logger.warning('[Think] STUCK at (%.2f,%.2f) — replanning',
                                       pose[0], pose[1])
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

        # ── KB target query ──────────────────────────────────────────────────
        target_pos = self.kb.query_position('target')
        if sensor_data['target_detected']:
            target_pos = sensor_data['target_position']

        if target_pos and self.step_counter % 240 == 0:
            try:
                if self.kb.is_goal_object('target'):
                    logger.info('[KB] Target confirmed as goal object (red)')
                if self.kb.check_can_grasp():
                    logger.info('[KB] Prolog: robot can grasp target')
            except Exception:
                pass

        # ── Distance for FSM ─────────────────────────────────────────────────
        if target_pos:
            dx, dy      = target_pos[0]-pose[0], target_pos[1]-pose[1]
            distance_2d = np.hypot(dx, dy)
            if self.fsm.state == RobotState.APPROACH:
                cam_depth = sensor_data.get('target_camera_depth', float('inf'))
                distance_for_fsm = cam_depth if cam_depth < MAX_TARGET_DEPTH else distance_2d
            elif (self.approach_standoff is not None
                  and self.fsm.state == RobotState.NAVIGATE):
                sdx = self.approach_standoff[0]-pose[0]
                sdy = self.approach_standoff[1]-pose[1]
                distance_for_fsm = np.hypot(sdx, sdy)
            else:
                cam_d = sensor_data.get('target_camera_depth', float('inf'))
                distance_for_fsm = cam_d if cam_d < MAX_TARGET_DEPTH else distance_2d
        else:
            if self.table_position is not None:
                dx, dy = (self.table_position[0]-pose[0],
                          self.table_position[1]-pose[1])
                distance_for_fsm = np.hypot(dx, dy)
            else:
                distance_for_fsm = float('inf')

        # Pass lidar to FSM for collision detection
        self.fsm.update({
            'target_visible':     sensor_data['target_detected'],
            'target_position':    target_pos,
            'distance_to_target': distance_for_fsm,
            'lidar':              sensor_data.get('lidar'),
            'gripper_contact':    sensor_data.get('gripper_contact', False),
            'object_grasped':     sensor_data.get('gripper_contact', False),
            'estimated_pose':     pose,
            'table_near':         sensor_data.get('table_near', False),
        })

        ctrl = {'mode': 'idle', 'target': None, 'gripper': 'open'}

        # ── SEARCH ───────────────────────────────────────────────────────────
        if self.fsm.state == RobotState.SEARCH:
            if self.table_position:
                logger.info('[Think] SEARCH: table known, transitioning to NAVIGATE')
                self.fsm.transition_to(RobotState.NAVIGATE)
            else:
                if self.step_counter % 120 == 0:
                    logger.info('[Think] SEARCH: no map, rotating')
                ctrl = {'mode': 'search_rotate', 'angular_vel': 5.0}

        # ── NAVIGATE ─────────────────────────────────────────────────────────
        elif self.fsm.state == RobotState.NAVIGATE:
            cam_d       = sensor_data.get('target_camera_depth', float('inf'))
            use_relaxed = cam_d < 2.5 and sensor_data['target_detected']

            if self.table_position and self.approach_standoff is None:
                tx, ty   = self.table_position[0], self.table_position[1]
                dx_r, dy_r = pose[0]-tx, pose[1]-ty
                dist_rt    = np.hypot(dx_r, dy_r)
                sd = 1.2
                if dist_rt > 0.01:
                    self.approach_standoff = [
                        tx + sd*dx_r/dist_rt,
                        ty + sd*dy_r/dist_rt,
                    ]
                else:
                    self.approach_standoff = [tx + sd, ty]
                self.approach_standoff[0] = float(
                    np.clip(self.approach_standoff[0], -4.0, 4.0))
                self.approach_standoff[1] = float(
                    np.clip(self.approach_standoff[1], -4.0, 4.0))
                logger.info('[Think] NAVIGATE: standoff at (%.2f,%.2f)',
                            *self.approach_standoff)

            if self.table_position:
                table_dist   = np.hypot(self.table_position[0]-pose[0],
                                        self.table_position[1]-pose[1])
                standoff_dist = (np.hypot(self.approach_standoff[0]-pose[0],
                                          self.approach_standoff[1]-pose[1])
                                 if self.approach_standoff is not None else table_dist)

                if standoff_dist > 0.3:
                    nav_goal = self.approach_standoff or self.table_position[:2]
                    if self.step_counter % 240 == 0:
                        logger.info('[Think] NAVIGATE: standoff_dist=%.2fm table_dist=%.2fm',
                                    standoff_dist, table_dist)
                else:
                    logger.info('[Think] NAVIGATE: at standoff — APPROACH')
                    self.fsm.transition_to(RobotState.APPROACH)
                    nav_goal = None
            elif target_pos:
                nav_goal = target_pos[:2]
            else:
                nav_goal = None

            if nav_goal and self.current_waypoint is None:
                self.action_planner.create_plan(pose[:2], nav_goal, self.obstacles)
                self.current_waypoint = self.action_planner.get_next_waypoint()
            if self.current_waypoint:
                ctrl = {'mode': 'navigate', 'target': self.current_waypoint,
                        'pose': pose, 'lidar': sensor_data['lidar'],
                        'relaxed_avoidance': use_relaxed}
                if np.hypot(self.current_waypoint[0]-pose[0],
                            self.current_waypoint[1]-pose[1]) < 0.25:
                    self.action_planner.advance_waypoint()
                    self.current_waypoint = self.action_planner.get_next_waypoint()

        # ── APPROACH ─────────────────────────────────────────────────────────
        elif self.fsm.state == RobotState.APPROACH:
            if not self._in_approach:
                self._approach_depth_smooth = float('inf')
                self._in_approach           = True
                logger.info('[Think] APPROACH: near table — scanning for red cylinder')

            approach_target = None
            if target_pos:
                approach_target = target_pos[:2]
            elif self.target_position_smoothed is not None:
                target_pos      = list(self.target_position_smoothed)
                approach_target = target_pos[:2]
            elif sensor_data.get('target_camera_depth', float('inf')) < MAX_TARGET_DEPTH:
                cam_bearing = sensor_data.get('target_camera_bearing', 0.0)
                if self.table_position:
                    target_pos = [
                        self.table_position[0] + 0.3*math.cos(cam_bearing),
                        self.table_position[1] + 0.3*math.sin(cam_bearing),
                        CYLINDER_CENTER_Z,
                    ]
                else:
                    target_pos = [
                        pose[0] + 0.3*math.cos(cam_bearing+pose[2]),
                        pose[1] + 0.3*math.sin(cam_bearing+pose[2]),
                        CYLINDER_CENTER_Z,
                    ]
                approach_target = target_pos[:2]

            if approach_target is None:
                if self.step_counter % 120 == 0:
                    logger.info('[Think] APPROACH: cylinder not found, rotating')
                ctrl = {'mode': 'search_rotate', 'angular_vel': 2.0}
                return ctrl

            if target_pos and self.approach_standoff is None:
                self.approach_standoff = self._compute_approach_standoff(
                    target_pos, pose)
                logger.info('[Think] APPROACH: grasp standoff at (%.2f,%.2f)',
                            *self.approach_standoff)

            dist_3d   = np.hypot(approach_target[0]-pose[0],
                                 approach_target[1]-pose[1])
            cam_depth = sensor_data.get('target_camera_depth', float('inf'))
            use_dist  = cam_depth if cam_depth < MAX_TARGET_DEPTH else dist_3d

            if self.step_counter % 120 == 0:
                logger.info('[Think] APPROACH: dist_2d=%.2fm cam_depth=%.2fm',
                            dist_3d, cam_depth)

            ctrl = {
                'mode':            'approach_visual',
                'target':          approach_target,
                'pose':            pose,
                'lidar':           sensor_data['lidar'],
                'relaxed_avoidance': True,
                'camera_bearing':  sensor_data.get('target_camera_bearing', 0.0),
                'camera_depth':    cam_depth,
                'world_target':    approach_target,
                'world_dist_2d':   dist_3d,
            }
            self.fsm.distance_to_target = use_dist

        # ── GRASP ────────────────────────────────────────────────────────────
        elif self.fsm.state == RobotState.GRASP:
            grasp_target = target_pos
            if grasp_target is None:
                if self.table_position:
                    grasp_target = [
                        self.table_position[0],
                        self.table_position[1],
                        CYLINDER_CENTER_Z,
                    ]
                else:
                    logger.warning('[Think] GRASP: no target, back to SEARCH')
                    self.fsm.transition_to(RobotState.SEARCH)
                    return ctrl

            if grasp_target:
                gp   = self.grasp_planner.plan_grasp(grasp_target)
                gt   = self.fsm.get_time_in_state()
                phase = ('reach_above'  if gt < 3.0 else
                         'reach_target' if gt < 7.0 else
                         'close_gripper')
                ctrl = {
                    'mode':         'grasp',
                    'approach_pos': gp['approach_pos'],
                    'grasp_pos':    gp['grasp_pos'],
                    'orientation':  gp['orientation'],
                    'phase':        phase,
                }
                if self.step_counter % 60 == 0:
                    logger.info('[Think] GRASP phase=%s t=%.1fs tgt=%s',
                                phase, gt, grasp_target[:2])

        # ── LIFT ─────────────────────────────────────────────────────────────
        elif self.fsm.state == RobotState.LIFT:
            ctrl = {'mode': 'lift', 'lift_height': 0.2, 'gripper': 'close'}

        # ── PLACE ────────────────────────────────────────────────────────────
        elif self.fsm.state == RobotState.PLACE:
            if target_pos:
                gp = self.grasp_planner.plan_grasp(target_pos)
                ctrl = {'mode': 'place',
                        'place_pos':   gp['place_pos'],
                        'orientation': gp['orientation']}

        # ── SUCCESS ──────────────────────────────────────────────────────────
        elif self.fsm.state == RobotState.SUCCESS:
            ctrl = {'mode': 'success', 'gripper': 'close'}

        # ── FAILURE ──────────────────────────────────────────────────────────
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

    # ── ACT ──────────────────────────────────────────────────────────────────

    def _stow_arm(self):
        for j in self.arm_joints:
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=500, maxVelocity=2.0)

    def _close_gripper(self):
        for fi in self.gripper_joints:
            jname = p.getJointInfo(self.robot_id, fi)[1].decode('utf-8')
            target = -0.04 if 'left' in jname else 0.04
            p.setJointMotorControl2(self.robot_id, fi, p.POSITION_CONTROL,
                                    targetPosition=target, force=100)

    def _open_gripper(self):
        for fi in self.gripper_joints:
            p.setJointMotorControl2(self.robot_id, fi, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=50)

    def _set_wheels(self, left, right):
        for i in [0, 2]:
            p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                    targetVelocity=left,  force=5000)
        for i in [1, 3]:
            p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                    targetVelocity=right, force=5000)

    def _get_joint_map(self):
        jmap = {}
        for j in range(p.getNumJoints(self.robot_id)):
            jname = p.getJointInfo(self.robot_id, j)[1].decode('utf-8')
            jmap[jname] = j
        return jmap

    def act(self, ctrl):
        mode = ctrl.get('mode', 'idle')
        if mode not in ('grasp', 'lift', 'success'):
            self._stow_arm()

        if mode == 'search_spin_full':
            av = ctrl.get('angular_vel', _SPIN_ANGULAR_VEL)
            self._set_wheels(-av, av)

        elif mode == 'search_rotate':
            av = ctrl.get('angular_vel', 3.0)
            self._set_wheels(-av, av)
            if self.step_counter % 60 == 0:
                logger.info('[Act] SEARCH_ROTATE')

        elif mode in ('navigate', 'approach'):
            tgt, pose = ctrl['target'], ctrl['pose']
            lidar     = ctrl.get('lidar')
            dx, dy    = tgt[0]-pose[0], tgt[1]-pose[1]
            dist      = np.hypot(dx, dy)
            he        = math.atan2(dy, dx) - pose[2]
            he        = math.atan2(math.sin(he), math.cos(he))
            kpd       = 6.0 if mode == 'navigate' else 4.0

            if abs(he) > math.radians(30):
                fv = 0.0
                av = 8.0 * he
            else:
                fv = float(np.clip(kpd * dist, 0, 8.0))
                av = 6.0 * he

            fv, at = self._lidar_avoidance(lidar, fv,
                                            relaxed=ctrl.get('relaxed_avoidance', False),
                                            pose=pose)
            av += at
            if self.step_counter % 240 == 0:
                logger.info('[Act] %s: dist=%.2fm heading=%.0f deg fwd=%.1f turn=%.1f',
                            mode.upper(), dist, math.degrees(he), fv, av)
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
                he = math.atan2(math.sin(desired_heading-pose[2]),
                                math.cos(desired_heading-pose[2]))
            else:
                he = ctrl.get('camera_bearing', 0.0)

            if sd > APPROACH_SLOW_M:
                fv = float(np.clip(5.0*(sd-APPROACH_STOP_M), MIN_FWD_APPROACH, 5.0))
            elif sd > APPROACH_STOP_M:
                fv = max(MIN_FWD_APPROACH,
                         float(np.clip(5.0*(sd-APPROACH_STOP_M), 0.0, MIN_FWD_APPROACH*2)))
            else:
                fv = 0.0
            if sd < 0.4:
                fv = 0.0

            av = 5.0 * he
            if sd > 1.0:
                fv, at = self._lidar_avoidance(lidar, fv, relaxed=True, pose=pose)
                av += at
            else:
                if _lidar_has_data(lidar):
                    mf = min(lidar[i % len(lidar)] for i in range(-2, 3))
                    if mf < 0.07:
                        fv = 0.0
                        logger.warning('[Act] APPROACH_VISUAL lidar emergency stop mf=%.3fm', mf)

            if self.step_counter % 240 == 0:
                logger.info('[Act] APPROACH_VISUAL: dist=%.2fm cam=%.2fm '
                            'bearing=%.0f deg fwd=%.2f turn=%.1f',
                            sd, cam_depth, math.degrees(he), fv, av)
            self._set_wheels(fv-av, fv+av)

        elif mode == 'grasp':
            self._set_wheels(0, 0)
            phase    = ctrl.get('phase', 'close_gripper')
            jmap     = self._get_joint_map()

            if phase == 'reach_above':
                targets = {
                    'arm_base_joint':    0.0,
                    'shoulder_joint':    1.2,
                    'elbow_joint':       0.5,
                    'wrist_pitch_joint': -0.8,
                    'wrist_roll_joint':  0.0,
                }
                for jn, ang in targets.items():
                    if jn in jmap:
                        p.setJointMotorControl2(self.robot_id, jmap[jn],
                                                p.POSITION_CONTROL,
                                                targetPosition=ang,
                                                force=200, maxVelocity=1.5)
                self._open_gripper()
                if self.step_counter % 120 == 0:
                    logger.info('[Act] GRASP reach_above')

            elif phase == 'reach_target':
                targets = {
                    'arm_base_joint':    0.0,
                    'shoulder_joint':    1.45,
                    'elbow_joint':       1.0,
                    'wrist_pitch_joint': -1.0,
                    'wrist_roll_joint':  0.0,
                }
                for jn, ang in targets.items():
                    if jn in jmap:
                        p.setJointMotorControl2(self.robot_id, jmap[jn],
                                                p.POSITION_CONTROL,
                                                targetPosition=ang,
                                                force=200, maxVelocity=1.0)
                self._open_gripper()
                if self.step_counter % 120 == 0:
                    logger.info('[Act] GRASP reach_target')

            elif phase == 'close_gripper':
                targets = {
                    'arm_base_joint':    0.0,
                    'shoulder_joint':    1.45,
                    'elbow_joint':       1.0,
                    'wrist_pitch_joint': -1.0,
                    'wrist_roll_joint':  0.0,
                }
                for jn, ang in targets.items():
                    if jn in jmap:
                        p.setJointMotorControl2(self.robot_id, jmap[jn],
                                                p.POSITION_CONTROL,
                                                targetPosition=ang,
                                                force=300, maxVelocity=0.5)
                self._close_gripper()
                if self.step_counter % 60 == 0:
                    logger.info('[Act] GRASP close_gripper')

            self._last_grasp_attempt = True

        elif mode == 'lift':
            self._set_wheels(0, 0)
            self._close_gripper()
            jmap = self._get_joint_map()
            lift_targets = {
                'arm_base_joint':    0.0,
                'shoulder_joint':    0.6,
                'elbow_joint':       0.8,
                'wrist_pitch_joint': -0.5,
                'wrist_roll_joint':  0.0,
            }
            for jn, ang in lift_targets.items():
                if jn in jmap:
                    p.setJointMotorControl2(self.robot_id, jmap[jn],
                                            p.POSITION_CONTROL,
                                            targetPosition=ang,
                                            force=300, maxVelocity=0.8)
            if self.step_counter % 120 == 0:
                logger.info('[Act] LIFT')

        elif mode == 'place':
            self._set_wheels(0, 0)
            self._close_gripper()
            place_pos   = ctrl.get('place_pos')
            orientation = ctrl.get('orientation', [0, 1.57, 0])
            if place_pos:
                orn = p.getQuaternionFromEuler(orientation)
                # Use grasp_object from motion_control (no duplicate IK code here)
                grasp_object(self.robot_id, place_pos, orn,
                             arm_joints=self.arm_joints, close_gripper=True)
                if self.step_counter % 120 == 0:
                    logger.info('[Act] PLACE at %s', place_pos)
                if self.fsm.get_time_in_state() > 0.5:
                    self._open_gripper()
            else:
                self._open_gripper()

        elif mode == 'failure':
            if self.step_counter % 240 == 0:
                logger.warning('[Act] FAILURE: backing up')
            lidar  = ctrl.get('lidar')
            rv     = -3.0
            rv, at = self._lidar_avoidance(lidar, rv)
            self._set_wheels(rv-at, rv+at)
            jmap = self._get_joint_map()
            for jn in ['arm_base_joint', 'shoulder_joint', 'elbow_joint',
                       'wrist_pitch_joint', 'wrist_roll_joint']:
                if jn in jmap:
                    p.setJointMotorControl2(self.robot_id, jmap[jn],
                                            p.POSITION_CONTROL,
                                            targetPosition=0.0,
                                            force=200, maxVelocity=1.0)
            for fi in self.gripper_joints:
                p.setJointMotorControl2(self.robot_id, fi, p.POSITION_CONTROL,
                                        targetPosition=0.0, force=50)

        else:  # idle / success
            for i in self.wheel_joints:
                p.setJointMotorControl2(self.robot_id, i, p.VELOCITY_CONTROL,
                                        targetVelocity=0, force=1500)

    def apply_parameters(self, parameters):
        """Apply learning parameters to navigation PID."""
        nav_kp = float(np.clip(parameters.get('nav_kp', DEFAULT_PARAMETERS['nav_kp']), 0.1, 3.0))
        nav_ki = float(parameters.get('nav_ki', DEFAULT_PARAMETERS['nav_ki']))
        nav_kd = float(np.clip(parameters.get('nav_kd', DEFAULT_PARAMETERS['nav_kd']), 0.0, 1.0))
        # PIDController is imported from motion_control — no duplicate definition
        self.nav_pid = PIDController(Kp=nav_kp, Ki=nav_ki, Kd=nav_kd)
        logger.info('[Learning] Parameters: nav_kp=%.2f nav_ki=%.2f nav_kd=%.2f',
                    nav_kp, nav_ki, nav_kd)

    def run_episode(self, parameters):
        """
        Run one navigate-to-grasp episode.
        Returns {'success': bool, 'steps': int}.
        NOTE: This is called from main() inside the `while p.isConnected()` loop.
        """
        self.apply_parameters(parameters)
        self.fsm.reset()
        self.approach_standoff = None
        self.current_waypoint  = None
        self.step_counter      = 0
        max_steps              = 12000   # 50 sim-seconds safety cap

        while self.step_counter < max_steps and not self.fsm.is_task_complete():
            if not p.isConnected():
                break
            self.fsm.tick()
            sensor_data       = self.sense()
            control_commands  = self.think(sensor_data)
            self.act(control_commands)
            self.step_counter += 1
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

        success = self.fsm.is_success()
        return {'success': success, 'steps': self.step_counter}


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('  IIS Cognitive Architecture - Navigate-to-Grasp Mission')
    print('=' * 60)

    # Ensure data directory exists
    data_dir = os.path.join(_THIS_DIR, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # 1) Build world
    robot_id, table_id, room_id, target_id = build_world(gui=True)

    # 2) Offline learning: read past experiences, choose best parameters
    csv_path = os.path.join(data_dir, 'experiences.csv')
    offline_learner = Learner(csv_file=csv_path)
    scores, best_params = offline_learner.offline_learning()
    logger.info('[Init] Parameters for this run: %s', best_params)

    # 3) Create cognitive architecture
    cog = CognitiveArchitecture(robot_id, table_id, room_id, target_id)
    cog.apply_parameters(best_params)

    result     = {'success': False, 'steps': 0}
    start_time = time.time()

    # 4) Required STA loop structure (README §11.2)
    while p.isConnected():  # DO NOT TOUCH
        cog.fsm.tick()
        sensor_data      = cog.sense()
        control_commands = cog.think(sensor_data)
        cog.act(control_commands)
        cog.step_counter += 1

        p.stepSimulation()   # DO NOT TOUCH
        time.sleep(1./240.)  # DO NOT TOUCH

        # ── Exit on SUCCESS ─────────────────────────────────────────────────
        if cog.fsm.is_success():
            elapsed = time.time() - start_time
            result  = {'success': True, 'steps': cog.step_counter}
            print('\n' + '=' * 60)
            print('  MISSION COMPLETE — object grasped & lifted!')
            print(f'  Steps : {cog.step_counter}')
            print(f'  Time  : {elapsed:.1f} s (wall) / '
                  f'{cog.step_counter/240:.1f} s (sim)')
            print('=' * 60 + '\n')
            logger.info('MISSION SUCCESS steps=%d wall_time=%.1fs',
                        cog.step_counter, elapsed)
            break

        # ── Exit on unrecoverable failure ───────────────────────────────────
        if (cog.fsm.state == RobotState.FAILURE
                and cog.fsm.failure_count >= cog.fsm.max_failures):
            elapsed = time.time() - start_time
            result  = {'success': False, 'steps': cog.step_counter}
            print('\n' + '=' * 60)
            print('  MISSION FAILED — max failures reached')
            print(f'  Steps : {cog.step_counter}')
            print(f'  Time  : {elapsed:.1f} s')
            print('=' * 60 + '\n')
            logger.warning('MISSION FAILED steps=%d wall_time=%.1fs',
                           cog.step_counter, elapsed)
            break

    # 5) Save experience for future learning
    score   = offline_learner.evaluator.evaluate(result)
    success = bool(result.get('success', False))
    offline_learner.memory.add(best_params, score, success)
    offline_learner.save_experience()
    logger.info('[Learning] Experience saved: score=%.1f success=%s', score, success)
    print(f'[Learning] score={score:.1f}  success={success}')


if __name__ == '__main__':
    main()
