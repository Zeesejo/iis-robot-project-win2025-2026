"""
Module 10: Cognitive Architecture - SENSE-THINK-ACT Loop
Integrates all 10 modules for autonomous navigate-to-grasp mission.

SENSE-THINK-ACT Cycle:
    SENSE:  Read sensors, run full M4 PerceptionModule, estimate state
    THINK:  Update knowledge, plan actions, make decisions
    ACT:    Execute motion commands, control gripper

FIXES in this revision
  [F1] fsm.tick() now called once per STA step – step-based timeouts work.
  [F2] CAMERA_HEIGHT updated to 0.67 m (robot-1.urdf: base_spawn(0.1)
       + torso_joint_z(0.3) + cam_z_offset(0.27) = 0.67 m).
  [F3] approach_standoff reset on every NAVIGATE re-entry, not only on
       FAILURE, so stale standoffs from previous runs are discarded.
  [F4] APPROACH_VISUAL rewritten: uses world-frame bearing to tracked
       target (not raw camera bearing which is 0 when no fresh detection),
       stops cleanly when depth < 0.55 m, and clamps fwd speed tightly.
  [F5] Raised dist-to-table rejection gate to 3.5 m (was 3.0).
  [F6] distance_for_fsm during APPROACH uses 2D world-frame distance to
       target (not frozen camera depth) so GRASP transition fires at <0.55m.
  [F7] approach_visual speed in act() uses np.hypot(world_tgt - pose)
       so the robot slows down proportionally and does not ram the table.
  [F8] _in_approach + _approach_depth_smooth reset when FSM leaves APPROACH
       so stale state never carries over to the next attempt.
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

# M2
from src.environment.world_builder import build_world
# M3
from src.modules.sensor_preprocessing import get_sensor_data, get_sensor_id
# M4 - full perception module
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
# M5
from src.modules.state_estimation import state_estimate, initialize_state_estimator
# M6
from src.modules.motion_control import PIDController, move_to_goal, grasp_object
# M7
from src.modules.fsm import RobotFSM, RobotState
from src.modules.action_planning import get_action_planner, get_grasp_planner
# M8
from src.modules.knowledge_reasoning import get_knowledge_base
# M9: Learning [DISABLED]
LEARNING_DEFAULTS = {'nav_kp': 1.0, 'nav_ki': 0.0, 'nav_kd': 0.1, 'angle_kp': 1.0}

# ── Robot physical constants ─────────────────────────────────────────────
WHEEL_RADIUS    = 0.1
WHEEL_BASELINE  = 0.45
# [F2] robot-1.urdf camera is on torso_link at xyz="0.12 0 0.27":
#   base_spawn(0.1) + torso_joint_z(0.3) + cam_z(0.27) = 0.67 m
CAMERA_HEIGHT   = 0.67
CAMERA_FORWARD  = 0.12   # camera 0.12 m forward on torso (kept for PCA path only)
DEPTH_NEAR      = 0.1
DEPTH_FAR       = 10.0

# ── Perception tuning ────────────────────────────────────────────────────
_FOV_V   = 60.0
_ASPECT  = 320.0 / 240.0
_FOV_H   = 2 * np.degrees(np.arctan(np.tan(np.radians(_FOV_V / 2)) * _ASPECT))
CAM_FX   = (320 / 2.0) / np.tan(np.radians(_FOV_H / 2))
CAM_FY   = (240 / 2.0) / np.tan(np.radians(_FOV_V / 2))
CAM_CX, CAM_CY = 160.0, 120.0

TARGET_COLOR     = 'red'
MAX_TARGET_DEPTH = 3.5   # [F5] gate raised
MIN_TARGET_DEPTH = 0.2
MAX_JUMP_M       = 1.0   # EMA jump filter threshold

# Camera tilt compensation (rpy="0 0.2 0" in robot-1.urdf)
_CAM_TILT = 0.2   # radians downward

# ── Approach tuning ──────────────────────────────────────────────────────
# [F6] FSM APPROACH→GRASP triggers when 2D world-frame distance < this
GRASP_RANGE_M   = 0.55
# [F7] During APPROACH, robot slows to near-zero at this world distance
APPROACH_SLOW_M = 1.0   # start slowing below 1.0 m
APPROACH_STOP_M = 0.55  # stop wheels at this distance


class CognitiveArchitecture:
    """Main cognitive architecture: Sense-Think-Act loop."""

    def __init__(self, robot_id, table_id, room_id, target_id):
        self.robot_id  = robot_id
        self.table_id  = table_id
        self.room_id   = room_id
        self.target_id = target_id

        # M5
        initialize_state_estimator()

        # M3
        self.sensor_camera_id, self.sensor_lidar_id = get_sensor_id(self.robot_id)

        # M4
        self.perception = PerceptionModule()

        # M7
        self.fsm            = RobotFSM()
        self.action_planner = get_action_planner()
        self.grasp_planner  = get_grasp_planner()

        # M8
        self.kb = get_knowledge_base()

        # M9 [DISABLED]
        nav_kp = LEARNING_DEFAULTS['nav_kp']
        nav_ki = LEARNING_DEFAULTS['nav_ki']
        nav_kd = LEARNING_DEFAULTS['nav_kd']

        # M6
        self.nav_pid = PIDController(Kp=nav_kp, Ki=nav_ki, Kd=nav_kd)

        # Robot joints
        self.wheel_joints = [0, 1, 2, 3]
        self.wheel_names  = ['fl_wheel_joint', 'fr_wheel_joint',
                             'bl_wheel_joint', 'br_wheel_joint']
        self.arm_joints      = []
        self.gripper_joints  = []
        self.lift_joint_idx  = None
        self.camera_link_idx = None
        self._detect_robot_joints()

        # Task state
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
        self._last_fsm_state           = None   # [F3] track state changes

        # M4 perception results
        self.last_perception_result = None
        self.table_plane_model      = None
        self.pca_target_pose        = None

        self._failure_reset_done = False

        # [F8] APPROACH-specific state – reset whenever we leave APPROACH
        self._in_approach           = False
        self._approach_depth_smooth = float('inf')

        # Timing
        self.step_counter = 0
        self.dt           = 1.0 / 240.0

        self._initialize_world_knowledge()
        self._initialize_motors()

    # ─────────────────────────── initialisation ───────────────────────────

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

    # ─────────────────────────── helpers ────────────────────────────────────

    def _check_gripper_contact(self):
        contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.target_id)
        if contacts and len(contacts) > 0:
            if self.step_counter % 60 == 0:
                print("[Sense] Gripper contact detected")
            return True
        return False

    def _compute_approach_standoff(self, target_pos, robot_pose):
        """
        Compute a standoff point 0.65 m from the target on the side
        facing the robot.  Uses table orientation if known.
        """
        standoff_dist = 0.65
        if self.table_orientation is not None:
            yaw  = p.getEulerFromQuaternion(self.table_orientation)[2]
            dir1 = [-math.sin(yaw),  math.cos(yaw)]
            dir2 = [ math.sin(yaw), -math.cos(yaw)]
            dx   = robot_pose[0] - target_pos[0]
            dy   = robot_pose[1] - target_pos[1]
            adir = dir1 if (dx*dir1[0]+dy*dir1[1]) > (dx*dir2[0]+dy*dir2[1]) else dir2
            return [target_pos[0] + standoff_dist*adir[0],
                    target_pos[1] + standoff_dist*adir[1]]
        dx, dy = target_pos[0]-robot_pose[0], target_pos[1]-robot_pose[1]
        d      = np.hypot(dx, dy)
        if d > 0.01:
            return [target_pos[0] - standoff_dist*dx/d,
                    target_pos[1] - standoff_dist*dy/d]
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
        if lidar is None or len(lidar) == 0:
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

    # ────────────────────── camera → world projection ──────────────────────

    def _pixel_depth_to_world(self, px, py, depth_m, robot_pose):
        """
        Convert (pixel, depth_metres) to world-frame [x,y,z].
        Accounts for camera tilt (_CAM_TILT rad downward pitch).
        """
        # Normalised image coordinates
        nx = (px - CAM_CX) / CAM_FX
        ny = (py - CAM_CY) / CAM_FY

        # Un-project to camera frame (z forward, x right, y down)
        cam_x = depth_m * nx
        cam_y = depth_m * ny
        cam_z = depth_m

        # Rotate by camera pitch (tilt down = positive pitch around y-axis)
        ct = math.cos(_CAM_TILT)
        st = math.sin(_CAM_TILT)
        body_forward = ct * cam_z - st * cam_y   # x in body frame
        body_up      = st * cam_z + ct * cam_y   # z in body frame (positive = up if cam_y<0)
        body_lateral = cam_x                     # y in body frame (right = positive)

        rx, ry, rt = robot_pose
        world_x = rx + body_forward * math.cos(rt) - (-body_lateral) * math.sin(rt)
        world_y = ry + body_forward * math.sin(rt) + (-body_lateral) * math.cos(rt)
        world_z = CAMERA_HEIGHT - body_up   # body_up positive = below camera

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
                    alpha * new_pos[k] + (1 - alpha) * self.target_position_smoothed[k]
                )
        self.target_position = list(self.target_position_smoothed)
        self.kb.add_position('target', *self.target_position)
        return True

    # ════════════════════════════ SENSE ═══════════════════════════════

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
                pca = self.pca_target_pose
                if self.step_counter % 240 == 0:
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
                depth_arr  = np.array(depth).reshape(240, 320)
                seg_mask, edge_map = edge_contour_segmentation(
                    bgr, min_contour_area=300)
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

                if self.table_position is not None:
                    dist_to_table = np.hypot(wp[0] - self.table_position[0],
                                             wp[1] - self.table_position[1])
                    if dist_to_table > MAX_TARGET_DEPTH:
                        if self.step_counter % 60 == 0:
                            print(f"[M4] Rejected detection at "
                                  f"({wp[0]:.2f},{wp[1]:.2f}) – "
                                  f"{dist_to_table:.2f}m from table")
                        continue

                if self.pca_target_pose is not None:
                    pca_c      = self.pca_target_pose['center']
                    pca_body_x = pca_c[2]
                    pca_body_y = -pca_c[0]
                    rx, ry, rt = estimated_pose
                    pca_wx     = rx + pca_body_x*math.cos(rt) - pca_body_y*math.sin(rt)
                    pca_wy     = ry + pca_body_x*math.sin(rt) + pca_body_y*math.cos(rt)
                    pca_wz     = CAMERA_HEIGHT - pca_c[1]
                    pca_world  = [pca_wx, pca_wy, pca_wz]
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

    # ════════════════════════════ THINK ═══════════════════════════════

    def think(self, sensor_data):
        pose = sensor_data['pose']

        # [F3] Reset standoff whenever we re-enter NAVIGATE from another state
        current_state = self.fsm.state
        if (current_state == RobotState.NAVIGATE
                and self._last_fsm_state != RobotState.NAVIGATE):
            self.approach_standoff = None
            self.current_waypoint  = None

        # [F8] Reset approach state whenever we LEAVE APPROACH
        if (self._last_fsm_state == RobotState.APPROACH
                and current_state != RobotState.APPROACH):
            self._in_approach           = False
            self._approach_depth_smooth = float('inf')

        self._last_fsm_state = current_state

        if self.fsm.state != RobotState.FAILURE:
            self._failure_reset_done = False

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

            # [F6] Use 2D world-frame distance as the FSM distance signal
            # during APPROACH so the GRASP transition fires correctly.
            # During NAVIGATE, keep using standoff distance so the robot
            # stops at the standoff point, not at the target itself.
            if self.fsm.state == RobotState.APPROACH:
                distance_for_fsm = distance_2d
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

        # ── SEARCH ──────────────────────────────────────────────────────────
        if self.fsm.state == RobotState.SEARCH:
            if self.table_position:
                td = np.hypot(self.table_position[0]-pose[0],
                              self.table_position[1]-pose[1])
                if td < 2.0:
                    ctrl = {'mode': 'search_orbit',
                            'table_pos': self.table_position[:2],
                            'pose': pose, 'orbit_radius': 2.0,
                            'lidar': sensor_data['lidar']}
                else:
                    ctrl = {'mode': 'search_approach',
                            'target': self.table_position[:2],
                            'pose': pose, 'angular_vel': 2.0,
                            'lidar': sensor_data['lidar']}
            else:
                ctrl = {'mode': 'search_rotate', 'angular_vel': 3.0}

        # ── NAVIGATE ────────────────────────────────────────────────────────
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

        # ── APPROACH ────────────────────────────────────────────────────────
        # [F4] Use world-frame target vector for bearing; tighter speed clamp.
        # [F7] Speed now computed from 2D world-frame distance (not frozen cam depth).
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

        # ── GRASP ───────────────────────────────────────────────────────────
        elif self.fsm.state == RobotState.GRASP:
            if target_pos:
                gp    = self.grasp_planner.plan_grasp(target_pos)
                gt    = self.fsm.get_time_in_state()
                phase = ('reach_above'  if gt < 2.5 else
                         'reach_target' if gt < 5.5 else
                         'close_gripper')
                ctrl = {'mode': 'grasp',
                        'approach_pos': gp['approach_pos'],
                        'grasp_pos':    gp['grasp_pos'],
                        'orientation':  gp['orientation'],
                        'phase':        phase}

        # ── LIFT ────────────────────────────────────────────────────────────
        elif self.fsm.state == RobotState.LIFT:
            ctrl = {'mode': 'lift', 'lift_height': 0.2, 'gripper': 'close'}

        # ── SUCCESS ─────────────────────────────────────────────────────────
        elif self.fsm.state == RobotState.SUCCESS:
            ctrl = {'mode': 'success', 'gripper': 'close'}

        # ── FAILURE ─────────────────────────────────────────────────────────
        elif self.fsm.state == RobotState.FAILURE:
            if not self._failure_reset_done:
                self.approach_standoff        = None
                self.current_waypoint         = None
                self.target_position_smoothed = None
                self.target_detection_count   = 0
                self.pca_target_pose          = None
                self.table_plane_model        = None
                self._failure_reset_done      = True
            ctrl = {'mode': 'failure', 'gripper': 'open',
                    'lidar': sensor_data['lidar']}

        return ctrl

    # ════════════════════════════ ACT ═════════════════════════════════

    def _stow_arm(self):
        for j in self.arm_joints:
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL,
                                    targetPosition=0.0, force=500, maxVelocity=2.0)

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

        if mode == 'search_rotate':
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
            tp, pose = ctrl['table_pos'], ctrl['pose']
            r_orb    = ctrl.get('orbit_radius', 2.0)
            lidar    = ctrl.get('lidar')
            dx, dy   = pose[0]-tp[0], pose[1]-tp[1]
            cur_r    = np.hypot(dx, dy)
            ang_ft   = math.atan2(dy, dx)
            tangent  = ang_ft + math.pi/2
            rerr     = cur_r - r_orb
            desired  = tangent + 0.5*rerr
            he       = math.atan2(math.sin(desired-pose[2]), math.cos(desired-pose[2]))
            fv       = 2.0
            av       = 4.0*he
            fv, at   = self._lidar_avoidance(lidar, fv, pose=pose)
            av      += at
            if self.step_counter % 240 == 0:
                print(f"[Act] ORBIT r={cur_r:.2f}m (target {r_orb:.1f}m)")
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
            # [F7] Speed is now based on 2D world-frame distance to target,
            # not on the frozen EMA camera depth.  This means the robot slows
            # down proportionally as it closes in, regardless of whether a
            # fresh depth pixel was seen this tick.
            pose       = ctrl.get('pose')
            world_tgt  = ctrl.get('world_target') or ctrl.get('target')
            lidar      = ctrl.get('lidar')
            cam_depth  = ctrl.get('camera_depth', float('inf'))
            world_dist = ctrl.get('world_dist_2d', float('inf'))

            # [F7] Use world_dist (2D) for speed control; fall back to
            # cam_depth only if world position is unavailable.
            if world_dist < float('inf'):
                sd = world_dist
            else:
                # EMA smooth depth (only decreases fast, increases slow)
                if cam_depth < self._approach_depth_smooth:
                    self._approach_depth_smooth = cam_depth
                else:
                    self._approach_depth_smooth = (0.7 * self._approach_depth_smooth
                                                   + 0.3 * cam_depth)
                sd = self._approach_depth_smooth

            # World-frame heading error to target
            if world_tgt is not None and pose is not None:
                dx_w = world_tgt[0] - pose[0]
                dy_w = world_tgt[1] - pose[1]
                desired_heading = math.atan2(dy_w, dx_w)
                he = math.atan2(math.sin(desired_heading - pose[2]),
                                math.cos(desired_heading - pose[2]))
            else:
                he = ctrl.get('camera_bearing', 0.0)

            # [F7] Speed: proportional to world distance, stop at APPROACH_STOP_M
            if sd > APPROACH_SLOW_M:
                fv = np.clip(2.0 * (sd - APPROACH_STOP_M), 0.3, 3.0)
            elif sd > APPROACH_STOP_M:
                fv = np.clip(1.5 * (sd - APPROACH_STOP_M), 0.05, 1.0)
            else:
                fv = 0.0   # within grasp range – stop wheels

            av = 5.0 * he

            # Lidar obstacle avoidance (relaxed — we're near the table)
            if sd > 1.0:
                fv, at = self._lidar_avoidance(lidar, fv, relaxed=True, pose=pose)
                av    += at
            else:
                if lidar:
                    mf = min(lidar[i % len(lidar)] for i in range(-2, 3))
                    if mf < 0.12:
                        fv = 0.0

            if self.step_counter % 240 == 0:
                print(f"[Act] APPROACH_VISUAL: world_dist={sd:.2f}m, "
                      f"cam_depth={cam_depth:.2f}m, "
                      f"bearing={np.degrees(he):.0f} deg, "
                      f"fwd={fv:.1f}, turn={av:.1f}")
            self._set_wheels(fv-av, fv+av)

        elif mode == 'grasp':
            self._set_wheels(0, 0)
            phase = ctrl.get('phase', 'close_gripper')
            ap    = ctrl['approach_pos']
            gpos  = ctrl['grasp_pos']
            orn   = p.getQuaternionFromEuler(ctrl['orientation'])
            if self.lift_joint_idx is not None:
                p.setJointMotorControl2(self.robot_id, self.lift_joint_idx,
                                        p.POSITION_CONTROL,
                                        targetPosition=0.3, force=100,
                                        maxVelocity=0.5)
            close = (phase == 'close_gripper')
            tgt_p = ap if phase == 'reach_above' else gpos
            if self.step_counter % 120 == 0:
                print(f"[Act] GRASP phase={phase}")
            grasp_object(self.robot_id, tgt_p, orn,
                         arm_joints=self.arm_joints or None,
                         close_gripper=close)

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


# ═════════════════════════════════ MAIN ══════════════════════════════════

def main():
    print("="*60)
    print("  IIS Cognitive Architecture - Navigate-to-Grasp Mission")
    print("="*60)
    print("  M4:  Perception - PerceptionModule (full pipeline)")
    print("       • detect_objects_by_color  (HSV)")
    print("       • edge_contour_segmentation")
    print("       • depth_to_point_cloud")
    print("       • RANSAC_Segmentation      (table plane)")
    print("       • compute_pca / refine_object_points (target pose)")
    print("       • SiftFeatureExtractor     (SIFT KB)")
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
    print(f"[Init] M9 DISABLED – "
          f"nav_kp={LEARNING_DEFAULTS['nav_kp']:.2f}  "
          f"angle_kp={LEARNING_DEFAULTS['angle_kp']:.2f}")
    print("[Init] Mission: navigate to table, grasp red cylinder\n")

    # ── SENSE-THINK-ACT loop ──────────────────────────────────────────────
    while p.isConnected():                      # DO NOT TOUCH
        try:
            cog.fsm.tick()                      # [F1] advance step-based FSM timer
            sensor_data      = cog.sense()
            control_commands = cog.think(sensor_data)
            cog.act(control_commands)
        except p.error as e:
            print(f"[Main] PyBullet disconnected: {e}")
            break

        if cog.fsm.state == RobotState.SUCCESS:
            if cog.fsm.get_time_in_state() > 3.0:
                print("\n" + "="*60)
                print("  MISSION COMPLETE – target grasped and lifted!")
                print("="*60)
                break

        if cog.step_counter % 240 == 0:
            pose = sensor_data['pose']
            print(f"[t={cog.step_counter/240:.0f}s] "
                  f"State={cog.fsm.state.name}  "
                  f"Pose=({pose[0]:.2f},{pose[1]:.2f},{np.degrees(pose[2]):.0f}°))")

        cog.step_counter += 1
        p.stepSimulation()                      # DO NOT TOUCH
        time.sleep(1./240.)                     # DO NOT TOUCH


if __name__ == "__main__":
    main()
