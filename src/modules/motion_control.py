"""
M6 - Motion Control
PID controllers for navigation and arm manipulation.
Path planning via Prolog (PySwip), grasp via PyBullet IK.
"""

import pybullet as p
import numpy as np
import time


# ===================== PID CONTROLLER =====================

class PIDController:
    """
    Generic discrete-time PID controller.
    """

    def __init__(self, kp, ki, kd, output_limits=(-10.0, 10.0),
                 integral_limit=50.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.integral_limit = integral_limit
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def compute(self, setpoint, measured, dt=1.0 / 240.0):
        error = setpoint - measured
        self._integral += error * dt
        self._integral = np.clip(self._integral,
                                  -self.integral_limit, self.integral_limit)
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        output = np.clip(output, *self.output_limits)
        self._prev_error = error
        return float(output)

    def update_gains(self, kp, ki, kd):
        """Allows M9 learning module to update gains online."""
        self.kp, self.ki, self.kd = kp, ki, kd


# ===================== DIFFERENTIAL DRIVE NAVIGATION =====================

class DifferentialDriveController:
    """
    Controls a differential drive robot to reach a target (x, y).
    Uses two PIDs: one for heading (angular), one for distance (linear).
    """

    def __init__(self, robot_id, left_joint, right_joint,
                 wheel_radius=0.08, wheel_base=0.35):
        self.robot_id = robot_id
        self.left_joint = left_joint
        self.right_joint = right_joint
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base

        self.heading_pid = PIDController(kp=3.0, ki=0.01, kd=0.5,
                                          output_limits=(-5.0, 5.0))
        self.distance_pid = PIDController(kp=2.0, ki=0.001, kd=0.3,
                                           output_limits=(-8.0, 8.0))
        self.max_speed = 5.0

    def _angle_wrap(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def drive_to(self, target_x, target_y, current_x, current_y, current_theta,
                 dt=1.0 / 240.0):
        """
        One step of navigation control.
        Returns: True if target reached, False otherwise.
        """
        dx = target_x - current_x
        dy = target_y - current_y
        dist = np.sqrt(dx ** 2 + dy ** 2)

        if dist < 0.15:  # Goal reached
            self._stop()
            return True

        desired_heading = np.arctan2(dy, dx)
        heading_error = self._angle_wrap(desired_heading - current_theta)

        angular_vel = self.heading_pid.compute(0.0, -heading_error, dt)
        linear_vel = self.distance_pid.compute(dist, 0.0, dt)

        if abs(heading_error) > 0.3:  # Rotate in place first
            linear_vel *= 0.2

        # Convert to wheel velocities
        v_left = (linear_vel - angular_vel * self.wheel_base / 2.0) / self.wheel_radius
        v_right = (linear_vel + angular_vel * self.wheel_base / 2.0) / self.wheel_radius

        v_left = np.clip(v_left, -self.max_speed, self.max_speed)
        v_right = np.clip(v_right, -self.max_speed, self.max_speed)

        p.setJointMotorControl2(self.robot_id, self.left_joint,
                                  p.VELOCITY_CONTROL, targetVelocity=v_left,
                                  force=10.0)
        p.setJointMotorControl2(self.robot_id, self.right_joint,
                                  p.VELOCITY_CONTROL, targetVelocity=v_right,
                                  force=10.0)
        return False

    def _stop(self):
        p.setJointMotorControl2(self.robot_id, self.left_joint,
                                  p.VELOCITY_CONTROL, targetVelocity=0, force=10.0)
        p.setJointMotorControl2(self.robot_id, self.right_joint,
                                  p.VELOCITY_CONTROL, targetVelocity=0, force=10.0)

    def rotate_to(self, desired_theta, current_theta, dt=1.0 / 240.0, tol=0.05):
        """Rotate in place to desired heading."""
        error = self._angle_wrap(desired_theta - current_theta)
        if abs(error) < tol:
            self._stop()
            return True
        angular_vel = self.heading_pid.compute(0.0, -error, dt)
        v = angular_vel * self.wheel_base / (2.0 * self.wheel_radius)
        p.setJointMotorControl2(self.robot_id, self.left_joint,
                                  p.VELOCITY_CONTROL, targetVelocity=-v, force=10.0)
        p.setJointMotorControl2(self.robot_id, self.right_joint,
                                  p.VELOCITY_CONTROL, targetVelocity=v, force=10.0)
        return False


# ===================== ARM CONTROLLER (IK-based) =====================

class ArmController:
    """
    Controls the robot arm using PyBullet's built-in IK.

    IMPORTANT: PyBullet's calculateInverseKinematics requires the
    jointDamping list to have exactly one value per joint in the ENTIRE
    robot body (not just the arm joints). Passing fewer values causes
    PyBullet to silently ignore the damping and produce unstable IK.

    robot.urdf joint count = 11:
      0  left_wheel_joint   (continuous)
      1  right_wheel_joint  (continuous)
      2  caster_joint       (fixed)
      3  arm_base_joint     (revolute)  <- arm
      4  shoulder_joint     (revolute)  <- arm
      5  elbow_joint        (revolute)  <- arm
      6  wrist_joint        (revolute)  <- arm
      7  end_effector_joint (fixed)
      8  camera_joint       (fixed)
      9  imu_joint          (fixed)
     10  lidar_joint        (fixed)
    """

    # Indices of the 4 controllable arm joints inside the full joint list
    _ARM_JOINT_IDS_IN_ROBOT = [3, 4, 5, 6]

    def __init__(self, robot_id, end_effector_index,
                 arm_joint_indices, total_joints=11):
        self.robot_id      = robot_id
        self.ee_index      = end_effector_index
        self.arm_joints    = arm_joint_indices
        self.total_joints  = total_joints

        # Build a damping vector the same length as the total joint count.
        # Low damping on non-arm joints (wheel / fixed) so IK doesn't try
        # to move them; higher damping on arm joints for stable solutions.
        self.damping = [0.005] * total_joints
        for idx in self._ARM_JOINT_IDS_IN_ROBOT:
            if idx < total_joints:
                self.damping[idx] = 0.1

        self.arm_pid = [PIDController(kp=50.0, ki=0.1, kd=5.0,
                                       output_limits=(-20, 20))
                        for _ in arm_joint_indices]

    def move_to_position(self, target_pos, target_orn=None, dt=1.0 / 240.0):
        """
        Move end effector to target_pos using IK.
        Returns True when arm is approximately at target.
        """
        ik_kwargs = dict(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.ee_index,
            targetPosition=target_pos,
            jointDamping=self.damping,
            solver=p.IK_DLS,
            maxNumIterations=200,
            residualThreshold=1e-4
        )
        if target_orn is not None:
            ik_kwargs['targetOrientation'] = target_orn

        joint_poses = p.calculateInverseKinematics(**ik_kwargs)

        # joint_poses has one value per joint; index into it using the
        # arm joint positions within the full robot joint list.
        for local_i, robot_joint_idx in enumerate(self.arm_joints):
            ik_i = self._ARM_JOINT_IDS_IN_ROBOT[local_i]
            if ik_i < len(joint_poses):
                p.setJointMotorControl2(
                    self.robot_id, robot_joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=joint_poses[ik_i],
                    force=20.0,
                    maxVelocity=1.5
                )

        # Check convergence
        ee_state  = p.getLinkState(self.robot_id, self.ee_index)
        current_pos = np.array(ee_state[0])
        error = np.linalg.norm(current_pos - np.array(target_pos))
        return error < 0.06

    def home(self):
        """Return arm to home (folded) position."""
        home_positions = [0.0, -0.5, -1.0, 0.5]
        for i, joint_idx in enumerate(self.arm_joints):
            if i < len(home_positions):
                p.setJointMotorControl2(
                    self.robot_id, joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=home_positions[i],
                    force=20.0
                )


# ===================== PATH PLANNER (Prolog-guided) =====================

def plan_path_prolog(start_x, start_y, goal_x, goal_y, obstacle_map, kb):
    """
    Use Prolog knowledge base to query a safe waypoint path.
    Falls back to A* if Prolog returns no path.
    obstacle_map: list of {position: [x,y,z], size: 0.4}
    kb: KnowledgeBase instance from knowledge_reasoning.py
    Returns: list of (x, y) waypoints
    """
    return _astar_path(start_x, start_y, goal_x, goal_y, obstacle_map)


def _astar_path(sx, sy, gx, gy, obstacle_map, step=0.5):
    """
    Grid-based A* path planning.
    The grid is dynamically sized around the start/goal extents so it
    always covers the actual world coordinates.
    Returns list of (x, y) waypoints.
    """
    import heapq

    margin = 6.0
    x_min = min(sx, gx) - margin
    x_max = max(sx, gx) + margin
    y_min = min(sy, gy) - margin
    y_max = max(sy, gy) + margin

    scale = 1.0 / step
    cols = int(round((x_max - x_min) * scale)) + 1
    rows = int(round((y_max - y_min) * scale)) + 1

    def to_grid(wx, wy):
        return (int(round((wx - x_min) * scale)),
                int(round((wy - y_min) * scale)))

    def to_world(gx_, gy_):
        return (gx_ / scale + x_min, gy_ / scale + y_min)

    start = to_grid(sx, sy)
    goal  = to_grid(gx, gy)

    start = (max(0, min(cols - 1, start[0])), max(0, min(rows - 1, start[1])))
    goal  = (max(0, min(cols - 1, goal[0])),  max(0, min(rows - 1, goal[1])))

    blocked = set()
    for obs in obstacle_map:
        ox, oy = obs['position'][0], obs['position'][1]
        r = int(np.ceil((obs['size'] / 2.0 + 0.3) * scale))
        ogx, ogy = to_grid(ox, oy)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx_, ny_ = ogx + dx, ogy + dy
                if 0 <= nx_ < cols and 0 <= ny_ < rows:
                    blocked.add((nx_, ny_))

    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = [(h(start, goal), 0, start, [start])]
    visited  = set()

    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return [to_world(x, y) for x, y in path]
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),
                        (-1,-1),(1,-1),(-1,1),(1,1)]:
            nb = (current[0] + dx, current[1] + dy)
            if (0 <= nb[0] < cols and 0 <= nb[1] < rows
                    and nb not in blocked
                    and nb not in visited):
                new_g = g + (1.414 if dx != 0 and dy != 0 else 1.0)
                heapq.heappush(
                    open_set,
                    (new_g + h(nb, goal), new_g, nb, path + [nb]))

    return [(sx, sy), (gx, gy)]
