"""
Module 7: Action Planning
High-level task sequencing and obstacle-aware waypoint generation.
"""

import numpy as np
import math


# Room bounds (keep 0.5m inset from ±5 walls)
ROOM_BOUNDS = (-4.5, 4.5, -4.5, 4.5)

# Clearance radii
_CLEARANCE_TABLE = 1.2
_CLEARANCE_SMALL = 0.65

_TABLE_AVOID_MARGIN = 0.35
_TABLE_BYPASS_MARGIN = 0.60


def _in_bounds(pt):
    return (ROOM_BOUNDS[0] <= pt[0] <= ROOM_BOUNDS[1] and
            ROOM_BOUNDS[2] <= pt[1] <= ROOM_BOUNDS[3])


def _clamp(pt):
    return [
        float(np.clip(pt[0], ROOM_BOUNDS[0], ROOM_BOUNDS[1])),
        float(np.clip(pt[1], ROOM_BOUNDS[2], ROOM_BOUNDS[3])),
    ]


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class ActionPlanner:
    """
    High-level waypoint planner with improved bypass logic.
    """

    def __init__(self):
        self.current_plan = []
        self.plan_index = 0
        self.goal = None
        self.obstacles = []
        self._table_pos = None
        self._table_size = None
        self._table_yaw = None

    # ------------------------------------------------------------------ #

    def create_plan(self, start_pos, goal_pos,
                    obstacles=None,
                    table_pos=None,
                    table_size=None,
                    table_yaw=None):

        if obstacles is None:
            obstacles = []

        self.goal = goal_pos
        self.obstacles = obstacles
        self._table_pos = table_pos
        self._table_size = table_size
        self._table_yaw = table_yaw

        if self._is_path_clear(start_pos, goal_pos):
            plan = [_clamp(goal_pos)]
        else:
            plan = self._plan_around_obstacles(start_pos, goal_pos)

        self.current_plan = plan + [goal_pos]  # ensure final goal is included
        self.plan_index = 0
        return plan

    # ------------------------------------------------------------------ #

    def get_next_waypoint(self):
        if self.plan_index < len(self.current_plan):
            return self.current_plan[self.plan_index]
        return None

    def advance_waypoint(self):
        print(f"[ActionPlanner] Advancing waypoint: {self.plan_index} -> {self.plan_index + 1}")
        if self.plan_index < len(self.current_plan):
            self.plan_index += 1
            return True
        return False

    def is_plan_complete(self):
        return self.plan_index >= len(self.current_plan)

    # ------------------------------------------------------------------ #

    def _is_path_clear(self, start, goal):
        if not self.obstacles:
            return True

        steps = 20
        for i in range(steps + 1):
            t = i / steps
            pt = [
                start[0] + t * (goal[0] - start[0]),
                start[1] + t * (goal[1] - start[1]),
            ]
            for obs in self.obstacles:
                if _distance(pt, obs) < _CLEARANCE_SMALL:
                    return False
        return True

    # ------------------------------------------------------------------ #
    # FIXED WAYPOINT LOGIC
    # ------------------------------------------------------------------ #

    def _plan_around_obstacles(self, start, goal):

        direction = np.array(goal) - np.array(start)
        norm = np.linalg.norm(direction)

        if norm < 1e-6:
            return [_clamp(goal)]

        direction = direction / norm
        perp = np.array([-direction[1], direction[0]])

        # generate left & right bypass candidates
        wp_left = np.array(start) + direction * 1.0 + perp * _TABLE_BYPASS_MARGIN
        wp_right = np.array(start) + direction * 1.0 - perp * _TABLE_BYPASS_MARGIN

        candidates = []

        for wp in [wp_left, wp_right]:
            wp = _clamp(wp.tolist())

            if not _in_bounds(wp):
                continue

            # Ensure both path segments are clear
            if not self._is_path_clear(start, wp):
                continue
            if not self._is_path_clear(wp, goal):
                continue

            candidates.append(wp)

        # If valid candidates exist → choose shortest total path
        if candidates:

            def cost(wp):
                total = _distance(start, wp) + _distance(wp, goal)

                # Penalize waypoint behind robot
                forward_vec = np.array(goal) - np.array(start)
                wp_vec = np.array(wp) - np.array(start)
                if np.dot(wp_vec, forward_vec) < 0:
                    total += 2.0  # strong penalty

                return total

            best = min(candidates, key=cost)
            return [best, _clamp(goal)]

        # fallback: just move slightly forward
        fallback = _clamp((np.array(start) + direction * 0.8).tolist())
        return [fallback, _clamp(goal)]
class GraspPlanner:
    def __init__(self, table_z=0.625, obj_height=0.12):
        # tune these two constants to your sim
        self.table_z = float(table_z)
        self.obj_height = float(obj_height)

        self.pregrasp_dz = 0.18     # above object
        self.grasp_clearance = 0.02 # don't penetrate

    def plan_grasp(self, object_pos, object_type='cylinder'):

        # object_pos may be [x,y] or [x,y,z]
        x = float(object_pos[0])
        y = float(object_pos[1])

        # Prefer provided z if valid, else derive from table height
        if len(object_pos) >= 3 and object_pos[2] is not None and np.isfinite(object_pos[2]):
            z_obj = float(object_pos[2])
        else:
            z_obj = self.table_z + 0.5 * self.obj_height  # approximate center/top

        grasp_z    = max(self.table_z + self.grasp_clearance, z_obj)
        approach_z = grasp_z + self.pregrasp_dz

        grasp_plan = {}
        grasp_plan['approach_pos'] = [x, y, approach_z]
        grasp_plan['grasp_pos']    = [x, y, grasp_z]
        grasp_plan['orientation']  = [0, np.pi / 2, 0]
        return grasp_plan
    
    def check_reachability(self, robot_pos, object_pos, max_reach=1.0):
        dist = float(np.hypot(object_pos[0] - robot_pos[0],
                            object_pos[1] - robot_pos[1]))
        return dist <= max_reach


# ─────────────────────────── singletons ─────────────────────────────────

_action_planner = None
_grasp_planner  = None


def get_action_planner():
    """Get or create the global action planner instance."""
    global _action_planner
    if _action_planner is None:
        _action_planner = ActionPlanner()
    return _action_planner


def get_grasp_planner():
    """Get or create the global grasp planner instance."""
    global _grasp_planner
    if _grasp_planner is None:
        _grasp_planner = GraspPlanner()
    return _grasp_planner
