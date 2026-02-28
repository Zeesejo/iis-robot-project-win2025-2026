"""
Module 7: Action Planning
High-level task sequencing and coordination for the robot mission.
Works with the FSM to manage the Search -> Navigate -> Approach -> Grasp sequence.

Fixes:
- Multi-obstacle path planning (iterative, not just midpoint)
- Proper grasp standoff computation using table geometry
- Room bounds respected at all times
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# Room bounds (inset 0.5 m from walls at ±5 m)
ROOM_BOUNDS = (-4.5, 4.5, -4.5, 4.5)

_CLEARANCE_TABLE = 1.8   # keep away from table
_CLEARANCE_SMALL = 0.8   # keep away from box obstacles


def _clearance_for(obs, table_pos=None):
    if table_pos is not None:
        if np.hypot(obs[0] - table_pos[0], obs[1] - table_pos[1]) < 0.5:
            return _CLEARANCE_TABLE
    return _CLEARANCE_SMALL


def _in_bounds(pt):
    return (ROOM_BOUNDS[0] <= pt[0] <= ROOM_BOUNDS[1] and
            ROOM_BOUNDS[2] <= pt[1] <= ROOM_BOUNDS[3])


def _clamp(pt):
    return [
        float(np.clip(pt[0], ROOM_BOUNDS[0], ROOM_BOUNDS[1])),
        float(np.clip(pt[1], ROOM_BOUNDS[2], ROOM_BOUNDS[3])),
    ]


def _wall_margin(pt):
    return min(
        pt[0] - ROOM_BOUNDS[0], ROOM_BOUNDS[1] - pt[0],
        pt[1] - ROOM_BOUNDS[2], ROOM_BOUNDS[3] - pt[1],
    )


class ActionPlanner:
    """
    High-level action planner using multi-obstacle iterative path planning.
    Plans sequences of waypoints from start to goal avoiding all obstacles.
    """

    def __init__(self):
        self.current_plan = []
        self.plan_index   = 0
        self.goal         = None
        self.obstacles    = []
        self._table_pos   = None

    def create_plan(self, start_pos, goal_pos, obstacles=None):
        """
        Create a collision-free waypoint plan from start to goal.
        Uses iterative per-segment obstacle checking so ALL obstacles
        on the path are bypassed, not just the one nearest the midpoint.
        """
        if obstacles is None:
            obstacles = []

        self.goal      = goal_pos
        self.obstacles = obstacles

        # Heuristic: table is the obstacle farthest from origin
        if obstacles:
            self._table_pos = max(obstacles, key=lambda o: np.hypot(o[0], o[1]))
        else:
            self._table_pos = None

        waypoints = self._plan_path(list(start_pos[:2]), list(goal_pos[:2]),
                                    max_iterations=20)
        self.current_plan = waypoints
        self.plan_index   = 0
        logger.info("[ActionPlanner] Plan: %d waypoints", len(waypoints))
        return waypoints

    def _plan_path(self, start, goal, max_iterations=20):
        """
        Iteratively insert bypass waypoints until no segment collides.
        Returns list of waypoints ending with the (clamped) goal.
        """
        path = [list(start), _clamp(goal)]

        for _ in range(max_iterations):
            new_path   = [path[0]]
            modified   = False
            for i in range(len(path) - 1):
                seg_start = path[i]
                seg_end   = path[i + 1]
                blocker   = self._first_blocker(seg_start, seg_end)
                if blocker is not None:
                    wp = self._bypass_waypoint(seg_start, seg_end, blocker)
                    new_path.append(wp)
                    modified = True
                new_path.append(seg_end)
            path = new_path
            if not modified:
                break

        # Remove the start position; return rest
        return path[1:]

    def _first_blocker(self, start, end):
        """Return the first obstacle that blocks the segment start->end, or None."""
        steps = 20
        for i in range(steps + 1):
            t  = i / steps
            pt = [
                start[0] + t * (end[0] - start[0]),
                start[1] + t * (end[1] - start[1]),
            ]
            for obs in self.obstacles:
                clearance = _clearance_for(obs, self._table_pos)
                if np.hypot(pt[0] - obs[0], pt[1] - obs[1]) < clearance:
                    return obs
        return None

    def _bypass_waypoint(self, start, end, blocker):
        """Compute a single waypoint that bypasses blocker between start and end."""
        clearance = _clearance_for(blocker, self._table_pos)

        dx, dy   = end[0] - start[0], end[1] - start[1]
        path_len = np.hypot(dx, dy)
        if path_len > 1e-6:
            dx /= path_len
            dy /= path_len
        else:
            dx, dy = 1.0, 0.0

        # Two perpendicular bypass directions
        perp = [[-dy, dx], [dy, -dx]]
        candidates = []
        for px, py in perp:
            wp = [blocker[0] + px * clearance * 1.1,
                  blocker[1] + py * clearance * 1.1]
            if _in_bounds(wp):
                candidates.append((wp, _wall_margin(wp)))
            else:
                clamped = _clamp(wp)
                candidates.append((clamped, _wall_margin(clamped)))

        # Prefer the candidate with best wall margin
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]

    # ---- waypoint iteration ----

    def get_next_waypoint(self):
        if self.plan_index < len(self.current_plan):
            return self.current_plan[self.plan_index]
        return None

    def advance_waypoint(self):
        if self.plan_index < len(self.current_plan):
            self.plan_index += 1
            return True
        return False

    def is_plan_complete(self):
        return self.plan_index >= len(self.current_plan)

    def replan(self, current_pos, detected_obstacles):
        """Re-plan when new obstacles are detected."""
        if self.goal is None:
            return []
        for obs in detected_obstacles:
            if not any(np.hypot(obs[0]-o[0], obs[1]-o[1]) < 0.3
                       for o in self.obstacles):
                self.obstacles.append(obs)
        return self.create_plan(current_pos, self.goal, self.obstacles)


# ─────────────────────────── Grasp Planner ───────────────────────────

class GraspPlanner:
    """
    Plans grasp attempts for the target cylinder.
    Table surface z=0.625, cylinder h=0.12 -> center z=0.685.
    """

    TABLE_SURFACE_Z   = 0.625
    CYLINDER_H        = 0.12
    CYLINDER_CENTER_Z = TABLE_SURFACE_Z + CYLINDER_H / 2.0   # 0.685

    def plan_grasp(self, object_pos, object_type='cylinder'):
        """
        Returns dict with approach_pos, grasp_pos, place_pos, orientation.
        All heights are in world-frame Z (meters).
        """
        # Use detected height if plausible, otherwise fall back to known value
        raw_z  = object_pos[2] if len(object_pos) > 2 else self.CYLINDER_CENTER_Z
        grasp_z = float(np.clip(raw_z - 0.02, self.TABLE_SURFACE_Z + 0.02, 0.75))

        approach_z = grasp_z + 0.18   # descend from 18 cm above
        place_z    = grasp_z + 0.20   # place 20 cm above grasp point

        xy = list(object_pos[:2])
        return {
            'approach_pos': xy + [approach_z],
            'grasp_pos':    xy + [grasp_z],
            'place_pos':    xy + [place_z],
            'orientation':  [0.0, float(np.pi / 2), 0.0],   # horizontal grasp
        }

    def check_reachability(self, robot_pos, object_pos, max_reach=1.0):
        dist = np.hypot(object_pos[0] - robot_pos[0],
                        object_pos[1] - robot_pos[1])
        return dist <= max_reach


# ─────────────────────────── singletons ────────────────────────────

_action_planner = None
_grasp_planner  = None


def get_action_planner():
    global _action_planner
    if _action_planner is None:
        _action_planner = ActionPlanner()
    return _action_planner


def get_grasp_planner():
    global _grasp_planner
    if _grasp_planner is None:
        _grasp_planner = GraspPlanner()
    return _grasp_planner
