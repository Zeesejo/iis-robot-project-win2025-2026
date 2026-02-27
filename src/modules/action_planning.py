"""
Module 7: Action Planning
High-level task sequencing and coordination for the robot mission.
Works with the FSM to manage the Search -> Navigate -> Grasp sequence.
"""

import numpy as np

# Room bounds (x_min, x_max, y_min, y_max) – walls are at ±5 m
# Use a 0.5 m inset so waypoints never land inside a wall.
ROOM_BOUNDS = (-4.5, 4.5, -4.5, 4.5)

# Clearance radii
_CLEARANCE_TABLE   = 2.0   # table is the big obstacle
_CLEARANCE_SMALL   = 1.0   # other obstacles


def _clearance_for(obs, table_pos=None):
    """Return the clearance radius to use for a given obstacle position."""
    if table_pos is not None:
        if np.hypot(obs[0] - table_pos[0], obs[1] - table_pos[1]) < 0.5:
            return _CLEARANCE_TABLE
    return _CLEARANCE_SMALL


def _in_bounds(pt):
    """True if point is inside room bounds."""
    return (ROOM_BOUNDS[0] <= pt[0] <= ROOM_BOUNDS[1] and
            ROOM_BOUNDS[2] <= pt[1] <= ROOM_BOUNDS[3])


def _clamp(pt):
    """Clamp a 2-D point to room bounds."""
    return [
        float(np.clip(pt[0], ROOM_BOUNDS[0], ROOM_BOUNDS[1])),
        float(np.clip(pt[1], ROOM_BOUNDS[2], ROOM_BOUNDS[3])),
    ]


class ActionPlanner:
    """
    High-level action planner that coordinates the robot's mission.
    Plans sequences of actions based on current state and goals.
    """

    def __init__(self):
        self.current_plan = []
        self.plan_index   = 0
        self.goal         = None
        self.obstacles    = []
        self._table_pos   = None   # cached so clearance is table-aware

    # ------------------------------------------------------------------ #

    def create_plan(self, start_pos, goal_pos, obstacles=None):
        """
        Create a high-level action plan to reach the goal.

        Args:
            start_pos: Current robot position [x, y]
            goal_pos:  Target position [x, y]
            obstacles: List of obstacle positions [[x,y], ...]

        Returns:
            List of waypoints to follow
        """
        if obstacles is None:
            obstacles = []

        self.goal      = goal_pos
        self.obstacles = obstacles

        # Guess which obstacle is the table (largest clearance needed)
        # – heuristic: obstacle farthest from origin is likely the table.
        if obstacles:
            self._table_pos = max(obstacles, key=lambda o: np.hypot(o[0], o[1]))
        else:
            self._table_pos = None

        plan = []
        if self._is_path_clear(start_pos, goal_pos):
            plan.append(_clamp(goal_pos))
        else:
            waypoints = self._plan_around_obstacles(start_pos, goal_pos)
            plan.extend(waypoints)

        self.current_plan = plan
        self.plan_index   = 0
        return plan

    def get_next_waypoint(self):
        """Get the next waypoint in the current plan."""
        if self.plan_index < len(self.current_plan):
            return self.current_plan[self.plan_index]
        return None

    def advance_waypoint(self):
        """Move to the next waypoint in the plan."""
        if self.plan_index < len(self.current_plan):
            self.plan_index += 1
            return True
        return False

    def is_plan_complete(self):
        """Check if all waypoints have been reached."""
        return self.plan_index >= len(self.current_plan)

    # ------------------------------------------------------------------ #

    def _is_path_clear(self, start, goal):
        """
        Check if direct path from start to goal is clear of obstacles.
        Uses per-obstacle clearance radii.
        """
        if not self.obstacles:
            return True

        steps = 10
        for i in range(steps + 1):
            t  = i / steps
            pt = [
                start[0] + t * (goal[0] - start[0]),
                start[1] + t * (goal[1] - start[1]),
            ]
            for obs in self.obstacles:
                clearance = _clearance_for(obs, self._table_pos)
                if np.hypot(pt[0] - obs[0], pt[1] - obs[1]) < clearance:
                    return False
        return True

    def _plan_around_obstacles(self, start, goal):
        """
        Create waypoints to navigate around obstacles.

        Strategy:
        1. Find obstacle closest to the midpoint of the direct path.
        2. Try BOTH perpendicular directions (left & right of the path).
        3. Pick the direction whose bypass waypoint is inside room bounds;
           if both are, pick the one farther from the wall.
        4. If neither is in bounds, scale the offset down until it fits
           (clamped fallback).
        5. Append the original goal at the end.
        """
        waypoints = []

        mid = [(start[0] + goal[0]) / 2.0,
               (start[1] + goal[1]) / 2.0]

        # Find obstacle closest to midpoint
        closest_obs  = None
        min_dist     = float('inf')
        for obs in self.obstacles:
            d = np.hypot(mid[0] - obs[0], mid[1] - obs[1])
            if d < min_dist:
                min_dist    = d
                closest_obs = obs

        if closest_obs:
            clearance = _clearance_for(closest_obs, self._table_pos)
            # Unit vector along the path
            dx, dy   = goal[0] - start[0], goal[1] - start[1]
            path_len = np.hypot(dx, dy)
            if path_len > 1e-6:
                dx /= path_len
                dy /= path_len

            # Two perpendicular unit directions
            perp_left  = [-dy,  dx]   # rotate +90°
            perp_right = [ dy, -dx]   # rotate -90°

            wp_left  = [
                closest_obs[0] + perp_left[0]  * clearance,
                closest_obs[1] + perp_left[1]  * clearance,
            ]
            wp_right = [
                closest_obs[0] + perp_right[0] * clearance,
                closest_obs[1] + perp_right[1] * clearance,
            ]

            left_ok  = _in_bounds(wp_left)
            right_ok = _in_bounds(wp_right)

            if left_ok and right_ok:
                # Both fine: prefer the one with more margin from the nearest wall
                def _wall_margin(pt):
                    return min(
                        pt[0] - ROOM_BOUNDS[0],
                        ROOM_BOUNDS[1] - pt[0],
                        pt[1] - ROOM_BOUNDS[2],
                        ROOM_BOUNDS[3] - pt[1],
                    )
                chosen = wp_left if _wall_margin(wp_left) >= _wall_margin(wp_right) else wp_right
            elif left_ok:
                chosen = wp_left
            elif right_ok:
                chosen = wp_right
            else:
                # Neither is in bounds – try the direction that points more
                # toward the centre (0, 0) and clamp.
                cx_l = -closest_obs[0] * perp_left[0]  + (-closest_obs[1]) * perp_left[1]
                cx_r = -closest_obs[0] * perp_right[0] + (-closest_obs[1]) * perp_right[1]
                raw  = wp_left if cx_l >= cx_r else wp_right
                chosen = _clamp(raw)

            waypoints.append(chosen)

        waypoints.append(_clamp(goal))
        return waypoints

    # ------------------------------------------------------------------ #

    def replan(self, current_pos, detected_obstacles):
        """
        Replan the path if new obstacles are detected.

        Args:
            current_pos:          Current robot position
            detected_obstacles:   Newly detected obstacles
        """
        if self.goal is None:
            return []
        # De-duplicate before merging
        for obs in detected_obstacles:
            if not any(np.hypot(obs[0]-o[0], obs[1]-o[1]) < 0.3
                       for o in self.obstacles):
                self.obstacles.append(obs)
        return self.create_plan(current_pos, self.goal, self.obstacles)


# ─────────────────────────── Grasp Planner ─────────────────────────────

class GraspPlanner:
    """
    Plans grasp attempts for the target object.
    Uses simple heuristics for approach and grasp configuration.
    """

    def __init__(self):
        self.grasp_offset = [0, 0, 0.20]  # Increased from 0.15 to 0.20 for safer approach
        self.grasp_height = 0.05
        # Cylinder dimensions from task spec: r=0.04m, h=0.12m
        self.cylinder_radius = 0.04
        self.cylinder_height = 0.12

    def plan_grasp(self, object_pos, object_type='cylinder'):
        """
        Plan a grasp for the target object.

        Args:
            object_pos:  Position [x, y, z] of object (center position)
            object_type: Type of object ('cylinder', 'cube', etc.)

        Returns:
            Dict with 'approach_pos' and 'grasp_pos'
        """
        import numpy as _np
        
        # Calculate grasp position
        # The object_pos is the center of the cylinder (z is at center height)
        # Cylinder sits on table at z=0.625, cylinder height=0.12, so center=0.685
        # We need to grasp from the side at the middle height of the cylinder
        
        # Use the detected object position height
        # If detected height seems wrong, use the known cylinder center height
        detected_z = object_pos[2]
        
        # The cylinder center is at TABLE_SURFACE_Z (0.625) + CYLINDER_HEIGHT/2 (0.06) = 0.685
        # But the gripper needs to go below center to get a good grasp
        # Gripper fingers are at gripper_base + some offset
        # We'll position the gripper slightly below the cylinder center
        grasp_z = max(detected_z - 0.02, 0.65)  # Slightly below center, min 0.65m (above table)
        
        # Approach position: higher up to avoid table collision
        # The table surface is at z=0.625, so we need to approach from above
        approach_z = grasp_z + 0.15  # 15cm above grasp point
        
        grasp_plan = {}
        # Approach position: above the grasp position
        grasp_plan['approach_pos'] = [
            object_pos[0],
            object_pos[1],
            approach_z,
        ]
        # Grasp position: at the middle height of the cylinder for picking it up
        grasp_plan['grasp_pos'] = [
            object_pos[0],
            object_pos[1],
            grasp_z,  # Slightly below center for stable grasp
        ]
        # Place position: above the original position to place the cylinder back on the table
        # Lift it up slightly above the original position, then release
        place_z = max(detected_z + 0.15, 0.75)  # Lift 15cm above, min 0.75m
        grasp_plan['place_pos'] = [
            object_pos[0],
            object_pos[1],
            place_z,  # Above the original position to release
        ]
        # Orientation: gripper parallel to ground (horizontal grasp)
        # Gripper should be horizontal to grasp the cylinder from the side
        grasp_plan['orientation'] = [0, _np.pi / 2, 0]
        return grasp_plan

    def check_reachability(self, robot_pos, object_pos, max_reach=1.0):
        """
        Check if object is within robot's reach.

        Args:
            robot_pos:  Robot base position
            object_pos: Target object position
            max_reach:  Maximum arm reach distance

        Returns:
            bool: True if reachable
        """
        dist = np.hypot(object_pos[0] - robot_pos[0],
                        object_pos[1] - robot_pos[1])
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
