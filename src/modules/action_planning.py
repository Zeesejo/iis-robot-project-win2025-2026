"""
Module 7: Action Planning
High-level task sequencing and coordination for the robot mission.
Works with the FSM to manage the Search -> Navigate -> Grasp sequence.
"""

import numpy as np


class ActionPlanner:
    """
    High-level action planner that coordinates the robot's mission.
    Plans sequences of actions based on current state and goals.
    """
    
    def __init__(self):
        self.current_plan = []
        self.plan_index = 0
        self.goal = None
        self.obstacles = []
        
    def create_plan(self, start_pos, goal_pos, obstacles=None):
        """
        Create a high-level action plan to reach the goal.
        
        Args:
            start_pos: Current robot position [x, y]
            goal_pos: Target position [x, y]
            obstacles: List of obstacle positions
            
        Returns:
            List of waypoints to follow
        """
        if obstacles is None:
            obstacles = []
            
        self.goal = goal_pos
        self.obstacles = obstacles
        
        # Simple straight-line plan (can be enhanced with path planning)
        plan = []
        
        # Check if direct path is clear
        if self._is_path_clear(start_pos, goal_pos):
            plan.append(goal_pos)
        else:
            # Create waypoints to avoid obstacles
            waypoints = self._plan_around_obstacles(start_pos, goal_pos)
            plan.extend(waypoints)
        
        self.current_plan = plan
        self.plan_index = 0
        return plan
    
    def get_next_waypoint(self):
        """Get the next waypoint in the current plan"""
        if self.plan_index < len(self.current_plan):
            return self.current_plan[self.plan_index]
        return None
    
    def advance_waypoint(self):
        """Move to the next waypoint in the plan"""
        if self.plan_index < len(self.current_plan):
            self.plan_index += 1
            return True
        return False
    
    def is_plan_complete(self):
        """Check if all waypoints have been reached"""
        return self.plan_index >= len(self.current_plan)
    
    def _is_path_clear(self, start, goal):
        """
        Check if direct path from start to goal is clear of obstacles.
        Uses simple distance check to nearby obstacles.
        """
        if not self.obstacles:
            return True
            
        # Sample points along the line
        steps = 10
        for i in range(steps + 1):
            t = i / steps
            point = [
                start[0] + t * (goal[0] - start[0]),
                start[1] + t * (goal[1] - start[1])
            ]
            
            # Check distance to each obstacle
            for obs in self.obstacles:
                dist = np.hypot(point[0] - obs[0], point[1] - obs[1])
                if dist < 1.0:  # Obstacle radius + safety margin (covers table 1.5x0.8)
                    return False
        
        return True
    
    def _plan_around_obstacles(self, start, goal):
        """
        Create waypoints to navigate around obstacles.
        Simple implementation: goes around the midpoint.
        """
        waypoints = []
        
        # Find obstacle closest to direct path
        mid = [(start[0] + goal[0])/2, (start[1] + goal[1])/2]
        
        closest_obs = None
        min_dist = float('inf')
        for obs in self.obstacles:
            dist = np.hypot(mid[0] - obs[0], mid[1] - obs[1])
            if dist < min_dist:
                min_dist = dist
                closest_obs = obs
        
        if closest_obs:
            # Create waypoint that goes around the obstacle
            # Choose perpendicular direction
            dx = goal[0] - start[0]
            dy = goal[1] - start[1]
            perp_x = -dy
            perp_y = dx
            length = np.hypot(perp_x, perp_y)
            if length > 0:
                perp_x /= length
                perp_y /= length
            
            # Waypoint offset from obstacle
            offset = 1.0
            waypoint = [
                closest_obs[0] + perp_x * offset,
                closest_obs[1] + perp_y * offset
            ]
            waypoints.append(waypoint)
        
        waypoints.append(goal)
        return waypoints
    
    def replan(self, current_pos, detected_obstacles):
        """
        Replan the path if new obstacles are detected.
        
        Args:
            current_pos: Current robot position
            detected_obstacles: Newly detected obstacles
        """
        if self.goal is None:
            return []
        
        # Update obstacles list
        self.obstacles.extend(detected_obstacles)
        
        # Create new plan from current position
        return self.create_plan(current_pos, self.goal, self.obstacles)


class GraspPlanner:
    """
    Plans grasp attempts for the target object.
    Uses simple heuristics for approach and grasp configuration.
    """
    
    def __init__(self):
        self.grasp_offset = [0, 0, 0.15]  # Approach from above
        self.grasp_height = 0.05  # Final grasp height above table
        
    def plan_grasp(self, object_pos, object_type='cylinder'):
        """
        Plan a grasp for the target object.
        
        Args:
            object_pos: Position [x, y, z] of object
            object_type: Type of object ('cylinder', 'cube', etc.)
            
        Returns:
            Dict with 'approach_pos' and 'grasp_pos'
        """
        grasp_plan = {}
        
        # Approach position (above the object)
        grasp_plan['approach_pos'] = [
            object_pos[0],
            object_pos[1],
            object_pos[2] + self.grasp_offset[2]
        ]
        
        # Final grasp position
        grasp_plan['grasp_pos'] = [
            object_pos[0],
            object_pos[1],
            object_pos[2]
        ]
        
        # Grasp orientation (pointing down for top grasp)
        grasp_plan['orientation'] = [0, np.pi/2, 0]  # Euler angles
        
        return grasp_plan
    
    def check_reachability(self, robot_pos, object_pos, max_reach=1.0):
        """
        Check if object is within robot's reach.
        
        Args:
            robot_pos: Robot base position
            object_pos: Target object position  
            max_reach: Maximum arm reach distance
            
        Returns:
            bool: True if reachable
        """
        dist = np.hypot(object_pos[0] - robot_pos[0], 
                       object_pos[1] - robot_pos[1])
        return dist <= max_reach


# Global planner instances
_action_planner = None
_grasp_planner = None

def get_action_planner():
    """Get or create the global action planner instance"""
    global _action_planner
    if _action_planner is None:
        _action_planner = ActionPlanner()
    return _action_planner

def get_grasp_planner():
    """Get or create the global grasp planner instance"""
    global _grasp_planner
    if _grasp_planner is None:
        _grasp_planner = GraspPlanner()
    return _grasp_planner

