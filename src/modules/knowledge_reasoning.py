"""
M8 - Knowledge Representation & Reasoning
Uses Prolog (via PySwip) to store and query semantic world knowledge:
object properties, affordances, safety constraints, and path planning.
"""

try:
    from pyswip import Prolog
    PROLOG_AVAILABLE = True
except ImportError:
    PROLOG_AVAILABLE = False
    print("[KnowledgeBase] WARNING: PySwip not available. Running in fallback mode.")

import numpy as np


PROLOG_RULES = """
% ============================================================
% IIS Robot Project - World Knowledge Base
% ============================================================

% --- Object Properties ---
color(target, red).
color(table, brown).
color(obstacle_blue,   blue).
color(obstacle_pink,   pink).
color(obstacle_orange, orange).
color(obstacle_yellow, yellow).
color(obstacle_black,  black).

mass(target, 0.5).
mass(table, 10.0).
mass(obstacle_blue,   10.0).
mass(obstacle_pink,   10.0).
mass(obstacle_orange, 10.0).
mass(obstacle_yellow, 10.0).
mass(obstacle_black,  10.0).

is_fixed(table).
is_fixed(obstacle_blue).
is_fixed(obstacle_pink).
is_fixed(obstacle_orange).
is_fixed(obstacle_yellow).
is_fixed(obstacle_black).

is_graspable(target).
is_navigable(table).

% --- Affordances ---
affordance(target, grasp).
affordance(table, navigate_to).
affordance(obstacle_blue,   avoid).
affordance(obstacle_pink,   avoid).
affordance(obstacle_orange, avoid).
affordance(obstacle_yellow, avoid).
affordance(obstacle_black,  avoid).

% --- Safety Rules ---
safe_to_approach(X) :- affordance(X, navigate_to), \\+ affordance(X, avoid).
should_avoid(X)     :- affordance(X, avoid).
can_grasp(X)        :- is_graspable(X), color(X, red).

% --- Path Safety Check ---
% safe_path(StartX, StartY, GoalX, GoalY) - verified externally through obstacle positions

% --- Object Identification by Color ---
identify_by_color(red, target).
identify_by_color(brown, table).
identify_by_color(blue,   obstacle_blue).
identify_by_color(pink,   obstacle_pink).
identify_by_color(orange, obstacle_orange).
identify_by_color(yellow, obstacle_yellow).
identify_by_color(black,  obstacle_black).
"""


class KnowledgeBase:
    """
    Prolog-based knowledge base for the robot's world model.
    Falls back to Python dict if PySwip is not available.
    """

    def __init__(self):
        self._obstacle_positions = {}   # name -> (x, y)
        self._table_position = None
        self._target_estimate = None

        if PROLOG_AVAILABLE:
            self.prolog = Prolog()
            self._load_rules()
        else:
            self.prolog = None

    def _load_rules(self):
        for line in PROLOG_RULES.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('%'):
                try:
                    self.prolog.assertz(line.rstrip('.'))
                except Exception:
                    pass  # Some compound facts need consult
        # Use consult via temp file for reliability
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pl',
                                         delete=False) as f:
            f.write(PROLOG_RULES)
            fname = f.name
        try:
            list(self.prolog.query(f'consult(\'{fname}\')'))
        except Exception as e:
            print(f"[KB] Prolog consult warning: {e}")
        finally:
            os.unlink(fname)

    # ---- Assert dynamic facts ----
    def assert_obstacle_position(self, name, x, y):
        """Store obstacle position."""
        self._obstacle_positions[name] = (x, y)
        if self.prolog:
            try:
                list(self.prolog.query(
                    f'retractall(position({name}, _, _))'))
                list(self.prolog.query(
                    f'assertz(position({name}, {x:.3f}, {y:.3f}))'))
            except Exception:
                pass

    def assert_table_position(self, x, y):
        self._table_position = (x, y)
        if self.prolog:
            try:
                list(self.prolog.query('retractall(position(table, _, _))'))
                list(self.prolog.query(
                    f'assertz(position(table, {x:.3f}, {y:.3f}))'))
            except Exception:
                pass

    def assert_target_estimate(self, x, y, z):
        self._target_estimate = (x, y, z)
        if self.prolog:
            try:
                list(self.prolog.query('retractall(estimated_position(target, _, _, _))'))
                list(self.prolog.query(
                    f'assertz(estimated_position(target, {x:.3f}, {y:.3f}, {z:.3f}))'))
            except Exception:
                pass

    # ---- Queries ----
    def query_can_grasp(self, obj='target'):
        if self.prolog:
            try:
                result = list(self.prolog.query(f'can_grasp({obj})'))
                return len(result) > 0
            except Exception:
                pass
        return obj == 'target'

    def query_should_avoid(self, obj):
        if self.prolog:
            try:
                result = list(self.prolog.query(f'should_avoid({obj})'))
                return len(result) > 0
            except Exception:
                pass
        return 'obstacle' in obj

    def query_affordance(self, obj, action):
        if self.prolog:
            try:
                result = list(self.prolog.query(f'affordance({obj}, {action})'))
                return len(result) > 0
            except Exception:
                pass
        return False

    def identify_object(self, color_name):
        if self.prolog:
            try:
                result = list(self.prolog.query(
                    f'identify_by_color({color_name}, X)'))
                if result:
                    return result[0]['X']
            except Exception:
                pass
        # Fallback
        mapping = {'red': 'target', 'brown': 'table',
                   'blue': 'obstacle_blue', 'pink': 'obstacle_pink',
                   'orange': 'obstacle_orange', 'yellow': 'obstacle_yellow',
                   'black': 'obstacle_black'}
        return mapping.get(color_name, 'unknown')

    def query_safe_path(self, sx, sy, gx, gy, clearance=0.5):
        """
        Check if a direct path is clear of all obstacles.
        Returns list of waypoints if safe, else None.
        """
        for name, (ox, oy) in self._obstacle_positions.items():
            # Check minimum distance from line segment to obstacle center
            d = self._point_to_segment_dist(ox, oy, sx, sy, gx, gy)
            if d < (0.4 + clearance):
                return None  # Path blocked, let motion planner handle detour
        return [(sx, sy), (gx, gy)]  # Direct path is clear

    def get_all_obstacle_positions(self):
        return list(self._obstacle_positions.values())

    def get_table_position(self):
        return self._table_position

    def get_target_estimate(self):
        return self._target_estimate

    def _point_to_segment_dist(self, px, py, ax, ay, bx, by):
        """Distance from point (px,py) to line segment (ax,ay)-(bx,by)."""
        dx, dy = bx - ax, by - ay
        if dx == dy == 0:
            return np.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))
        return np.hypot(px - (ax + t * dx), py - (ay + t * dy))

    def populate_from_scene_map(self, scene_map):
        """Initialize KB from the world_builder scene_map dict."""
        obstacle_names = ['obstacle_blue', 'obstacle_pink', 'obstacle_orange',
                          'obstacle_yellow', 'obstacle_black']
        for i, obs in enumerate(scene_map.get('obstacles', [])):
            name = obstacle_names[i] if i < len(obstacle_names) else f'obstacle_{i}'
            pos = obs['position']
            self.assert_obstacle_position(name, pos[0], pos[1])

        table_pos = scene_map.get('table', {}).get('position', [0, 0, 0])
        self.assert_table_position(table_pos[0], table_pos[1])
        print(f"[KB] Populated from scene map: {len(self._obstacle_positions)} obstacles, table at {self._table_position}")
