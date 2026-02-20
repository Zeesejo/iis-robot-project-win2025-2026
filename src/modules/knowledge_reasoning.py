"""
Module 8: Knowledge Representation & Reasoning
Uses Prolog (PySwip) with Dynamic_KB.pl to store and query semantic knowledge.
Provides a Python interface to the Prolog knowledge base with dict fallback.
"""

import os

try:
    from pyswip import Prolog
    PROLOG_AVAILABLE = True
except ImportError:
    print("[Knowledge] PySwip not installed. Knowledge reasoning disabled.")
    PROLOG_AVAILABLE = False


class KnowledgeBase:
    """
    Prolog-based knowledge representation for the robot's world model.
    Loads Dynamic_KB.pl for structured knowledge about objects, sensors,
    robot links, spatial reasoning, and affordances.
    Falls back to a Python dict if Prolog is unavailable.
    """
    
    def __init__(self):
        self.prolog = None
        self.facts = {}  # Always initialize dict fallback
        
        if PROLOG_AVAILABLE:
            try:
                self.prolog = Prolog()
                self._load_dynamic_kb()
                print("[Knowledge] Prolog KB initialized with Dynamic_KB.pl")
            except Exception as e:
                print(f"[Knowledge] Prolog initialization failed: {e}")
                print("[Knowledge] Falling back to dict-based knowledge store")
                self.prolog = None
        else:
            print("[Knowledge] Running without Prolog - using dict fallback")
    
    def _load_dynamic_kb(self):
        """Load the Dynamic_KB.pl Prolog knowledge base file"""
        if not self.prolog:
            return
        kb_path = os.path.join(os.path.dirname(__file__), "Dynamic_KB.pl")
        if os.path.exists(kb_path):
            try:
                self.prolog.consult(kb_path)
                print(f"[Knowledge] Loaded Dynamic_KB.pl from {kb_path}")
            except Exception as e:
                print(f"[Knowledge] Warning: Could not load Dynamic_KB.pl: {e}")
                self._initialize_facts_fallback()
        else:
            print(f"[Knowledge] Dynamic_KB.pl not found at {kb_path}, using inline facts")
            self._initialize_facts_fallback()
    
    def _initialize_facts_fallback(self):
        """Initialize basic Prolog facts inline if Dynamic_KB.pl is missing"""
        if not self.prolog:
            return
        # Define object types
        self.prolog.assertz("object_type(table, furniture)")
        self.prolog.assertz("object_type(target, graspable)")
        self.prolog.assertz("object_type(obstacle, static)")
        # Define colors
        self.prolog.assertz("color(target, red)")
        self.prolog.assertz("color(table, brown)")
        # Define properties
        self.prolog.assertz("graspable(target)")
        self.prolog.assertz("fixed(table)")
        self.prolog.assertz("fixed(obstacle)")
        # Define affordances
        self.prolog.assertz("affords(target, grasp)")
        self.prolog.assertz("affords(table, support)")
        self.prolog.assertz("affords(obstacle, avoid)")
        # Rules for reasoning
        self.prolog.assertz("can_grasp_object(X) :- graspable(X)")
        self.prolog.assertz("should_avoid(X) :- object_type(X, static)")
        self.prolog.assertz("is_goal(X) :- color(X, red)")
    
    # ==================== Position Management ====================
    
    def add_position(self, object_name, x, y, z):
        """Add or update object position in knowledge base"""
        if self.prolog:
            try:
                # Use update_position from Dynamic_KB.pl (retractall + assert)
                list(self.prolog.query(
                    f"update_position({object_name}, {x}, {y}, {z})"
                ))
            except Exception:
                # Fallback to manual retract/assert
                try:
                    list(self.prolog.query(
                        f"retractall(position({object_name}, _, _, _))"
                    ))
                except Exception:
                    pass
                try:
                    self.prolog.assertz(
                        f"position({object_name}, {x}, {y}, {z})"
                    )
                except Exception:
                    pass
        # Always update dict fallback too
        self.facts[f'position_{object_name}'] = (x, y, z)
    
    def query_position(self, object_name):
        """Get position of an object"""
        if self.prolog:
            try:
                results = list(self.prolog.query(
                    f"position({object_name}, X, Y, Z)"
                ))
                if results:
                    r = results[0]
                    return (float(r['X']), float(r['Y']), float(r['Z']))
            except Exception:
                pass
        return self.facts.get(f'position_{object_name}')
    
    # ==================== Object Management ====================
    
    def add_detected_object(self, object_id, object_type, color, position):
        """Add a detected object to the knowledge base"""
        if self.prolog:
            # Only assert object_type if not already known
            try:
                existing = list(self.prolog.query(f"object_type({object_id}, _)"))
                if not existing:
                    self.prolog.assertz(f"object_type({object_id}, {object_type})")
            except Exception:
                pass
            # Only assert color if not already known
            try:
                existing = list(self.prolog.query(f"color({object_id}, _)"))
                if not existing:
                    self.prolog.assertz(f"color({object_id}, {color})")
            except Exception:
                pass
            if position:
                self.add_position(object_id, position[0], position[1], position[2])
        # Always update dict
        self.facts[f'object_{object_id}'] = {
            'type': object_type,
            'color': color,
            'position': position
        }
    
    def query_objects_by_color(self, color):
        """Find all objects with specified color"""
        if self.prolog:
            try:
                results = list(self.prolog.query(f"color(X, {color})"))
                return [str(r['X']) for r in results]
            except Exception:
                pass
        # Fallback: search dict
        return [k.replace('object_', '') for k, v in self.facts.items() 
                if 'object_' in k and isinstance(v, dict) and v.get('color') == color]
    
    def query_graspable(self):
        """Find all graspable/pickable objects"""
        if self.prolog:
            try:
                results = list(self.prolog.query("can_pick(X)"))
                if results:
                    return [str(r['X']) for r in results]
                # Fallback to can_grasp_object rule
                results = list(self.prolog.query("can_grasp_object(X)"))
                return [str(r['X']) for r in results]
            except Exception:
                pass
        return ['target']  # Fallback
    
    def is_goal_object(self, object_id):
        """Check if object is the goal (red target)"""
        if self.prolog:
            try:
                results = list(self.prolog.query(f"is_goal({object_id})"))
                return len(results) > 0
            except Exception:
                pass
        obj = self.facts.get(f'object_{object_id}')
        return obj and isinstance(obj, dict) and obj.get('color') == 'red'
    
    def should_avoid_object(self, object_id):
        """Check if object should be avoided (obstacle)"""
        if self.prolog:
            try:
                results = list(self.prolog.query(f"should_avoid({object_id})"))
                return len(results) > 0
            except Exception:
                pass
        obj = self.facts.get(f'object_{object_id}')
        return obj and isinstance(obj, dict) and obj.get('type') == 'static'
    
    # ==================== M8 Knowledge Representation Methods ====================
    
    def objects(self):
        """Return a list of all objects in the world"""
        if self.prolog:
            try:
                return [str(obj['X']) for obj in self.prolog.query("object(X)")]
            except Exception:
                pass
        return list(set(
            k.replace('object_', '') for k in self.facts
            if k.startswith('object_')
        ))
    
    def colors(self):
        """Return a dictionary mapping object -> color"""
        if self.prolog:
            try:
                return {str(obj['X']): str(obj['C'])
                        for obj in self.prolog.query("color(X, C)")}
            except Exception:
                pass
        return {k.replace('object_', ''): v.get('color', 'unknown')
                for k, v in self.facts.items()
                if k.startswith('object_') and isinstance(v, dict)}
    
    def fixed_objects(self):
        """Return a list of all fixed objects"""
        if self.prolog:
            try:
                return [str(obj['X']) for obj in self.prolog.query("is_fixed(X)")]
            except Exception:
                pass
        return []
    
    def movable_objects(self):
        """Return a list of all movable objects"""
        if self.prolog:
            try:
                return [str(obj['X']) for obj in self.prolog.query("is_movable(X)")]
            except Exception:
                pass
        return ['target']
    
    def pickable_objects(self):
        """Return a list of objects that can be picked"""
        if self.prolog:
            try:
                return [str(obj['X']) for obj in self.prolog.query("can_pick(X)")]
            except Exception:
                pass
        return ['target']
    
    def robot_links(self):
        """Return a list of all robot links"""
        if self.prolog:
            try:
                return [str(obj['X']) for obj in self.prolog.query("robot_link(X)")]
            except Exception:
                pass
        return []
    
    def robot_capabilities(self):
        """Return a dictionary describing robot capabilities"""
        caps = {'sensors': [], 'can_move': True, 'has_arm': True}
        if self.prolog:
            try:
                for obj in self.prolog.query("has_sensor(robot, S)"):
                    caps['sensors'].append(str(obj['S']))
            except Exception:
                pass
        return caps
    
    def sensors(self):
        """Return a list of all sensors"""
        if self.prolog:
            try:
                return [str(s['S']) for s in self.prolog.query("sensor(S, T)")]
            except Exception:
                pass
        return []
    
    def sensor_types(self):
        """Return a dictionary mapping sensor -> type"""
        if self.prolog:
            try:
                return {str(s['S']): str(s['T'])
                        for s in self.prolog.query("sensor(S, T)")}
            except Exception:
                pass
        return {}
    
    def detectable_objects(self):
        """Return a list of objects currently detectable by robot sensors"""
        if self.prolog:
            try:
                objs = [str(obj['Obj'])
                        for obj in self.prolog.query("detectable_objects(Obj)")]
                return list(set(objs))
            except Exception:
                pass
        return []
    
    def check_can_grasp(self):
        """Check if robot can currently grasp the target (Prolog reasoning)"""
        if self.prolog:
            try:
                results = list(self.prolog.query("can_grasp(target)"))
                return len(results) > 0
            except Exception:
                pass
        return False
    
    def check_success(self):
        """Check if the mission is achievable (navigate + grasp)"""
        if self.prolog:
            try:
                return bool(list(self.prolog.query("success")))
            except Exception:
                pass
        return False
    
    def get_robot_path_history(self):
        """Return the robot path history from Prolog KB"""
        if self.prolog:
            try:
                return [(float(x['X']), float(x['Y']), float(x['Z']))
                        for x in self.prolog.query(
                            "robot_position_history(X,Y,Z)"
                        )]
            except Exception:
                pass
        return []


# Global knowledge base instance
_kb_instance = None

def get_knowledge_base():
    """Get or create the global knowledge base instance"""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance

