"""
Module 8: Knowledge Representation & Reasoning
Uses Prolog (PySwip) to store and query semantic knowledge about the world.
"""

try:
    from pyswip import Prolog
    PROLOG_AVAILABLE = True
except ImportError:
    print("[Knowledge] PySwip not installed. Knowledge reasoning disabled.")
    PROLOG_AVAILABLE = False


class KnowledgeBase:
    """
    Prolog-based knowledge representation for the robot's world model.
    Stores facts about objects, their properties, and relationships.
    """
    
    def __init__(self):
        self.prolog = None
        if PROLOG_AVAILABLE:
            self.prolog = Prolog()
            self._initialize_facts()
        else:
            print("[Knowledge] Running without Prolog - using dict fallback")
            self.facts = {}
    
    def _initialize_facts(self):
        """Initialize basic Prolog facts and rules"""
        if not self.prolog:
            return
            
        # Define object types
        self.prolog.assertz("object_type(table, furniture)")
        self.prolog.assertz("object_type(target, graspable)")
        self.prolog.assertz("object_type(obstacle, static)")
        
        # Define colors
        self.prolog.assertz("color(target, red)")
        self.prolog.assertz("color(table, brown)")
        self.prolog.assertz("color(obstacle1, blue)")
        self.prolog.assertz("color(obstacle2, pink)")
        self.prolog.assertz("color(obstacle3, orange)")
        self.prolog.assertz("color(obstacle4, yellow)")
        self.prolog.assertz("color(obstacle5, black)")
        
        # Define properties
        self.prolog.assertz("graspable(target)")
        self.prolog.assertz("fixed(table)")
        self.prolog.assertz("fixed(obstacle)")
        
        # Define affordances (what actions are possible with objects)
        self.prolog.assertz("affords(target, grasp)")
        self.prolog.assertz("affords(table, support)")
        self.prolog.assertz("affords(obstacle, avoid)")
        
        # Rules for reasoning
        self.prolog.assertz("can_grasp(X) :- graspable(X)")
        self.prolog.assertz("should_avoid(X) :- object_type(X, static)")
        self.prolog.assertz("is_goal(X) :- color(X, red)")
    
    def add_position(self, object_name, x, y, z):
        """Add or update object position in knowledge base"""
        if self.prolog:
            # Remove old position if exists
            query = f"retractall(position({object_name}, _, _, _))"
            try:
                list(self.prolog.query(query))
            except:
                pass
            # Add new position
            self.prolog.assertz(f"position({object_name}, {x}, {y}, {z})")
        else:
            self.facts[f'position_{object_name}'] = (x, y, z)
    
    def add_detected_object(self, object_id, object_type, color, position):
        """Add a detected object to the knowledge base"""
        if self.prolog:
            self.prolog.assertz(f"object_type({object_id}, {object_type})")
            self.prolog.assertz(f"color({object_id}, {color})")
            if position:
                x, y, z = position
                self.prolog.assertz(f"position({object_id}, {x}, {y}, {z})")
        else:
            self.facts[f'object_{object_id}'] = {
                'type': object_type,
                'color': color,
                'position': position
            }
    
    def query_objects_by_color(self, color):
        """Find all objects with specified color"""
        if self.prolog:
            query = f"color(X, {color})"
            try:
                results = list(self.prolog.query(query))
                return [r['X'] for r in results]
            except:
                return []
        else:
            # Fallback: search dict
            return [k.replace('object_', '') for k, v in self.facts.items() 
                    if 'object_' in k and v.get('color') == color]
    
    def query_graspable(self):
        """Find all graspable objects"""
        if self.prolog:
            query = "can_grasp(X)"
            try:
                results = list(self.prolog.query(query))
                return [r['X'] for r in results]
            except:
                return []
        else:
            return ['target']  # Fallback
    
    def query_position(self, object_name):
        """Get position of an object"""
        if self.prolog:
            query = f"position({object_name}, X, Y, Z)"
            try:
                results = list(self.prolog.query(query))
                if results:
                    r = results[0]
                    return (r['X'], r['Y'], r['Z'])
            except:
                pass
            return None
        else:
            return self.facts.get(f'position_{object_name}')
    
    def is_goal_object(self, object_id):
        """Check if object is the goal (red target)"""
        if self.prolog:
            query = f"is_goal({object_id})"
            try:
                results = list(self.prolog.query(query))
                return len(results) > 0
            except:
                return False
        else:
            obj = self.facts.get(f'object_{object_id}')
            return obj and obj.get('color') == 'red'
    
    def should_avoid_object(self, object_id):
        """Check if object should be avoided (obstacle)"""
        if self.prolog:
            query = f"should_avoid({object_id})"
            try:
                results = list(self.prolog.query(query))
                return len(results) > 0
            except:
                return False
        else:
            obj = self.facts.get(f'object_{object_id}')
            return obj and obj.get('type') == 'static'


# Global knowledge base instance
_kb_instance = None

def get_knowledge_base():
    """Get or create the global knowledge base instance"""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance

