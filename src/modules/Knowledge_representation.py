from pyswip import Prolog

class KnowledgeRepresentation:
    def __init__(self, kb_file="Dynamic_KB.pl"):
        self.prolog = Prolog()
        self.prolog.consult(kb_file)

    def objects(self):
        """Return all objects in the world"""
        return [obj['X'] for obj in self.prolog.query("object(X)")]

    def colors(self):
        """Return dict of object -> color"""
        return {obj['X']: obj['C'] for obj in self.prolog.query("color(X, C)")}

    def fixed_objects(self):
        """Return list of fixed objects"""
        return [obj['X'] for obj in self.prolog.query("is_fixed(X)")]

    def movable_objects(self):
        """Return list of movable objects"""
        return [obj['X'] for obj in self.prolog.query("is_movable(X)")]

    def pickable_objects(self):
        """Return objects that can be picked"""
        return [obj['X'] for obj in self.prolog.query("can_pick(X)")]

    def object_position(self, obj_name):
        """Return position (x,y,z) of an object"""
        q = list(self.prolog.query(f"position({obj_name}, X, Y, Z)"))
        if q:
            return (q[0]['X'], q[0]['Y'], q[0]['Z'])
        return None

    def robot_links(self):
        """Return list of robot links"""
        return [obj['X'] for obj in self.prolog.query("robot_link(X)")]

    def robot_capabilities(self):
        """Return robot capabilities"""
        caps = {}
        for obj in self.prolog.query("has_sensor(robot, S)"):
            caps.setdefault('sensors', []).append(obj['S'])
        for obj in self.prolog.query("can_move(robot)"):
            caps['can_move'] = True
        for obj in self.prolog.query("has_arm(robot)"):
            caps['has_arm'] = True
        return caps

# === TEST QUERIES ===
if __name__ == "__main__":
    kr = KnowledgeRepresentation()

    print("Objects:", kr.objects())
    print("Colors:", kr.colors())
    print("Fixed objects:", kr.fixed_objects())
    print("Movable objects:", kr.movable_objects())
    print("Pickable objects:", kr.pickable_objects())
    print("Target position:", kr.object_position('target'))
    print("Robot links:", kr.robot_links())
    print("Robot capabilities:", kr.robot_capabilities())
