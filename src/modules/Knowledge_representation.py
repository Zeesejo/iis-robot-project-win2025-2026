from pyswip import Prolog

class KnowledgeRepresentation:
    """
    Python interface for the Prolog Knowledge Base (Dynamic_KB.pl).
    Provides access to object properties, robot info, sensors, and dynamic updates.
    """

    def __init__(self, kb_file="Dynamic_KB.pl"):
        """
        Initialize the Prolog KB interface.
        """
        self.prolog = Prolog()
        self.prolog.consult(kb_file)

    # -----------------------------
    # Basic Object Queries
    # -----------------------------
    def objects(self):
        """Return a list of all objects in the world."""
        return [obj['X'] for obj in self.prolog.query("object(X)")]

    def colors(self):
        """Return a dictionary mapping object -> color."""
        return {obj['X']: obj['C'] for obj in self.prolog.query("color(X, C)")}

    def fixed_objects(self):
        """Return a list of all fixed objects."""
        return [obj['X'] for obj in self.prolog.query("is_fixed(X)")]

    def movable_objects(self):
        """Return a list of all movable objects."""
        return [obj['X'] for obj in self.prolog.query("is_movable(X)")]

    def pickable_objects(self):
        """Return a list of objects that can be picked."""
        return [obj['X'] for obj in self.prolog.query("can_pick(X)")]

    def object_position(self, obj_name):
        """
        Return the (x, y, z) position of an object.
        Returns None if the object is not found.
        """
        q = list(self.prolog.query(f"position({obj_name}, X, Y, Z)"))
        if q:
            return (q[0]['X'], q[0]['Y'], q[0]['Z'])
        return None

    # -----------------------------
    # Robot Info
    # -----------------------------
    def robot_links(self):
        """Return a list of all robot links."""
        return [obj['X'] for obj in self.prolog.query("robot_link(X)")]

    def robot_capabilities(self):
        """
        Return a dictionary describing robot capabilities:
        - sensors: list of attached sensors
        - can_move: True/False
        - has_arm: True/False
        """
        caps = {}
        for obj in self.prolog.query("has_sensor(robot, S)"):
            caps.setdefault('sensors', []).append(obj['S'])
        for _ in self.prolog.query("can_move(robot)"):
            caps['can_move'] = True
        for _ in self.prolog.query("has_arm(robot)"):
            caps['has_arm'] = True
        return caps

    # -----------------------------
    # Sensor Queries
    # -----------------------------
    def sensors(self):
        """Return a list of all sensors in the environment."""
        return [s['S'] for s in self.prolog.query("sensor(S, T)")]

    def sensor_types(self):
        """Return a dictionary mapping sensor -> type."""
        return {s['S']: s['T'] for s in self.prolog.query("sensor(S, T)")}

    def detectable_objects(self):
        """
        Return a list of objects currently detectable by robot sensors.
        Removes duplicates automatically.
        """
        objs = [obj['Obj'] for obj in self.prolog.query("detectable_objects(Obj)")]
        return list(set(objs))  # remove duplicates

    def visible_to_sensor(self, sensor_name):
        """
        Return a list of objects visible to a specific sensor.
        """
        objs = [obj['Obj'] for obj in self.prolog.query(f"visible_to_sensor({sensor_name}, Obj)")]
        return list(set(objs))  # remove duplicates

    # -----------------------------
    # Dynamic Updates
    # -----------------------------
    def update_object_position(self, obj_name, x, y, z):
        """
        Update the position of an object in the Prolog KB.
        Typically called after a sensor reading.
        """
        self.prolog.query(f"update_position({obj_name},{x},{y},{z})")

    # -----------------------------
    # Movement / Success Queries
    # -----------------------------
    def navigate_to_table(self):
        """
        Ask Prolog to move the robot towards the table recursively.
        Updates robot_position_history dynamically.
        """
        return list(self.prolog.query("navigate_to_table"))

    def check_success(self):
        """
        Return True if the robot can successfully grasp the target.
        """
        return bool(list(self.prolog.query("success")))

    # -----------------------------
    # Debug / Helpers
    # -----------------------------
    def list_objects(self):
        """Return a list of all object names (debug)."""
        return [obj['X'] for obj in self.prolog.query("list_objects")]

    def list_pickable(self):
        """Return a list of pickable objects (debug)."""
        return [obj['X'] for obj in self.prolog.query("list_pickable")]

    def list_reachable(self):
        """Return a list of reachable objects (debug)."""
        return [obj['X'] for obj in self.prolog.query("list_reachable")]

    def print_robot_path(self):
        """Return the robot path history as a list of (X,Y,Z) tuples."""
        return [(x['X'], x['Y'], x['Z']) for x in self.prolog.query("robot_position_history(X,Y,Z)")]
