import pybullet as p
import pybullet_data
import random
import time
import json
import os
import math

# --- Configuration ---
ROOM_DIM = 10.0
OBSTACLE_COUNT = 5
OBSTACLE_COLORS = [
    [0, 0, 1, 1],  # Blue
    [1, 0.75, 0.8, 1],  # Pink
    [1, 0.64, 0, 1],  # Orange
    [1, 1, 0, 1],  # Yellow
    [0, 0, 0, 1]  # Black
]
TABLE_DIM = [1.5, 0.8]  # x, y
TABLE_HEIGHT = 0.625
TARGET_HEIGHT = 0.12

# File Paths (Relative to where python is executed)
URDF_PATH = "../src/environment/"
ROBOT_URDF = "../src/robot/robot.urdf"


def get_random_pos(bounds, min_dist_from_origin=1.0):
    """Generates random x, y within bounds, ensuring it's not too close to (0,0) where robot starts."""
    while True:
        x = random.uniform(bounds[0], bounds[1])
        y = random.uniform(bounds[2], bounds[3])
        if math.sqrt(x ** 2 + y ** 2) > min_dist_from_origin:
            return [x, y]


def is_overlapping(pos, radius, existing_objects):
    """Simple circular distance check to prevent spawning objects inside each other."""
    for obj in existing_objects:
        existing_pos = obj['pos']
        existing_radius = obj['radius']
        dist = math.sqrt((pos[0] - existing_pos[0]) ** 2 + (pos[1] - existing_pos[1]) ** 2)
        if dist < (radius + existing_radius + 0.2):  # 0.2 buffer
            return True
    return False


def build_world(gui=True):
    if gui:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # 1. Load Room (Which now includes the floor)
    room_id = p.loadURDF(os.path.join(URDF_PATH, "room.urdf"), useFixedBase=True)

    # 2. Set Floor Friction (Requirement: mu=0.5)
    # The floor is link index 0 in our new URDF (since it's the first child of world)
    # We set lateral friction to 0.5
    p.changeDynamics(room_id, -1, lateralFriction=0.5)
    p.changeDynamics(room_id, 0, lateralFriction=0.5)

    # 3. Load Robot (at 0,0,0) - Loaded to ensure we don't spawn obstacles on top of it
    robot_start_pos = [0, 0, 0]
    robot_start_orn = p.getQuaternionFromEuler([0, 0, 0])
    robot_id = p.loadURDF(ROBOT_URDF, robot_start_pos, robot_start_orn)

    # Store spawned objects for collision checking logic
    # Structure: {'pos': [x,y], 'radius': r}
    spawned_objects = [{'pos': [0, 0], 'radius': 0.5}]  # Robot radius approx

    knowledge_base = {
        "room": {"w": 10, "l": 10, "h": 10},
        "obstacles": [],
        "table": {}
    }

    # 4. Spawn Table (Randomized)
    # Bounds: -4 to 4 to keep inside 10x10 room
    while True:
        t_pos_xy = get_random_pos([-4, 4, -4, 4])
        if not is_overlapping(t_pos_xy, 1.0, spawned_objects):  # Table radius approx 1.0
            break

    table_pos = [t_pos_xy[0], t_pos_xy[1], 0]
    table_orn = p.getQuaternionFromEuler([0, 0, random.uniform(-3.14, 3.14)])
    table_id = p.loadURDF(os.path.join(URDF_PATH, "table.urdf"), table_pos, table_orn, useFixedBase=True)

    spawned_objects.append({'pos': t_pos_xy, 'radius': 1.0})
    knowledge_base['table'] = {"position": table_pos, "orientation": list(table_orn), "size": [1.5, 0.8]}

    # 5. Spawn Target Object (ON the table)
    # We must calculate the position relative to the table center + table rotation
    # For simplicity in this logic, we spawn it at table center (x,y) but z = table_height + object_half_height
    # If you want it randomized ON the table surface, you need local offset rotated by quaternion.

    # Random offset on table surface (local coords)
    dx = random.uniform(-0.6, 0.6)  # Table is 1.5 long
    dy = random.uniform(-0.3, 0.3)  # Table is 0.8 wide

    # Transform local offset to world
    # (Simple logic: just place it centered for now to ensure stability, or use getMatrix)
    # To keep it simple and robust for student code:
    target_z = 0.625 + (0.12 / 2.0) + 0.01  # table height + half cyl height + margin
    target_pos = [t_pos_xy[0], t_pos_xy[1], target_z]

    target_id = p.loadURDF(os.path.join(URDF_PATH, "target.urdf"), target_pos, useFixedBase=False)
    # Note: We do NOT add target to knowledge_base['target_pose'] because the robot must find it.

    # 6. Spawn Obstacles
    for i in range(OBSTACLE_COUNT):
        while True:
            o_pos_xy = get_random_pos([-4.5, 4.5, -4.5, 4.5])
            if not is_overlapping(o_pos_xy, 0.5, spawned_objects):  # Obstacle radius approx 0.5
                break

        obs_pos = [o_pos_xy[0], o_pos_xy[1], 0.2]  # Half height of 0.4 box
        obs_orn = p.getQuaternionFromEuler([0, 0, random.uniform(0, 3.14)])
        obs_id = p.loadURDF(os.path.join(URDF_PATH, "obstacle.urdf"), obs_pos, obs_orn, useFixedBase=False)

        # Apply Color
        color = OBSTACLE_COLORS[i]
        p.changeVisualShape(obs_id, -1, rgbaColor=color)
        p.changeDynamics(obs_id, -1, mass=10.0)

        spawned_objects.append({'pos': o_pos_xy, 'radius': 0.4})
        knowledge_base['obstacles'].append({
            "id": i,
            "position": obs_pos,
            "orientation": list(obs_orn),
            "color_rgba": color
        })

    # 7. Save Initial Map
    with open("initial_map.json", "w") as f:
        json.dump(knowledge_base, f, indent=4)

    print("World Generated. Map saved to initial_map.json")

    return robot_id, table_id, room_id, target_id


if __name__ == "__main__":
    # Test execution
    build_world(gui=True)
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1. / 240.)