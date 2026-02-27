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
    [0, 0, 1, 1],       # Blue
    [1, 0.75, 0.8, 1],  # Pink
    [1, 0.64, 0, 1],    # Orange
    [1, 1, 0, 1],       # Yellow
    [0, 0, 0, 1]        # Black
]
TABLE_DIM = [1.5, 0.8]  # x, y
TABLE_HEIGHT = 0.625
TARGET_HEIGHT = 0.12

# File Paths (relative to this file's location for portability)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = _THIS_DIR
# FIX: use robot-1.urdf (prismatic lift + camera on GRIPPER)
ROBOT_URDF = os.path.join(_THIS_DIR, "..", "robot", "robot.urdf")


def get_random_pos(bounds, min_dist_from_origin=1.0):
    """Generates random x, y within bounds, not too close to (0,0)."""
    while True:
        x = random.uniform(bounds[0], bounds[1])
        y = random.uniform(bounds[2], bounds[3])
        if math.sqrt(x ** 2 + y ** 2) > min_dist_from_origin:
            return [x, y]


def is_overlapping(pos, radius, existing_objects):
    """Simple circular distance check."""
    for obj in existing_objects:
        existing_pos = obj['pos']
        existing_radius = obj['radius']
        dist = math.sqrt((pos[0] - existing_pos[0]) ** 2 + (pos[1] - existing_pos[1]) ** 2)
        if dist < (radius + existing_radius + 0.2):
            return True
    return False


def build_world(gui=True):
    if gui:
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(
            cameraDistance=8.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0]
        )
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        print("[WorldBuilder] GUI enabled with camera positioned at robot")
    else:
        p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    p.setPhysicsEngineParameter(
        numSolverIterations=150,
        erp=0.8,
        contactERP=0.8,
        frictionERP=0.8,
        enableConeFriction=1
    )

    # 1. Room
    room_id = p.loadURDF(os.path.join(URDF_PATH, "room.urdf"), useFixedBase=True)
    p.changeDynamics(room_id, -1, lateralFriction=0.5)
    p.changeDynamics(room_id,  0, lateralFriction=0.5)

    # 2. Robot
    robot_start_pos = [0, 0, 0.1]
    robot_start_orn = p.getQuaternionFromEuler([0, 0, 0])
    robot_id = p.loadURDF(ROBOT_URDF, robot_start_pos, robot_start_orn)

    for wheel_joint in [0, 1, 2, 3]:
        p.changeDynamics(robot_id, wheel_joint,
                         lateralFriction=1.5,
                         spinningFriction=0.001,
                         rollingFriction=0.001)
    p.changeDynamics(robot_id, -1, lateralFriction=0.5)
    print(f"[WorldBuilder] Robot loaded at {robot_start_pos} with wheel dynamics configured")

    spawned_objects = [{'pos': [0, 0], 'radius': 0.5}]
    knowledge_base = {"room": {"w": 10, "l": 10, "h": 10}, "obstacles": [], "table": {}}

    # 3. Table
    while True:
        t_pos_xy = get_random_pos([-4, 4, -4, 4])
        if not is_overlapping(t_pos_xy, 1.0, spawned_objects):
            break

    # Debug FIX table position for testing:
    t_pos_xy = [2.0, 0.0]
    table_pos = [t_pos_xy[0], t_pos_xy[1], 0]
    table_orn = p.getQuaternionFromEuler([0, 0, random.uniform(-3.14, 3.14)])
    table_id  = p.loadURDF(os.path.join(URDF_PATH, "table.urdf"), table_pos, table_orn, useFixedBase=True)
    spawned_objects.append({'pos': t_pos_xy, 'radius': 1.0})
    knowledge_base['table'] = {
        "position": table_pos,
        "orientation": list(table_orn),
        "size": [1.5, 0.8]
    }

    # 4. Target (on table)
    dx = random.uniform(-0.6, 0.6)
    dy = random.uniform(-0.3, 0.3)
    table_yaw = p.getEulerFromQuaternion(table_orn)[2]
    world_dx   = dx * math.cos(table_yaw) - dy * math.sin(table_yaw)
    world_dy   = dx * math.sin(table_yaw) + dy * math.cos(table_yaw)
    target_z   = TABLE_HEIGHT + (TARGET_HEIGHT / 2.0) + 0.01
    target_pos = [t_pos_xy[0] + world_dx, t_pos_xy[1] + world_dy, target_z]
    target_id  = p.loadURDF(os.path.join(URDF_PATH, "target.urdf"), target_pos, useFixedBase=False)
    # [F21] Explicitly force pure red so PyBullet camera RGB is reliable.
    # URDF <material> color is often ignored by the renderer; changeVisualShape
    # writes directly to the OpenGL visual, guaranteeing H~0, S~255, V~255 in HSV.
    p.changeVisualShape(target_id, -1, rgbaColor=[1, 0, 0, 1])
    print(f"[WorldBuilder] Target placed at ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}) - color forced red")

    # 5. Obstacles
    for i in range(OBSTACLE_COUNT):
        while True:
            o_pos_xy = get_random_pos([-4.5, 4.5, -4.5, 4.5])
            if not is_overlapping(o_pos_xy, 0.5, spawned_objects):
                break
        obs_pos = [o_pos_xy[0], o_pos_xy[1], 0.2]
        obs_orn = p.getQuaternionFromEuler([0, 0, random.uniform(0, 3.14)])
        obs_id  = p.loadURDF(os.path.join(URDF_PATH, "obstacle.urdf"), obs_pos, obs_orn, useFixedBase=False)
        color   = OBSTACLE_COLORS[i]
        p.changeVisualShape(obs_id, -1, rgbaColor=color)
        p.changeDynamics(obs_id, -1, mass=10.0)
        spawned_objects.append({'pos': o_pos_xy, 'radius': 0.4})
        knowledge_base['obstacles'].append({
            "id": i,
            "position": obs_pos,
            "orientation": list(obs_orn),
            "color_rgba": color
        })

    # 6. Save map
    with open("initial_map.json", "w") as f:
        json.dump(knowledge_base, f, indent=4)

    print("World Generated. Map saved to initial_map.json")
    return robot_id, table_id, room_id, target_id


if __name__ == "__main__":
    build_world(gui=True)
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1. / 240.)
