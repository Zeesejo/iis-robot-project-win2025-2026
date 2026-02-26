"""
M2 - World Builder
Generates a randomized scene: room, table, target object, 5 obstacles.
Saves the initial scene configuration map to a JSON file.
Rule: robot must NOT use p.getBasePositionAndOrientation() except here for map validation.
"""

import pybullet as p
import pybullet_data
import numpy as np
import json
import os
import random

# ---------- Paths ----------
ENV_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_DIR = os.path.join(ENV_DIR, '..', 'robot')
MAP_PATH = os.path.join(ENV_DIR, '..', '..', 'executables', 'scene_map.json')

OBSTACLE_COLORS = [
    [0.0, 0.0, 1.0, 1],   # Blue
    [1.0, 0.4, 0.7, 1],   # Pink
    [1.0, 0.5, 0.0, 1],   # Orange
    [1.0, 1.0, 0.0, 1],   # Yellow
    [0.0, 0.0, 0.0, 1],   # Black
]


def _sample_non_overlapping(existing_positions, min_dist=1.2, xlim=(-4, 4), ylim=(-4, 4)):
    """Sample a 2D position that doesn't overlap with existing ones."""
    for _ in range(1000):
        x = random.uniform(*xlim)
        y = random.uniform(*ylim)
        ok = all(np.linalg.norm([x - ex, y - ey]) > min_dist
                 for ex, ey in existing_positions)
        if ok:
            return x, y
    raise RuntimeError("Could not place object without overlap after 1000 tries.")


def build_world(physics_client=None, gui=True):
    """
    Build the full scene in PyBullet.
    Returns a dict with all object IDs and the initial scene map.
    """
    if physics_client is None:
        mode = p.GUI if gui else p.DIRECT
        physics_client = p.connect(mode)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)

    # ---------- Floor (via PyBullet plane for physics) ----------
    plane_id = p.loadURDF('plane.urdf')
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.2, 0.2, 0.2, 1])

    occupied = []

    # ---------- Robot ----------
    robot_x, robot_y = _sample_non_overlapping(occupied, min_dist=1.0)
    occupied.append((robot_x, robot_y))
    robot_id = p.loadURDF(
        os.path.join(ROBOT_DIR, 'robot.urdf'),
        basePosition=[robot_x, robot_y, 0.08],
        baseOrientation=p.getQuaternionFromEuler([0, 0, random.uniform(0, 2 * np.pi)])
    )

    # ---------- Table ----------
    table_x, table_y = _sample_non_overlapping(occupied, min_dist=1.5)
    occupied.append((table_x, table_y))
    table_id = p.loadURDF(
        os.path.join(ENV_DIR, 'table.urdf'),
        basePosition=[table_x, table_y, 0],
        useFixedBase=True
    )

    # ---------- Target Object (on table surface z=0.625+0.06=0.685) ----------
    target_x = table_x + random.uniform(-0.5, 0.5)
    target_y = table_y + random.uniform(-0.3, 0.3)
    target_z = 0.625 + 0.06   # table surface + half-height of cylinder
    target_id = p.loadURDF(
        os.path.join(ENV_DIR, 'target_object.urdf'),
        basePosition=[target_x, target_y, target_z]
    )

    # ---------- Obstacles ----------
    obstacle_ids = []
    for i in range(5):
        ox, oy = _sample_non_overlapping(occupied, min_dist=0.8)
        occupied.append((ox, oy))
        obs_id = p.loadURDF(
            os.path.join(ENV_DIR, 'obstacle.urdf'),
            basePosition=[ox, oy, 0],
            useFixedBase=True
        )
        p.changeVisualShape(obs_id, -1, rgbaColor=OBSTACLE_COLORS[i])
        obstacle_ids.append(obs_id)

    # ---------- Stabilize simulation ----------
    for _ in range(100):
        p.stepSimulation()

    # ---------- Build initial scene map ----------
    # Use getBasePositionAndOrientation here ONLY for map building (allowed in world_builder)
    robot_pos, robot_orn = p.getBasePositionAndOrientation(robot_id)
    table_pos, _ = p.getBasePositionAndOrientation(table_id)
    obs_info = []
    for i, oid in enumerate(obstacle_ids):
        opos, _ = p.getBasePositionAndOrientation(oid)
        obs_info.append({
            "id": oid,
            "position": list(opos),
            "color": OBSTACLE_COLORS[i],
            "size": 0.4
        })

    scene_map = {
        "robot": {
            "id": robot_id,
            "position": list(robot_pos),
            "orientation": list(robot_orn)
        },
        "table": {
            "id": table_id,
            "position": list(table_pos),
            "surface_z": 0.625,
            "size": [1.5, 0.8]
        },
        "target_object": {
            "id": target_id,
            "position_unknown": True,   # Robot must estimate this via perception
            "color": [1.0, 0.0, 0.0],
            "radius": 0.04,
            "height": 0.12
        },
        "obstacles": obs_info,
        "room": {
            "size": [10, 10, 10],
            "floor_color": [0.2, 0.2, 0.2],
            "wall_color": [0.8, 0.8, 0.8],
            "ceiling_color": [1.0, 1.0, 1.0]
        }
    }

    # Persist map
    os.makedirs(os.path.dirname(MAP_PATH), exist_ok=True)
    with open(MAP_PATH, 'w') as f:
        json.dump(scene_map, f, indent=2)

    print(f"[WorldBuilder] Scene map saved to {MAP_PATH}")

    return {
        "robot_id": robot_id,
        "plane_id": plane_id,
        "table_id": table_id,
        "target_id": target_id,
        "obstacle_ids": obstacle_ids,
        "scene_map": scene_map,
        "physics_client": physics_client
    }


if __name__ == '__main__':
    world = build_world(gui=True)
    print("Scene built. IDs:", {k: v for k, v in world.items() if k != 'scene_map'})
    import time
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1. / 240.)
