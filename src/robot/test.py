import os
import pybullet as p

p.connect(p.DIRECT)

here = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(here, "robot.urdf")
print("URDF:", urdf_path, "exists?", os.path.exists(urdf_path))

robot_id = p.loadURDF(urdf_path, useFixedBase=True)
