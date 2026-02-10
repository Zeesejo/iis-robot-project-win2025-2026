import pybullet as p
import pybullet_data
import time
import math
from pyswip import Prolog

# --------------------------
# PID Controller Class
# --------------------------
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, current_value, dt):
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# --------------------------
# Robot Initialization
# --------------------------
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load robot URDF
robot_id = p.loadURDF("robot.urdf", basePosition=[0, 0, 0.0])

# Define joints for navigation and arm
wheel_joints = [0, 1]  # example: left/right wheels
arm_joints = [2, 3, 4, 5, 6]  # example: 5-DOF arm

# --------------------------
# Prolog Path Planner
# --------------------------
prolog = Prolog()
# Load Prolog knowledge base
prolog.consult("path_planning.pl")

def plan_path(start, goal):
    # Returns a list of waypoints from Prolog
    query = f"plan_path({start}, {goal}, Path)."
    for sol in prolog.query(query):
        return sol["Path"]
    return []

# --------------------------
# Motion Control Functions
# --------------------------
def move_to_goal(goal_pos, dt=1./240.):
    """
    Move robot base using PID controller to goal position (x, y)
    """
    # Get current position
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    x, y, _ = pos

    # PID Controllers for X and Y separately
    pid_x = PIDController(1.0, 0.0, 0.1, setpoint=goal_pos[0])
    pid_y = PIDController(1.0, 0.0, 0.1, setpoint=goal_pos[1])

    while math.hypot(goal_pos[0]-x, goal_pos[1]-y) > 0.05:
        x, y, _ = p.getBasePositionAndOrientation(robot_id)[0]
        vx = pid_x.compute(x, dt)
        vy = pid_y.compute(y, dt)

        # Convert to wheel velocities (differential drive)
        # Example simple conversion:
        left_wheel_vel = vx - vy
        right_wheel_vel = vx + vy

        p.setJointMotorControlArray(
            robot_id,
            wheel_joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[left_wheel_vel, right_wheel_vel]
        )
        p.stepSimulation()
        time.sleep(dt)

def grasp_object(target_pos, target_orient):
    """
    Move the robot arm to grasp an object using Inverse Kinematics
    """
    # Use PyBullet IK
    ik_solution = p.calculateInverseKinematics(robot_id, arm_joints[-1], target_pos, targetOrientation=target_orient)
    
    # Apply to arm joints
    for i, joint in enumerate(arm_joints):
        p.setJointMotorControl2(
            robot_id,
            joint,
            controlMode=p.POSITION_CONTROL,
            targetPosition=ik_solution[i],
            force=500
        )
    # Step simulation until arm reaches target
    for _ in range(240):  # 1 second simulation at 240 Hz
        p.stepSimulation()
        time.sleep(1./240.)

# --------------------------
# Example Execution
# --------------------------
# 1. Plan path using Prolog
path = plan_path([0, 0], [2, 2])  # Start at (0,0), goal at (2,2)
for waypoint in path:
    move_to_goal(waypoint)

# 2. Grasp object at position
target_position = [2.0, 2.0, 0.625]  # Example table height
target_orientation = p.getQuaternionFromEuler([0, math.pi, 0])
grasp_object(target_position, target_orientation)

# Disconnect simulation
p.disconnect()

