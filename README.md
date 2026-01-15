# IIS Master Project: Autonomous Navigate-to-Grasp Challenge

## 1. Project Overview
This project requires you to design, implement, and integrate the full cognitive and motor stack for a robotic agent. You must apply the 10 core modules of the **Intelligent Interactive Systems (IIS)** course to move from a raw URDF model to a fully autonomous, reasoning, and learning agent situated in a physical environment.

<p align="center">
  <img src="./imgs/Illustration.jpg" alt="Robot in Action" width="600">
</p>

---

## 2. Task Specification (Module 1)
**Objective:** The robot must navigate a room with obstacles, reach a table, and successfully grasp an object placed on top of it.

* **Embodiedness:** Define and respect the robot’s physical constraints (mass, torque limits, kinematic chain).
* **Situatedness:** The room is a $10\text{m} \times 10\text{m} \times 10\text{m}$ cube. **Every execution generates a randomized initial scene configuration (i.e., object position).** You must keep this initial configuration as map in your knowledge base for guiding navigation, except the pose of the target object.
    * **Floor/Walls/Ceiling:** Floor ([0.2, 0.2, 0.2], $\mu=0.5$), Walls ([0.8, 0.8, 0.8]), Ceiling ([1.0, 1.0, 1.0]).
    * **The Table:** Surface $1.5\text{m} \times 0.8\text{m}$ at $z=0.625\text{m}$. Mass: $10\text{kg}$. Color: Brown ([0.5, 0.3, 0.1]). Position: Randomized on floor.
    * **Target Object:** Cylinder ($r=0.04\text{m}, h=0.12\text{m}$). Mass: $0.5\text{kg}$. Color: Red ([1.0, 0.0, 0.0]). Position: Randomized on table surface.
    * **Obstacles:** Five static cubes ($0.4\text{m}$ side). Mass: $10\text{kg}$. Color: Blue ([0.0, 0.0, 1.0]), Pink, Orange, Yellow, Black . Position: Randomized on floor.
* **Success Conditions:** Robot reaches the table and lifts the object without colliding with obstacles.

---

## 3. The 10 Technical Modules

### [M1] System Requirements
Document the "Embodiedness" and "Situatedness" of your agent. Define the task constraints and safety boundaries.

### [M2] Hardware (URDF)
Create a custom URDF in `src/robot/`. Define the kinematic tree (base + arm), visual/collision shapes, and inertial properties. Mount links for sensors (e.g., imu, odometry, and RGB-D cameras). Create also the URDFs of the room, obstacles, table and target object in `src/environment/`. Finally, write the program `world_builder.py` for generating the scene with a random configuration. Any execution of this program generates a new configuration as discussed earlier.

### [M3] Sensors (Preprocessing)
Acquire sensor data by using the sensor wrapper library in `src/robot/sensor_wrapper.py`. This library requires you to input the link ids of the sensors mounted on the robot. Model noise (e.g., handling noise $\mu, \sigma$ via Law of Large Numbers) and preprocess the data.

### [M4] Perception
Detect objects and obstacles based on predefined attributes (color, size). Identify the table plane using **RANSAC**. Use **PCA** (Principal Component Analysis) on the object/obstacle point cloud to find the optimal pose for avoidance and grasping.

### [M5] State Estimation
Implement a **Particle Filter** to fuse noisy sensor data and control inputs into a reliable state estimate $(x, y, \theta)$.

### [M6] Motion Control
Develop **PID Controllers** for both navigation and arm manipulation (Rely **pyBullet's Internal Controller**). Address steady-state errors and overshoot. The path planning is performed with **Prolog (PySwip)** whereas grasp planning is performed with **Inverse Kinematics Function of PyBullet**.

### [M7] Action Planning
Design a high-level action sequencer (Finite State Machine or Task Tree) to manage the mission: `Search -> Navigate -> Grasp`. Consider failure recovery.

### [M8] Knowledge Representation
Use **Prolog (PySwip)** to store semantic information about the world state (e.g., `color(target, red)`, `is_fixed(obstacle)`). Query the KB to reason about object properties, and affordances.

### [M9] Learning
Optimize your system through experience. Implement a routine to "learn" or tune parameters (e.g., PID gains or vision thresholds) based on past success/failure. Learning can take place online or offline.

### [M10] Cognitive Architecture
Integrate all modules into a unified "Sense-Think-Act" loop in `executive/cognitive_architecture.py`.

---

## 4. Environment & Tools
* **Simulator:** PyBullet (Rigid body physics).
* **Language:** Python 3.10.
* **Logic Engine:** SWI-Prolog (via PySwip).
* **Deployment:** Dockerized environment compatible with BinderHub.

---

## 5. Repository Structure
- `/executables`: Integration and execution (The Cognitive Architecture).
- `/src/modules`: Individual logic for Perception, Control, Planning, etc.
- `/src/robot`: URDF files and sensor wrappers.
- `/src/environment`: World building and physics parameters. Ensure scene configuration randomization and provide data for initial scene map.

---

## 6. A Work Plan

- **Week 1 (Modules 1-3):** Define the task and "build" the robot in the URDF, mounting sensors, and generate the environment.
- **Week 2 (Modules 4-5):** Implement perception of objects and navigation of robot.
- **Week 3 (Modules 6-8):** Write the motion planner, the PID controller for the wheels/arm and the Prolog logic to represent and access knowledge.
- **Week 4 (Modules 9-10):** Integrate the loop, handle failures (e.g., robot hits obstacle), and run "experience trials" to optimize their parameters.
- **Week 5 (Tests):** Ensure the system is running as expected and identify limitations.
- **Week 6 (Presentation):** Prepare documentation and final presentation.

---

## 7. Evaluation Criteria and Weight

- **Module Implementation**, 50% (5% per module)
- **System Robustness**, 20% (Handling noise, changes & pushes)
- **Integration & Logic**, 20% (Clean architecture & reasoning)
- **Optimization**, 10% (Speed/Performance)


## 8.  🐳 Docker Installation & Launch Guide for Linux Systems

Run the following commands in sequence to set up the environment, configure permissions, and launch the simulation:

```bash
# 1. Install Docker Compose
sudo apt-get update && sudo apt-get install docker-compose-plugin

# 2. Add user to Docker group & apply permissions
sudo usermod -aG docker $USER && newgrp docker

# 3. Grant GUI access for X11 (Required for PyBullet window)
xhost +local:docker

# 4. Build and Launch the simulation
docker compose up --build

# 5. Next Launches of the simulation
docker compose up

```
## 9. 🚀 Windows Installation & Launch Guide

### 1. Requirements
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) (WSL 2 enabled)
* [VcXsrv Windows X Server](https://sourceforge.net/projects/vcxsrv/)

### 2. GUI Setup (Required every session)
To see the PyBullet window on Windows, you must start the X-Server manually:
1. Open **XLaunch**.
2. Select **Multiple Windows** → **Next**.
3. Select **Start no client** → **Next**.
4. **Crucial:** Check **"Disable Access Control"**.
5. Click **Finish** (Keep it running in the system tray).

### 3. Execution Commands
Open PowerShell in the project root:

```powershell
# Build the environment
docker compose build

# Launch the simulation
docker compose up

```

## 10. 🍎 macOS Installation & Launch Guide

### 1. Requirements
* [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
* [XQuartz](https://www.xquartz.org/)

### 2. GUI Setup (Required once)
1. Install XQuartz and **Restart your Mac**.
2. Open XQuartz, go to **Settings** (or Preferences) → **Security**.
3. Check the box: **"Allow connections from network clients"**.
4. Quit and restart XQuartz.

### 3. GUI Bridge (Every session)
In your Mac terminal, you must allow Docker to connect to XQuartz:
```bash
# Allow local connections
xhost +localhost
```
### 4. Execution Commands
Open PowerShell in the project root:

```bash
# Build the environment
docker compose build

# Launch the simulation
docker compose up

```

Note that your working directory on the local machine is linked to the working directory of the container (see binder/docker-compose.yml). Changes on your local machine are directly reflected on the docker container. Hence, no need to rebuild the container after modifying your codes.
