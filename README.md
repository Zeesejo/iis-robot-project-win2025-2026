# IIS Master Project: Autonomous Navigate-to-Grasp Challenge

## 1. Project Overview
This project requires you to design, implement, and integrate the full cognitive and motor stack for a robotic agent. You must apply the 10 core modules of the **Intelligent Interactive Systems (IIS)** course to move from a raw URDF model to a fully autonomous, reasoning, and learning agent situated in a physical environment.

---

## 2. Task Specification (Module 1)
**Objective:** The robot must navigate a room with obstacles, reach a table, and successfully grasp an object placed on top of it.

* **Embodiedness:** Consider your robot's physical constraints (mass, torque limits, kinematic chain).
* **Situatedness:** The environment includes floor friction, obstacles, and a target table.
* **Success Conditions:** Robot reaches the table and lifts the object without colliding with obstacles.

---

## 3. The 10 Technical Modules

### [M1] System Requirements
Document the "Embodiedness" and "Situatedness" of your agent. Define the task constraints and safety boundaries.

### [M2] Hardware (URDF)
Create a custom URDF in `src/robot/`. Define the kinematic tree (base + arm), visual/collision shapes, and inertial properties.

### [M3] Sensors (Preprocessing)
Mount joint encoders, odometry, and depth cameras. Implement denoising and data synchronization.

### [M4] Perception
Identify the table plane using **RANSAC**. Use **PCA** (Principal Component Analysis) on the object point cloud to find the optimal grasping pose.

### [M5] State Estimation
Implement a **Kalman Filter** to fuse noisy sensor data and control inputs into a reliable state estimate $(\hat{x}, \hat{y}, \hat{\theta})$.

### [M6] Motion Control
Develop **PID Controllers** for both wheel navigation and arm manipulation. Address steady-state errors and overshoot.

### [M7] Action Planning
Design a high-level action sequencer (Finite State Machine or Task Tree) to manage the mission: `Search -> Navigate -> Grasp`.

### [M8] Knowledge Representation
Use **Prolog (PySwip)** to store semantic information. Query the Knowledge Base to reason about object affordances and URDF frame relations.

### [M9] Learning
Optimize your system through experience. Implement a routine to "learn" or tune parameters (e.g., PID gains or vision thresholds) based on past success/failure.

### [M10] Cognitive Architecture
Integrate all modules into a unified "Sense-Think-Act" loop in `notebooks/main_application.ipynb`.

---

## 4. Environment & Tools
* **Simulator:** PyBullet (Rigid body physics).
* **Language:** Python 3.10.
* **Logic Engine:** SWI-Prolog (via PySwip).
* **Deployment:** Dockerized environment compatible with BinderHub.

---

## 5. Repository Structure
- `/notebooks`: Integration and execution (The Cognitive Architecture).
- `/src/modules`: Individual logic for Perception, Control, Planning, etc.
- `/src/robot`: URDF files and sensor wrappers.
- `/src/environment`: World building and physics parameters.
