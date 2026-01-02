# IIS Master Project: Autonomous Navigate-to-Grasp Challenge

## 1. Project Overview
This project requires you to design, implement, and integrate the full cognitive and motor stack for a robotic agent. You must apply the 10 core modules of the **Intelligent Interactive Systems (IIS)** course to move from a raw URDF model to a fully autonomous, reasoning, and learning agent situated in a physical environment.

---

## 2. Task Specification (Module 1)
**Objective:** The robot must navigate a room with obstacles, reach a table, and successfully grasp an object placed on top of it.

* **Embodiedness:** Consider your robot's physical constraints (mass, torque limits, kinematic chain).
* **Situatedness:** The environment includes floor friction, obstacles, and a target table. For the sake of simplification, objects and obstables are just cylinders. The room is cubic with a flat horizontal floor. The table is a traditional four-leg support.
* **Success Conditions:** Robot reaches the table and lifts the object without colliding with obstacles.

---

## 3. The 10 Technical Modules

### [M1] System Requirements
Document the "Embodiedness" and "Situatedness" of your agent. Define the task constraints and safety boundaries.

### [M2] Hardware (URDF)
Create a custom URDF in `src/robot/`. Define the kinematic tree (base + arm), visual/collision shapes, and inertial properties.

### [M3] Sensors (Preprocessing)
Mount joint encoders, odometry, and rgbd cameras. Implement denoising and data synchronization.

### [M4] Perception
Detect objects and obstacles based of predefined attributes such as color, size. Identify the table plane using **RANSAC**. Use **PCA** (Principal Component Analysis) on the object/obstacle point cloud to find the optimal pose, which is essential for obstacle avoidance and grasping.

### [M5] State Estimation
Implement a **Kalman Filter** to fuse noisy sensor data and control inputs into a reliable state estimate $(\hat{x}, \hat{y}, \hat{\theta})$.

### [M6] Motion Control
Develop **PID Controllers** for both wheel navigation and arm manipulation. Address steady-state errors and overshoot.

### [M7] Action Planning
Design a high-level action sequencer (Finite State Machine or Task Tree) to manage the mission: `Search -> Navigate -> Grasp`. Consider failure recovery.

### [M8] Knowledge Representation
Use **Prolog (PySwip)** to store semantic information about the world state. Query the Knowledge Base to reason about object, properties, affordances and URDF frame relations.

### [M9] Learning
Optimize your system through experience. Implement a routine to "learn" or tune parameters (e.g., PID gains or vision thresholds) based on past success/failure.

### [M10] Cognitive Architecture
Integrate all modules into a unified "Sense-Think-Act" loop in `notebooks/cognitive_robots.ipynb`.

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

---

## 6. A Work Plan

- Week 1 (Modules 1-3): Define the task and "build" the robot in the URDF, mounting sensors, and generate the environment.
- Week 2 (Modules 4-5): Implement perception of objects and navigation of robot
- Week 3 (Modules 6-8): Write the motion planner, the PID controller for the wheels/arm and the Prolog logic to represent and access knowledge.
- Week 4 (Modules 9-10): Integrate the loop, handle failures (e.g., robot hits obstacle), and run "experience trials" to optimize their parameters.
- Week 6 (Tests): Ensure the system is running as expected and identify limitations.
- Week 7 (Presentation): Prepare documentation and presentation.

---

## 7. Evaluation Criteria and Weight

- Module Implementation, 50% (5% per module)
- System Robustness, 20% (Handling noise, changes & pushes)
- Integration & Logic, 20% (Clean architecture & reasoning)
- Optimization, 10% (Speed)
