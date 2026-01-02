# IIS Master Project: Autonomous Navigate-to-Grasp Challenge

## 1. Project Overview
This project requires you to design, implement, and integrate the full cognitive and motor stack for a robotic agent. You must apply the 10 core modules of the **Intelligent Interactive Systems (IIS)** course to move from a raw URDF model to a fully autonomous, reasoning, and learning agent situated in a physical environment.

---

## 2. Task Specification (Module 1)
**Objective:** The robot must navigate a room with obstacles, reach a table, and successfully grasp an object placed on top of it.

* **Embodiedness:** Define and respect the robot’s physical constraints (mass, torque limits, kinematic chain).
* **Situatedness:** The room is a $10\text{m} \times 10\text{m} \times 10\text{m}$ cube. **Every execution generates a randomized initial scene configuration (i.e., object position).** You must keep this initial configuration as map in your knowledge base for guiding navigation, except the pose of the target object.
    * **Floor/Walls/Ceiling:** Floor ([0.2, 0.2, 0.2], $\mu=0.5$), Walls ([0.8, 0.8, 0.8]), Ceiling ([1.0, 1.0, 1.0]).
    * **The Table:** Surface $1.5\text{m} \times 0.8\text{m}$ at $z=0.625\text{m}$. Mass: $10\text{kg}$. Color: Brown ([0.5, 0.3, 0.1]). Position: Randomized on floor.
    * **Target Object:** Cylinder ($r=0.04\text{m}, h=0.12\text{m}$). Mass: $0.5\text{kg}$. Color: Red ([1.0, 0.0, 0.0]). Position: Randomized on table surface.
    * **Obstacles:** Five static cubes ($0.4\text{m}$ side). Mass: $10\text{kg}$. Color: Blue ([0.0, 0.0, 1.0]). Position: Randomized on floor.
* **Success Conditions:** Robot reaches the table and lifts the object without colliding with obstacles.

---

## 3. The 10 Technical Modules

### [M1] System Requirements
Document the "Embodiedness" and "Situatedness" of your agent. Define the task constraints and safety boundaries.

### [M2] Hardware (URDF)
Create a custom URDF in `src/robot/`. Define the kinematic tree (base + arm), visual/collision shapes, and inertial properties. Create also the URDFs of the room, obstacles, table and target object in `src/environment/`. Finally, write the program `world_builder.py` for generating the scene with a random configuration. Any execution of this program generates a new configuration as discussed earlier.

### [M3] Sensors (Preprocessing)
Mount joint encoders, odometry, and RGB-D cameras. Model noise (e.g., handling noise $\mu, \sigma$ via Law of Large Numbers) and data synchronization.

### [M4] Perception
Detect objects and obstacles based on predefined attributes (color, size). Identify the table plane using **RANSAC**. Use **PCA** (Principal Component Analysis) on the object/obstacle point cloud to find the optimal pose for avoidance and grasping.

### [M5] State Estimation
Implement a **Particle Filter** to fuse noisy sensor data and control inputs into a reliable state estimate $(x, y, \theta)$.

### [M6] Motion Control
Develop **PID Controllers** for both wheel navigation and arm manipulation. Address steady-state errors and overshoot.

### [M7] Action Planning
Design a high-level action sequencer (Finite State Machine or Task Tree) to manage the mission: `Search -> Navigate -> Grasp`. Consider failure recovery.

### [M8] Knowledge Representation
Use **Prolog (PySwip)** to store semantic information about the world state (e.g., `color(target, red)`, `is_fixed(obstacle)`). Query the KB to reason about object properties, affordances, and URDF frame relations.

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
