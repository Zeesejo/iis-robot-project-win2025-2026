"""
Module 9: Learning
Parameter optimization via replay buffer, evaluation, and mutation.
Supports online learning (trial-based), offline learning (replay),
and baseline comparison.
"""

import random
from collections import deque
import csv
import os
import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# =========================
# Default Parameters (tuned for this robot)
# =========================
DEFAULT_PARAMETERS = {
    # Navigation PID gains
    "nav_kp": 2.5,
    "nav_ki": 0.0,
    "nav_kd": 0.1,

    # Angle/heading PID gains
    "angle_kp": 5.0,
    "angle_ki": 0.0,
    "angle_kd": 0.01,

    # Arm/Gripper PID gains
    "arm_kp": 1.2,
    "arm_ki": 0.0,
    "arm_kd": 0.05,

    # Vision thresholds
    "vision_threshold": 0.5,

    # Robot speed limits
    "max_linear_speed": 0.6,
    "max_angular_speed": 1.0
}


# =========================
# Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, capacity=100):
        self.buffer = deque(maxlen=capacity)

    def add(self, parameters, score):
        self.buffer.append((parameters.copy(), float(score)))

    def get_all(self):
        return list(self.buffer)

    def size(self):
        return len(self.buffer)


# =========================
# Evaluator
# =========================
class Evaluator:
    def evaluate(self, simulation_result):
        """
        Evaluate a simulation result.

        Args:
            simulation_result: dict with optional keys:
                - 'success': bool (did the robot complete the task?)
                - 'steps': int (number of simulation steps taken)
                - 'nav_kp', 'arm_kp', etc. (parameter values for heuristic scoring)

        Returns:
            dict with 'score' key, or float score
        """
        if isinstance(simulation_result, dict) and 'success' in simulation_result:
            # Real simulation result
            score = 0
            if simulation_result.get("success", False):
                score += 1000
            score -= simulation_result.get("steps", 0) * 0.1
            return score
        else:
            # Heuristic scoring based on parameter values
            score = 0.5
            if isinstance(simulation_result, dict):
                score += 0.1 * min(simulation_result.get("nav_kp", 1.0), 2.0)
                score += 0.1 * min(simulation_result.get("arm_kp", 1.0), 2.0)
            score += random.uniform(-0.05, 0.05)
            return {"score": score}


# =========================
# Optimizer
# =========================
class Optimizer:
    def random_parameters(self):
        """Generate random parameters based on defaults with small perturbation"""
        params = DEFAULT_PARAMETERS.copy()
        params["nav_kp"] = params["nav_kp"] + random.uniform(-0.3, 0.3)
        params["angle_kp"] = params["angle_kp"] + random.uniform(-0.3, 0.3)
        params["max_linear_speed"] = params["max_linear_speed"] + random.uniform(-0.1, 0.1)
        params["max_angular_speed"] = params["max_angular_speed"] + random.uniform(-0.1, 0.1)
        return params

    def mutate(self, params, scale=0.05):
        """Mutate selected parameters with small random perturbation"""
        new = params.copy()
        for k in ["nav_kp", "angle_kp", "max_linear_speed", "max_angular_speed"]:
            if k in new:
                new[k] = new[k] + random.uniform(-scale, scale)
        return new

    def get_best(self, memory):
        """Get the best parameters from memory and mutate them"""
        experiences = memory.get_all()
        if not experiences:
            return self.random_parameters()
        best = max(experiences, key=lambda x: x[1])
        return self.mutate(best[0])


# =========================
# Learner
# =========================
class Learner:
    def __init__(self, robot=None, csv_file="data/experiences.csv"):
        """
        Initialize the learner.

        Args:
            robot: Optional reference to the cognitive architecture / robot module.
                   If provided, run_trial() will call robot.run_episode(parameters).
                   If None, run_trial() uses heuristic evaluation.
            csv_file: Path to CSV experience file for persistence.
        """
        self.robot = robot
        self.memory = ReplayBuffer()
        self.evaluator = Evaluator()
        self.optimizer = Optimizer()
        self.csv_file = csv_file
        self.load_experience()

    # -------- CSV Handling --------
    def load_experience(self):
        """Load past experiences from CSV file"""
        if not os.path.exists(self.csv_file):
            try:
                os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
            except (OSError, ValueError):
                pass
            try:
                with open(self.csv_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    header = list(DEFAULT_PARAMETERS.keys()) + ["score"]
                    writer.writerow(header)
            except (OSError, PermissionError):
                pass
            return

        try:
            with open(self.csv_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    params = {k: float(v) for k, v in row.items() if k != "score"}
                    score = float(row["score"])
                    self.memory.add(params, score)
        except (OSError, ValueError, KeyError):
            pass

    def save_experience(self):
        """Save all experiences to CSV file"""
        experiences = self.memory.get_all()
        try:
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(list(DEFAULT_PARAMETERS.keys()) + ["score"])
                for params, score in experiences:
                    row = [params.get(k, 0.0) for k in DEFAULT_PARAMETERS.keys()] + [score]
                    writer.writerow(row)
        except (OSError, PermissionError):
            pass

    # -------- Trial Execution --------
    def run_trial(self, parameters):
        """
        Run a trial with given parameters.
        If robot is connected, calls robot.run_episode(parameters).
        Otherwise returns the parameters dict for heuristic evaluation.
        """
        if self.robot is not None and hasattr(self.robot, 'run_episode'):
            return self.robot.run_episode(parameters)
        return parameters

    def run_and_store(self, parameters):
        """Run a trial, evaluate, store experience, and return score"""
        result = self.run_trial(parameters)
        eval_result = self.evaluator.evaluate(result)

        # Handle both dict and float score returns
        if isinstance(eval_result, dict):
            score = eval_result.get("score", 0.0)
        else:
            score = float(eval_result)

        self.memory.add(parameters, score)
        self.save_experience()
        return score

    # -------- Online Learning --------
    def online_learning(self, episodes=30, initial_parameters=None):
        """Run online learning with trial-based parameter optimization"""
        params = initial_parameters or self.optimizer.random_parameters()
        scores = []
        for _ in range(episodes):
            score = self.run_and_store(params)
            scores.append(score)
            params = self.optimizer.get_best(self.memory)
        return scores

    # -------- Offline Learning (Replay) --------
    def offline_learning(self):
        """Use stored experiences to find best parameters without new trials"""
        experiences = self.memory.get_all()
        if not experiences:
            print("[Learning] No stored experience for offline learning")
            return [], None
        best = max(experiences, key=lambda x: x[1])
        scores = [exp[1] for exp in experiences]
        return scores, best[0]

    # -------- Baseline (No Learning) --------
    def baseline(self, episodes=30):
        """Run baseline episodes with random parameters (no optimization)"""
        scores = []
        for _ in range(episodes):
            params = self.optimizer.random_parameters()
            score = self.run_and_store(params)
            scores.append(score)
        return scores

    # -------- Get Current Best Parameters --------
    def get_best_parameters(self):
        """Return the best parameters found so far"""
        experiences = self.memory.get_all()
        if not experiences:
            return DEFAULT_PARAMETERS.copy()
        best = max(experiences, key=lambda x: x[1])
        return best[0]

    # -------- Plot Results --------
    def plot_results(self, baseline_scores, online_scores, offline_scores=None):
        """Plot comparison of learning strategies"""
        if not MATPLOTLIB_AVAILABLE:
            print("[Learning] matplotlib not available, skipping plot")
            return
        plt.figure(figsize=(10, 6))
        plt.plot(baseline_scores, label="Baseline (No Learning)")
        plt.plot(online_scores, label="Online Learning")
        if offline_scores:
            plt.plot(offline_scores, label="Offline Replay")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.title("Learning Module Performance Comparison")
        plt.legend()
        plt.show()


# =========================
# Main Test
# =========================
if __name__ == "__main__":
    learner = Learner()

    # Baseline
    baseline_scores = learner.baseline(episodes=30)

    # Online Learning
    online_scores = learner.online_learning(episodes=30, initial_parameters=DEFAULT_PARAMETERS)

    # Offline Learning
    offline_scores, best_params = learner.offline_learning()
    print(f"Best offline parameters: {best_params}")

    # Plot comparison
    learner.plot_results(baseline_scores, online_scores, offline_scores)
