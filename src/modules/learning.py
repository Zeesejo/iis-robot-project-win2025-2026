"""
Module 9: Learning
Parameter optimization via replay buffer, evaluation, and mutation.
Supports offline learning (replay).
"""

import random
from collections import deque
import csv
import os

# =========================
# Default Parameters (tuned for this robot)
# =========================
DEFAULT_PARAMETERS = {
    # Navigation PID gains
    "nav_kp": 1.0,
    "nav_ki": 0.0,
    "nav_kd": 0.1,

    # Angle/heading PID gains
    "angle_kp": 1.0,
}


# =========================
# Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, capacity=100):
        self.buffer = deque(maxlen=capacity)

    def add(self, parameters, score, success):
        self.buffer.append((parameters.copy(), float(score), bool(success)))

    def get_all(self):
        return list(self.buffer)


# =========================
# Evaluator
# =========================
class Evaluator:
    def evaluate(self, simulation_result):
        if isinstance(simulation_result, dict) and 'success' in simulation_result:
            score = 0
            if simulation_result.get("success", False):
                score += 1000
            score -= simulation_result.get("steps", 0) * 0.1
            return float(score)
        else:
            score = 0.5
            if isinstance(simulation_result, dict):
                score += 0.1 * min(simulation_result.get("nav_kp", 1.0), 2.0)
            score += random.uniform(-0.05, 0.05)
            return float(score)

# =========================
# Learner
# =========================
class Learner:
    def __init__(self, csv_file="data/experiences.csv"):
        self.memory = ReplayBuffer()
        self.evaluator = Evaluator()
        self.csv_file = csv_file
        self.load_experience()

    # -------- CSV Handling --------
    def load_experience(self):
        if not os.path.exists(self.csv_file):
            os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(list(DEFAULT_PARAMETERS.keys()) + ["score", "success"])
            return

        with open(self.csv_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                params = {k: float(v) for k, v in row.items()
                        if k not in ("score", "success")}
                score = float(row["score"])
                success = bool(int(row.get("success", "0")))
                self.memory.add(params, score, success)

    def save_experience(self):
        experiences = self.memory.get_all()
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(DEFAULT_PARAMETERS.keys()) + ["score", "success"])
            for params, score, success in experiences:
                row = [params.get(k, 0.0) for k in DEFAULT_PARAMETERS.keys()]
                row += [score, int(success)]
                writer.writerow(row)

    # -------- Offline Learning (Replay) --------
    def offline_learning(self):
        """
        Use stored experiences to pick best parameters.
        - If no experience at all: use defaults.
        - If only failures: use defaults.
        - Else: best among successful runs.
        """
        experiences = self.memory.get_all()
        if not experiences:
            print("[Learning] No stored experience at all; using defaults.")
            return [], DEFAULT_PARAMETERS.copy()

        successes = [exp for exp in experiences if exp[2]]  # (params, score, success)
        if not successes:
            print("[Learning] No successful runs yet; using defaults.")
            scores = [exp[1] for exp in experiences]
            return scores, DEFAULT_PARAMETERS.copy()

        best = max(successes, key=lambda x: x[1])
        scores = [exp[1] for exp in experiences]
        return scores, best[0]

# =========================
# Main Test
# =========================
if __name__ == "__main__":
    learner = Learner()

    # Offline Learning
    offline_scores, best_params = learner.offline_learning()
    print(f"Best offline parameters: {best_params}")