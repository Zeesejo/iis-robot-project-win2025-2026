"""
Module 9: Learning
Parameter optimization via replay buffer, evaluation, and mutation.
Supports offline learning (replay) and proper experience logging.
"""

import random
from collections import deque
import csv
import os
import logging

logger = logging.getLogger(__name__)

# =========================
# Default Parameters
# =========================
DEFAULT_PARAMETERS = {
    # Navigation PID gains
    "nav_kp": 1.0,
    "nav_ki": 0.0,
    "nav_kd": 0.1,
    # Angle/heading PID gain
    "angle_kp": 1.0,
    # Approach speed limit
    "approach_speed": 5.0,
    # Grasp force
    "grasp_force": 200.0,
}


# =========================
# Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, capacity=200):
        self.buffer = deque(maxlen=capacity)

    def add(self, parameters, score, success):
        self.buffer.append((parameters.copy(), float(score), bool(success)))

    def get_all(self):
        return list(self.buffer)

    def __len__(self):
        return len(self.buffer)


# =========================
# Evaluator
# =========================
class Evaluator:
    def evaluate(self, simulation_result):
        """
        Score an episode result.
        success gives +1000, each step costs -0.1.
        Partial credit for getting close (collision-free steps bonus).
        """
        if not isinstance(simulation_result, dict):
            return 0.0

        score = 0.0
        success = simulation_result.get("success", False)
        steps   = simulation_result.get("steps", 0)
        collisions = simulation_result.get("collisions", 0)

        if success:
            score += 1000.0
        # Time penalty
        score -= steps * 0.1
        # Collision penalty
        score -= collisions * 50.0
        # Partial reward: distance_remaining (lower = better)
        dist_rem = simulation_result.get("distance_remaining", None)
        if dist_rem is not None and not success:
            score += max(0.0, (5.0 - dist_rem) * 20.0)

        return float(score)


# =========================
# Learner
# =========================
class Learner:
    def __init__(self, csv_file="data/experiences.csv"):
        self.memory    = ReplayBuffer()
        self.evaluator = Evaluator()
        self.csv_file  = csv_file
        self._ensure_dir()
        self.load_experience()

    def _ensure_dir(self):
        d = os.path.dirname(self.csv_file)
        if d:
            os.makedirs(d, exist_ok=True)

    # ---- CSV Handling ----

    def load_experience(self):
        """Load past experiences from CSV."""
        if not os.path.exists(self.csv_file):
            self._write_header()
            logger.info("[Learning] Created new experience file: %s", self.csv_file)
            print(f"[Learning] Created new experience file: {self.csv_file}")
            return

        loaded = 0
        try:
            with open(self.csv_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        params = {k: float(v) for k, v in row.items()
                                  if k not in ("score", "success")}
                        score   = float(row.get("score", 0))
                        success = bool(int(row.get("success", "0")))
                        self.memory.add(params, score, success)
                        loaded += 1
                    except (ValueError, KeyError):
                        pass
        except Exception as e:
            logger.warning("[Learning] Could not load %s: %s", self.csv_file, e)
        msg = f"[Learning] Loaded {loaded} past experiences from {self.csv_file}"
        print(msg)
        logger.info(msg)

    def _write_header(self):
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(DEFAULT_PARAMETERS.keys()) + ["score", "success"])

    def save_experience(self):
        """Persist all experiences to CSV."""
        experiences = self.memory.get_all()
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(DEFAULT_PARAMETERS.keys()) + ["score", "success"])
            for params, score, success in experiences:
                row = [params.get(k, DEFAULT_PARAMETERS.get(k, 0.0))
                       for k in DEFAULT_PARAMETERS.keys()]
                row += [round(score, 4), int(success)]
                writer.writerow(row)
        msg = (f"[Learning] Saved {len(experiences)} experiences to {self.csv_file}")
        print(msg)
        logger.info(msg)

    # ---- Offline Learning (Replay) ----

    def offline_learning(self):
        """
        Pick best parameters from stored experiences.
        Falls back to defaults if no successes recorded yet.
        Returns (scores_list, best_params_dict).
        """
        experiences = self.memory.get_all()
        scores = [exp[1] for exp in experiences]

        if not experiences:
            print("[Learning] No stored experience — using defaults.")
            return [], DEFAULT_PARAMETERS.copy()

        successes = [exp for exp in experiences if exp[2]]
        if not successes:
            # Best failure so far
            best = max(experiences, key=lambda x: x[1])
            msg = (f"[Learning] No successful run yet — using best-scoring params "
                   f"(score={best[1]:.1f}). Defaults merged.")
            print(msg)
            logger.info(msg)
            merged = DEFAULT_PARAMETERS.copy()
            merged.update(best[0])
            return scores, merged

        best = max(successes, key=lambda x: x[1])
        msg = (f"[Learning] Best successful params chosen "
               f"(score={best[1]:.1f}, "
               f"from {len(successes)} successes / {len(experiences)} total runs).")
        print(msg)
        logger.info(msg)
        merged = DEFAULT_PARAMETERS.copy()
        merged.update(best[0])
        return scores, merged

    def record_episode(self, params, result):
        """
        Convenience: evaluate + add + save in one call.
        Call after every episode.
        """
        score   = self.evaluator.evaluate(result)
        success = bool(result.get("success", False))
        self.memory.add(params, score, success)
        self.save_experience()
        print(f"[Learning] Episode recorded: success={success}, "
              f"score={score:.1f}, buffer_size={len(self.memory)}")
        logger.info("[Learning] Episode: success=%s score=%.1f steps=%s",
                    success, score, result.get('steps', '?'))
        return score


# =========================
# Main Test
# =========================
if __name__ == "__main__":
    learner = Learner()
    offline_scores, best_params = learner.offline_learning()
    print(f"Best offline parameters: {best_params}")
    # Simulate recording a run
    learner.record_episode(best_params, {"success": True, "steps": 1200, "collisions": 0})
