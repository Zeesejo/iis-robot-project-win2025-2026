"""
M9 - Learning Module
Online and offline parameter optimization.
Tracks success/failure of grasps and navigation, tunes:
  - PID gains (kp, ki, kd) for heading and distance controllers
  - Vision detection thresholds
Uses a simple hill-climbing / experience replay approach.
"""

import numpy as np
import json
import os

PARAMS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'executables', 'learned_params.json')


class ExperienceBuffer:
    """Stores trial outcomes for offline learning."""

    def __init__(self, maxlen=100):
        self.buffer = []
        self.maxlen = maxlen

    def add(self, params, outcome):
        """
        params: dict of parameter names -> values
        outcome: float in [0, 1] — logistic-squashed trial score.
        """
        self.buffer.append({'params': params.copy(), 'outcome': float(outcome)})
        if len(self.buffer) > self.maxlen:
            self.buffer.pop(0)

    def best_params(self):
        """Return params from the trial with the highest outcome."""
        if not self.buffer:
            return None
        return max(self.buffer, key=lambda e: e['outcome'])['params']

    def mean_outcome(self):
        if not self.buffer:
            return 0.0
        return float(np.mean([e['outcome'] for e in self.buffer]))


class OnlinePIDTuner:
    """
    Hill-climbing online tuner for PID gains.
    After each trial, perturbs gains slightly and keeps the better set.
    """

    def __init__(self, initial_gains, perturbation=0.05, decay=0.99):
        """
        initial_gains: dict {'kp': float, 'ki': float, 'kd': float}
        """
        self.current = initial_gains.copy()
        self.best = initial_gains.copy()
        self.best_score = -np.inf
        self.perturbation = perturbation
        self.decay = decay
        self._trial_gains = None

    def suggest_gains(self):
        """Propose a perturbed set of gains for the next trial."""
        self._trial_gains = {}
        for k, v in self.current.items():
            delta = np.random.uniform(-self.perturbation, self.perturbation) * v
            self._trial_gains[k] = max(1e-4, v + delta)
        return self._trial_gains.copy()

    def report_outcome(self, score):
        """
        score: float (raw, can be negative) — higher is better.
        Updates best gains if improved.
        """
        if self._trial_gains is None:
            return
        if score > self.best_score:
            self.best_score = score
            self.best = self._trial_gains.copy()
            self.current = self._trial_gains.copy()
            print(f"[Tuner] New best PID gains: {self.best} (score={score:.4f})")
        self.perturbation *= self.decay  # Reduce exploration over time

    def get_best(self):
        return self.best.copy()


class VisionThresholdTuner:
    """
    Tunes color detection tolerance based on detection success.
    If detection fails too often, widen tolerance; if false positives, tighten.
    """

    def __init__(self, initial_tol=0.25, step=0.02, limits=(0.1, 0.5)):
        self.tol = initial_tol
        self.step = step
        self.limits = limits
        self._false_pos = 0
        self._missed = 0
        self._total = 0

    def report(self, detected, correct):
        """
        detected: bool (did we detect something?)
        correct: bool (was the detection correct?)
        """
        self._total += 1
        if detected and not correct:
            self._false_pos += 1
        if not detected and correct:
            self._missed += 1

        if self._total % 20 == 0:
            miss_rate = self._missed / self._total
            fp_rate = self._false_pos / self._total
            if miss_rate > 0.3:
                self.tol = min(self.tol + self.step, self.limits[1])
                print(f"[VisionTuner] Widening tolerance to {self.tol:.3f}")
            elif fp_rate > 0.2:
                self.tol = max(self.tol - self.step, self.limits[0])
                print(f"[VisionTuner] Tightening tolerance to {self.tol:.3f}")
            self._false_pos = 0
            self._missed = 0
            self._total = 0

    def get_tolerance(self):
        return self.tol


class LearningModule:
    """
    Unified learning module: wraps PID tuner + vision tuner + experience buffer.
    Persists learned parameters across runs.
    """

    def __init__(self):
        default_heading_gains = {'kp': 3.0, 'ki': 0.01, 'kd': 0.5}
        default_distance_gains = {'kp': 2.0, 'ki': 0.001, 'kd': 0.3}

        # Try to load previously learned parameters
        saved = self._load_params()
        if saved:
            default_heading_gains = saved.get('heading_pid', default_heading_gains)
            default_distance_gains = saved.get('distance_pid', default_distance_gains)
            print("[Learning] Loaded previously learned parameters.")

        self.heading_tuner = OnlinePIDTuner(default_heading_gains)
        self.distance_tuner = OnlinePIDTuner(default_distance_gains)
        self.vision_tuner = VisionThresholdTuner()
        self.experience = ExperienceBuffer()

    def start_trial(self):
        """Call at start of each episode. Returns suggested PID gains."""
        h_gains = self.heading_tuner.suggest_gains()
        d_gains = self.distance_tuner.suggest_gains()
        return {'heading_pid': h_gains, 'distance_pid': d_gains}

    def end_trial(self, success, distance_remaining, collision_count):
        """
        Call at end of each episode with outcome metrics.
        success: bool
        distance_remaining: float (lower is better)
        collision_count: int

        Raw score can be negative (penalised by distance + collisions).
        We store a logistic-squashed version in [0, 1] for the ExperienceBuffer
        so that best_params() comparisons remain well-defined.
        """
        raw_score = (1.0 if success else 0.0) - 0.1 * distance_remaining - 0.2 * collision_count

        # Report raw score to PID tuners (they track sign correctly)
        self.heading_tuner.report_outcome(raw_score)
        self.distance_tuner.report_outcome(raw_score)

        # Normalise to [0, 1] for experience buffer (logistic squash)
        outcome = 1.0 / (1.0 + np.exp(-raw_score))

        params = {
            'heading_pid': self.heading_tuner.get_best(),
            'distance_pid': self.distance_tuner.get_best(),
            'vision_tol': self.vision_tuner.get_tolerance()
        }
        self.experience.add(params, outcome)
        self._save_params(params)
        print(f"[Learning] Trial ended — raw_score={raw_score:.4f}, outcome={outcome:.4f}")
        return params

    def get_current_params(self):
        return {
            'heading_pid': self.heading_tuner.get_best(),
            'distance_pid': self.distance_tuner.get_best(),
            'vision_tol': self.vision_tuner.get_tolerance()
        }

    def _save_params(self, params):
        os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)
        with open(PARAMS_PATH, 'w') as f:
            json.dump(params, f, indent=2)

    def _load_params(self):
        if os.path.exists(PARAMS_PATH):
            with open(PARAMS_PATH) as f:
                return json.load(f)
        return None
