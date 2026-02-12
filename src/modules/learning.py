import random
from collections import deque
import csv
import os
import matplotlib.pyplot as plt

# Define robot parameters for learning
DEFAULT_PARAMETERS = {
    # Navigation PID gains
    "nav_kp": 1.0,
    "nav_ki": 0.05,
    "nav_kd": 0.01,
    
    # Arm/Gripper PID gains
    "arm_kp": 1.2,
    "arm_ki": 0.0,
    "arm_kd": 0.05,
    
    # Vision thresholds
    "vision_threshold": 0.5,
    
    # Robot speed limits
    "max_linear_speed": 0.5,
    "max_angular_speed": 1.0
}

# =========================
# Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, capacity=100):
        self.buffer = deque(maxlen=capacity)

    def add(self, parameters, score):
        self.buffer.append((parameters, score))

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
        Replace with actual performance metrics later
        """
        # Example: pretend that higher nav_kp and arm_kp slightly improve the score
        score = 0.5  # base score
        score += 0.1 * min(simulation_result.get("nav_kp", 1.0), 2.0)
        score += 0.1 * min(simulation_result.get("arm_kp", 1.0), 2.0)
        score += random.uniform(-0.05, 0.05)  # simulate noise

        # result = robot_module.run_with_parameters(parameters)

        # Return result for evaluator (can be more complex later)
        return {"score": score}

# =========================
# Optimizer
# =========================
class Optimizer:
    def get_best(self, memory):
        experiences = memory.get_all()
        if not experiences:
            return self.random_parameters()

        best = max(experiences, key=lambda x: x[1])
        best_params = best[0]

        return self.mutate(best_params)

    def mutate(self, params):
        new_params = {}
        for key, value in params.items():
            new_params[key] = value + random.uniform(-0.05, 0.05)
        return new_params

    def random_parameters(self):
        return {"kp": random.uniform(0.5,2.0),
                "ki": random.uniform(0.0,1.0),
                "kd": random.uniform(0.0,1.0)}

# =========================
# Learner
# =========================
class Learner:
    def __init__(self, csv_file="data/experiences.csv"):
        self.memory = ReplayBuffer()
        self.evaluator = Evaluator()
        self.optimizer = Optimizer()
        self.csv_file = csv_file
        self.load_experience()

    # -------- Load / Save CSV Experience --------
    def load_experience(self):
        if not os.path.exists(self.csv_file):
            os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
            with open(self.csv_file,"w",newline="") as f:
                writer = csv.writer(f)
                header = list(DEFAULT_PARAMETERS.keys()) + ["score"]
                writer.writerow(header)
            return

        with open(self.csv_file,"r",newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert all numeric values except 'score'
                params = {k: float(v) for k,v in row.items() if k != "score"}
                score = float(row["score"])
                self.memory.add(params, score)


    def save_experience(self):
        with open(self.csv_file,"w",newline="") as f:
            writer = csv.writer(f)
            # header
            if self.memory.size() > 0:
                first_params = self.memory.get_all()[0][0]
                header = list(first_params.keys()) + ["score"]
            else:
                header = list(DEFAULT_PARAMETERS.keys()) + ["score"]
            writer.writerow(header)
            # write rows
            for params, score in self.memory.get_all():
                row = [params[key] for key in first_params.keys()] + [score]
                writer.writerow(row)


    # -------- Run Trial (Interface with Robot Module) --------
    def run_trial(self, parameters):
        """
        Replace with actual robot module
        """

        # Call the robot module’s function that actually runs a trial
        # result = robot_module.run_with_parameters(parameters)
        # return result
        result = parameters 
        return result

    def run_and_store(self, parameters):
        result = self.run_trial(parameters)
        score = self.evaluator.evaluate(result)  # evaluate once
        self.memory.add(parameters, score)
        self.save_experience()
        return score

    # -------- Online Learning --------
    def online_learning(self, episodes=30, initial_parameters=None):
        if initial_parameters:
            params = initial_parameters
        else:
            params = self.optimizer.random_parameters()
        scores = []

        for ep in range(episodes):
            score = self.run_and_store(params)
            scores.append(score)
            params = self.optimizer.get_best(self.memory)
        return scores

    # -------- Offline Learning (Replay) --------
    def offline_learning(self):
        """
        Use stored experiences to find best parameters without new trials
        """
        experiences = self.memory.get_all()
        if not experiences:
            print("No experiences available for offline learning")
            return []
        # Pick best-so-far parameters repeatedly
        best_exp = max(experiences, key=lambda x: x[1])  # (parameters, score)
        best_params = best_exp[0]                        # get parameters dict
        scores = [exp[1] for exp in experiences]
        return scores, best_params

    # -------- Baseline (No Learning) --------
    def baseline(self, episodes=30):
        scores = []
        for _ in range(episodes):
            params = self.optimizer.random_parameters()
            score = self.run_and_store(params)
            scores.append(score)
        return scores

    # -------- Plot Results --------
    def plot_results(self, baseline_scores, online_scores, offline_scores):
        plt.figure(figsize=(10,6))
        plt.plot(range(len(baseline_scores)), baseline_scores, label="Baseline (No Learning)")
        plt.plot(range(len(online_scores)), online_scores, label="Online Learning")
        if offline_scores:
            plt.plot(range(len(offline_scores)), offline_scores, label="Offline Learning")
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

    # Offline Learning (from stored experiences)
    offline_scores = learner.offline_learning()

    # Plot comparison
    learner.plot_results(baseline_scores, online_scores, offline_scores)
