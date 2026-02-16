import random
from collections import deque
import csv
import os
import matplotlib.pyplot as plt
import sys

# Add project root to sys.path so Python can find 'executables'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# from executables.cognitive_architecture import CognitiveArchitecture, build_world


# =========================
# Default Parameters
# =========================
DEFAULT_PARAMETERS = {
    "nav_kp": 2.5,      # strong enough to move
    "nav_ki": 0,
    "nav_kd": 0.5,     # small derivative

    "angle_kp": 3.0,    # moderate turn
    "angle_ki": 0.0,
    "angle_kd": 0.01,

    "arm_kp": 1.2,
    "arm_ki": 0.0,
    "arm_kd": 0.05,
    
    "vision_threshold": 0.5,
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
        score = 0
        if simulation_result["success"]:
            score += 1000
        score -= simulation_result["steps"] * 0.1
        return score




# =========================
# Optimizer
# =========================
class Optimizer:

    def random_parameters(self):
        return {
            key: value + random.uniform(-0.3, 0.3)
            for key, value in DEFAULT_PARAMETERS.items()
        }

    def mutate(self, params, scale=0.05):
        return {k: v + random.uniform(-scale, scale) for k,v in params.items()}


    def get_best(self, memory):
        experiences = memory.get_all()

        if not experiences:
            return self.random_parameters()

        best = max(experiences, key=lambda x: x[1])
        return self.mutate(best[0])


# =========================
# Learner
# =========================
class Learner:

    def __init__(self, robot, csv_file="data/experiences.csv"):
        self.robot = robot     # <-- THIS LINE (connection to architecture)

        self.memory = ReplayBuffer()
        self.evaluator = Evaluator()
        self.optimizer = Optimizer()
        self.csv_file = csv_file
        self.load_experience()



    # -------- CSV Handling --------
    def load_experience(self):

        if not os.path.exists(self.csv_file):

            os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)

            with open(self.csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(list(DEFAULT_PARAMETERS.keys()) + ["score"])
            return

        with open(self.csv_file, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                params = {k: float(v) for k, v in row.items() if k != "score"}
                score = float(row["score"])
                self.memory.add(params, score)


    def save_experience(self):

        experiences = self.memory.get_all()

        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(list(DEFAULT_PARAMETERS.keys()) + ["score"])

            for params, score in experiences:
                row = [params[k] for k in DEFAULT_PARAMETERS.keys()] + [score]
                writer.writerow(row)


    # -------- Trial Execution --------
    def run_trial(self, parameters):
        """
        Later connect to robot module here
        """
        
        return self.robot.run_episode(parameters)



    def run_and_store(self, parameters):

        result = self.run_trial(parameters)
        score = self.evaluator.evaluate(result)

        self.memory.add(parameters, score)
        self.save_experience()

        return score


    # -------- Baseline --------
    def baseline(self, episodes=30):

        scores = []

        for _ in range(episodes):
            params = self.optimizer.random_parameters()
            scores.append(self.run_and_store(params))

        return scores


    # -------- Online Learning --------
    def online_learning(self, episodes=30, initial_parameters=None):

        params = initial_parameters or self.optimizer.random_parameters()
        scores = []

        for _ in range(episodes):
            score = self.run_and_store(params)
            scores.append(score)
            params = self.optimizer.get_best(self.memory)

        return scores


    # -------- Offline Learning --------
    def offline_learning(self):

        experiences = self.memory.get_all()

        if not experiences:
            print("No stored experience")
            return [], None

        best = max(experiences, key=lambda x: x[1])
        scores = [exp[1] for exp in experiences]

        return scores, best[0]


    # -------- Plot --------
    def plot_results(self, baseline_scores, online_scores, offline_scores):

        plt.figure(figsize=(10,6))

        plt.plot(baseline_scores, label="Baseline")
        plt.plot(online_scores, label="Online Learning")

        if offline_scores:
            plt.plot(offline_scores, label="Offline Replay")

        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.legend()
        plt.show()


# =========================
# MAIN TEST
# =========================
# if __name__ == "__main__":

#     robot_id, table_id, room_id, target_id = build_world(gui=False)
#     cog_arch = CognitiveArchitecture(robot_id, table_id, room_id, target_id)

#     learner = Learner(robot=cog_arch)

#     baseline_scores = learner.baseline(2)
#     online_scores = learner.online_learning(2, DEFAULT_PARAMETERS)
#     offline_scores, best_params = learner.offline_learning()

#     print("\nBest offline parameters:", best_params)

#     learner.plot_results(baseline_scores, online_scores, offline_scores)

