import numpy as np
import torch
from env.fssp_env import FSSPEnvironment
from agent.dqn_agent import DQNAgent


def evaluate_agent(agent, test_instances):
    total_rewards = []
    for jobs, machines in test_instances:
        env = FSSPEnvironment(jobs, machines)
        state = env.reset()
        state = state.flatten()
        done = False
        total_reward = 0
        metaheuristic_info = None

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = next_state.flatten()
            state = next_state
            total_reward += reward
            metaheuristic_info = info

        total_rewards.append(total_reward)
        print(
            f"Metaheuristic: {metaheuristic_info['metaheuristic']}, Hyperparameters: {metaheuristic_info['hyperparameters']}, Makespan: {metaheuristic_info['makespan']}")

    average_reward = - np.max(total_rewards)
    return average_reward


# Example: Create test instances
test_instances = [(np.array([[26, 59, 78, 88, 69],
                             [38, 62, 90, 54, 30],
                             [27, 44, 64, 47, 61],
                             [88, 10, 49, 83, 35],
                             [95, 23, 47, 84, 53],
                             [55, 64, 20,  9, 98],
                             [54, 47, 61, 30, 94],
                             [63, 68, 93, 11, 33],
                             [23, 54, 36, 92, 77],
                             [45,  9, 47, 63, 31],
                             [86, 30, 70, 62, 54],
                             [43, 31, 54, 75, 71],
                             [43, 92, 87, 48, 78],
                             [40,  7, 13, 23,  9],
                             [37, 14, 40, 85, 79],
                             [54, 95, 34, 23, 51],
                             [35, 76, 55,  4, 76],
                             [59, 82, 13, 31, 56],
                             [43, 91, 11, 13, 80],
                             [50, 37,  5, 98, 72]]), [1, 2, 3, 4, 5]) for _ in range(10)]

# Initialize environment to get state and action dimensions
jobs, machines = test_instances[0]
env = FSSPEnvironment(jobs, machines)
state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
action_dim = env.action_space.n

# Initialize the agent
agent = DQNAgent(state_dim, action_dim)

# Load the trained model
agent.model.load_state_dict(torch.load("dqn_fssp_model.pth"))
print("Model loaded from dqn_fssp_model.pth")

# Evaluate the agent
average_reward = evaluate_agent(agent, test_instances)
print(f'Average Reward on Test Instances: {average_reward}')
