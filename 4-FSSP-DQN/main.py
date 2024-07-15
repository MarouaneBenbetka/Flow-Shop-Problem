import numpy as np
import torch
from env.fssp_env import FSSPEnvironment
from agent.dqn_agent import DQNAgent

# Initialize environment and agent
jobs = np.array([[54, 79, 16, 66, 58],
                 [83,  3, 89, 58, 56],
                 [15, 11, 49, 31, 20],
                 [71, 99, 15, 68, 85],
                 [77, 56, 89, 78, 53],
                 [36, 70, 45, 91, 35],
                 [53, 99, 60, 13, 53],
                 [38, 60, 23, 59, 41],
                 [27,  5, 57, 49, 69],
                 [87, 56, 64, 85, 13],
                 [76,  3,  7, 85, 86],
                 [91, 61,  1,  9, 72],
                 [14, 73, 63, 39,  8],
                 [29, 75, 41, 41, 49],
                 [12, 47, 63, 56, 47],
                 [77, 14, 47, 40, 87],
                 [32, 21, 26, 54, 58],
                 [87, 86, 75, 77, 18],
                 [68,  5, 77, 51, 68],
                 [94, 77, 40, 31, 28]])
# jobs = np.random.rand(10, 5)  # Example: 10 jobs, 5 machines
machines = [1, 2, 3, 4, 5]
env = FSSPEnvironment(jobs, machines)

state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim)
num_episodes = 300

# Hyperparameter sets
hyperparameter_sets = [
    {"population_size": 10, "mutation_rate": 0.1, "generations": 50},
    {"initial_temperature": 100, "cooling_rate": 0.95, "max_iterations": 1000},
    {"tabu_list_size": 10, "max_iterations": 100},
    # Add more hyperparameter sets as needed
]

for episode in range(num_episodes):
    state = env.reset()
    state = state.flatten()
    done = False
    total_reward = 0
    metaheuristic_info = None

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = next_state.flatten()
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        metaheuristic_info = info

    agent.replay()

    if (episode + 1) % 30 == 0:
        print(
            f'Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}')
        print(
            f"Metaheuristic: {metaheuristic_info['metaheuristic']}, Hyperparameters: {metaheuristic_info['hyperparameters']}, Makespan: {metaheuristic_info['makespan']}")

# Save the trained model
torch.save(agent.model.state_dict(), "dqn_fssp_model.pth")
print("Model saved as dqn_fssp_model.pth")
