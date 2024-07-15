from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from agent.dqn_agent import DQNAgent
from env.fssp_env import FSSPEnvironment
import torch
import numpy as np
from tkinter import ttk
import tkinter as tk
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class TrainingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FSSP Training Simulation")

        self.setup_ui()

        self.jobs = np.array([[26, 59, 78, 88, 69],
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
                              [50, 37,  5, 98, 72]])
        self.machines = [1, 2, 3, 4, 5]
        self.env = FSSPEnvironment(self.jobs, self.machines)

        state_dim = self.env.observation_space.shape[0] * \
            self.env.observation_space.shape[1]
        action_dim = self.env.action_space.n
        self.agent = DQNAgent(state_dim, action_dim)

        self.num_episodes = 200
        self.episode_rewards = []

    def setup_ui(self):
        self.start_button = ttk.Button(
            self.root, text="Start Training", command=self.start_training)
        self.start_button.pack(pady=10)

        self.checkpoint_label = ttk.Label(
            self.root, text="Checkpoint Interval:")
        self.checkpoint_label.pack(pady=5)

        self.checkpoint_entry = ttk.Entry(self.root)
        self.checkpoint_entry.insert(0, "50")  # Default value
        self.checkpoint_entry.pack(pady=5)

        self.progress_label = ttk.Label(self.root, text="Progress: 0%")
        self.progress_label.pack(pady=10)

        self.progress_bar = ttk.Progressbar(
            self.root, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.pack(pady=10)

        self.figure, self.ax = plt.subplots()
        self.ax.set_title("Training Progress")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Total Reward")

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(pady=10)

    def start_training(self):
        self.start_button.config(state=tk.DISABLED)
        self.root.after(100, self.train)

    def train(self):
        checkpoint_interval = int(self.checkpoint_entry.get())

        for episode in range(self.num_episodes):
            state = self.env.reset()
            state = state.flatten()
            done = False
            total_reward = 0

            while not done:
                action = self.agent.act(state)
                next_state, reward, done, info = self.env.step(action)
                next_state = next_state.flatten()
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            self.agent.replay()
            self.episode_rewards.append(total_reward)

            if (episode + 1) % checkpoint_interval == 0:
                self.update_progress(episode + 1)
                self.update_plot()
                # self.save_model(f"dqn_fssp_model_{episode + 1}.pth")

        self.save_model("dqn_fssp_model_final.pth")
        self.start_button.config(state=tk.NORMAL)

    def update_progress(self, episode):
        progress = (episode / self.num_episodes) * 100
        self.progress_label.config(text=f"Progress: {progress:.2f}%")
        self.progress_bar['value'] = progress
        self.root.update_idletasks()

    def update_plot(self):
        self.ax.clear()
        self.ax.plot(self.episode_rewards)
        self.ax.set_title("Training Progress")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Total Reward")
        self.canvas.draw()

    def save_model(self, filename):
        torch.save(self.agent.model.state_dict(), filename)
        print(f"Model saved as {filename}")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop()
