import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# env = gym.make("CartPole-v1", render_mode="rgb_array")
env = gym.make("CartPole-v1", render_mode="human")


class Agent:
    def __init__(self, action_space, n_states, epsilon=0.5, gamma=0.9):
        self.epsilon = epsilon  # 初始的 探索率
        self.gamma = gamma    # 折扣因子

        self.action_space = action_space  # 动作空间
        self.n_states = n_states  # 状态空间
        self.q_table = np.zeros((n_states, len(action_space)))  # Q表

    def choose_action(self, state, episode):
        epsilon = self.epsilon * np.exp(-episode)  # epsilon 随着 episode 衰减
        if np.random.random() < epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = np.argmax(self.q_table[state, :])
        return action

    def q_learning(self):
        pass




done = False
state = env.reset(seed=123)
score = 0
frames = []
for episode in range(2):
    done = False
    score = 0
    state = env.reset(seed=123)
    frames = []
    while not done:
        frames.append(env.render())  # 记录每一帧的图像
        action = env.action_space.sample()  # 随机选择一个动作
        observation, reward, terminated, truncated, info = env.step(action)
        score += reward
        done = terminated or truncated
