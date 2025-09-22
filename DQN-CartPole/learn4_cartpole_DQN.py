
import gymnasium as gym
import numpy as np
from collections import namedtuple
import random

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os



# 定义一个Transition元组，用于存储状态、动作、下一个状态和奖励
# s_t, a_t => s_{t+1}, r
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))



# 记忆回放
# 构造批次化训练数据，让训练更稳定
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None) # 增加占位，即预先分配空间，方便后续索引
        self.memory[self.index] = transition
        self.index = (self.index + 1) % self.capacity # 循环队列的指针循环（有限数生成的循环群，阶数为capacity）

    def sample(self, batch_size):
        # return np.random.choice(self.memory, batch_size, replace=False) # 要求得用np.array格式，不如直接用random.sample
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class QNet(nn.Module):
    def __init__(self, n_states, n_actions):
        super(QNet, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        self.model = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
    def forward(self, x):
        # x.shape: batch_size * n_states
        # out.shape: batch_size * n_actions
        # print("Input shape:", x.shape)  # 调试输入维度
        return self.model(x)



class Agent:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states  # 状态空间
        self.n_actions = n_actions  # 动作空间
        self.gamma = 0.8    # 折扣因子
        self.lr = 0.01

        self.epsilon_start = 0.5  # 初始的 探索率
        self.epsilon_end = 0.01  # 最终的 探索率
        self.epsilon_decay = 200  # 探索率衰减的步数

        self.batch_size = 32
        self.capacity = 10000  # 记忆回放的容量
        self.memory = ReplayMemory(self.capacity)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNet(n_states, n_actions).to(self.device)
        self.target_net = QNet(n_states, n_actions).to(self.device)  # 新增目标网络

        self.model_path = "./models/CartPole_QNet.pth"
        if os.path.exists(self.model_path):
            self.load_policy_net_from(self.model_path)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()
        self.update_counter = 0  # 计数器，用于控制目标网络的更新频率
        self.update_frequency = 10  # 目标网络更新的频率

    def choose_action(self, state, episode=0):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-episode/self.epsilon_decay)  # epsilon 随着 episode 衰减
        if np.random.random() < epsilon:
            action = np.random.choice(self.n_actions)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0) # 增加 batch_size 维度
                q_values = self.policy_net(state) # 得到每个状态的 q_values
                action = q_values.argmax().item() # 选择最大的 q_values 对应的 action
        return int(action)

    def memorize(self, state, action, next_state, reward):
        transition = Transition(state, action, next_state, reward)
        self.memory.push(transition)

    def update_q_function(self):
        self._replay()

    def _replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*batch)) # 解包，对应成元组后，再打包

        # 转为 tensor
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(np.array(batch.action), dtype=torch.int64).to(self.device)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(self.device)
        reward_batch = torch.tensor(np.array(batch.reward), dtype=torch.float32).to(self.device)

        # 计算当前的 Q(s_t, a_t)
        current_q = self.policy_net(state_batch) # 返回维度： batch_size, n_actions
        current_q = current_q.gather(1, action_batch.unsqueeze(1)) # 选取所做的动作对应的Q值，维度： batch_size, 1

        # 计算下一状态的 Q(s_{t+1}, a_{t+1})
        next_q = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q = reward_batch + self.gamma * next_q

        loss = self.criterion(current_q, expected_q.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 定期更新目标网络
        self.update_counter += 1
        if self.update_counter % self.update_frequency == 0:  # 改进点7：目标网络更新[4](@ref)
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def load_policy_net_from(self, model_path):
        self.policy_net.load_state_dict(torch.load(model_path))
        print(f"policy_net loaded from: {model_path}")

    def save_policy_net_to(self, model_path):
        torch.save(self.policy_net.state_dict(), model_path)
        print(f"policy_net saved to: {model_path}")


def get_reward(state, step):
    """动态奖励函数设计"""
    x, x_dot, theta, theta_dot = state
    angle_penalty = abs(theta) / 0.42  # 0.42≈24°弧度
    velocity_bonus = 0.1 * (0.5 - abs(theta_dot) / 1.0)

    # 基础奖励 + 角度补偿 + 速度奖励
    reward = 1.0 + (0.5 - angle_penalty) + velocity_bonus

    # # 每坚持x步额外奖励
    # if step % 10 == 0:
    #     reward += 1
    if step % 30 == 0:
        reward += 3
    if step % 50 == 0:
        reward += 10
    # reward += step / 10
    if step % 500 == 0:
        reward += 1000

    return reward


# Agent与环境交互
env = gym.make("CartPole-v1", render_mode="human")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

agent = Agent(n_actions=n_actions, n_states=n_states)

max_episodes = 200
max_steps = 10000

step_list = []
for episode in range(max_episodes):
    state, _ = env.reset(seed=0)

    for step in range(max_steps):
        action = agent.choose_action(state, episode=episode)
        next_state, _, terminated, truncated, info = env.step(action)
        reward = get_reward(next_state, step)

        agent.memorize(state, action, next_state, reward)

        # 每x步更新一次网络
        if step % 4 == 0:
            agent.update_q_function()

        state = next_state

        if terminated or truncated:
            step_list.append(step + 1)
            print(f"Episode: {episode+1}, Steps: {step+1}")
            break

# 保存模型参数
agent.save_policy_net_to(agent.model_path)

plt.plot(step_list)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.savefig('./outputs/CartPole_steps.png')  # 保存图像文件
plt.show()



