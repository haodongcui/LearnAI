
import torch
import torch.nn as nn
import torch.optim as optim

import gym_super_mario_bros                                 # 超级玛丽环境, 基于NES模拟器
from nes_py.wrappers import JoypadSpace                     # NES模拟器的手柄动作控制器
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT    # 动作空间的封装, 选用SIMPLE_MOVEMENT
from gym.wrappers import FrameStack                         # 帧堆叠器, 将连续的帧堆叠成一个状态, 感知连续画面
from gym.wrappers import GrayScaleObservation
from gym.wrappers import ResizeObservation
import gym

import numpy as np
import random
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from collections import namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None) # 增加占位
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity # 循环队列的指针循环
    def sample(self, batch_size):
        # if len(self.memory) < batch_size:
        #     return self.memory
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class QNet(nn.Module):
    def __init__(self, n_states, n_actions):
        super(QNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(n_states, 32, kernel_size=4, stride=2, padding=1),  # 输出: (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 输出: (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 输出: (64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),  # 展平: (64 * 7 * 7 = 3136)
            nn.Linear(64 * 21 * 21, 512),  # 全连接层
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    def forward(self, x):
        x = self.model(x)
        return x



class Agent:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = 0.001
        self.gamma = 0.99
        self.tau = 0.01                 # 目标网络的软更新参数
        self.epsilon_start = 0.7
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.batch_size = 64
        self.capacity = 10000

        self.weights_dir = "weights"
        self.memory = ReplayMemory(capacity=self.capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = self.epsilon_start
        self.target_update_mode = 'soft'    # 'hard' or 'soft'

        self.policy_net = QNet(n_states, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()

        self.target_net = QNet(n_states, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_update_counter = 0      # 计数器，用于控制目标网络的更新频率
        self.target_update_frequency = 2    # 目标网络更新的频率


    def choose_action(self, state):
        if np.random.rand() < self.epsilon_start:
            action = random.randint(0, self.n_actions - 1)
        else:
            state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            action = torch.argmax(q_values).item()
        return action

    def memory_push(self, state, action, next_state, reward):
        transition = Transition(state, action, next_state, reward)
        self.memory.push(transition)

    def update(self, episode):
        # 记忆回放, 更新Q网络
        self._replay()
        # 更新探索率
        self.epsilon = max(self.epsilon_min, self.epsilon_start * self.epsilon_decay ** episode)
        # 目标网络更新
        if self.target_update_mode == 'hard':
            self._target_update_hard()
        elif self.target_update_mode == 'soft':
            self._target_update_soft()
        else:
            raise ValueError("Invalid target update mode. Choose 'hard' or 'soft'.")

    def _target_update_hard(self):
        self.target_update_counter += 1
        if self.target_update_counter % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _target_update_soft(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def _replay(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(np.array(batch.action), dtype=torch.int64).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(np.array(batch.reward), dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(self.device)

        current_q = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            # best_actions = current_q.argmax(1)
            best_actions = self.policy_net(next_state_batch).argmax(1)
            next_q = self.target_net(next_state_batch)
            next_q = next_q.gather(1, best_actions.unsqueeze(1)) # gather(dim,index): 在特定维度按照索引提取值
            target_q = reward_batch + self.gamma * next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_checkpoint(self, episode):
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
        weights = self.policy_net.state_dict()
        epsilon = self.epsilon
        file = dict(weights=weights, epsilon=epsilon)
        filename = os.path.join(self.weights_dir, f'checkpoint_{episode}.pth')
        torch.save(file, filename)
        print(f'---checkpoint saved when episode: {episode}---')

    def load_checkpoint(self, filename):
        checkout = torch.load(filename)
        weights = checkout['weights']
        self.epsilon = checkout['epsilon']
        self.policy_net.load_state_dict(weights)
        print(f'---checkpoint loaded from {filename}---')


class ResizeObservationSqueeze(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = shape
        # 初始化父类的缩放逻辑
        self.resize_wrapper = ResizeObservation(env, shape)

    def observation(self, obs):
        # 先调用父类的缩放方法
        resized_obs = self.resize_wrapper.observation(obs)
        # 压缩冗余维度（如 (84,84,1) -> (84,84)）
        return np.squeeze(resized_obs)


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

def plot_steps(step_list, episode):
    plt.plot(step_list)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.savefig(f'./outputs/steps_{episode}.png')  # 保存图像文件
    plt.close()



def get_reward(info, prev_info=None):
    reward = 0
    if prev_info is None:
        return reward
    # 基础移动奖励（鼓励向右移动）
    if prev_info:
        reward += 0.2 * (info['x_pos'] - prev_info['x_pos'])  # 每向右1像素+0.1分
    # 关键事件奖励
    if info['coins'] > prev_info['coins']:
        reward += 10  # 每枚金币+10分[3](@ref)
    if info['status'] == 'tall' and prev_info['status'] == 'small':
        reward += 50  # 吃蘑菇变大+50分[7](@ref)
    if info['flag_get']:
        reward += 1000  # 通关+1000分[1](@ref)
    if info['life'] < prev_info['life']:
        reward -= 100  # 死亡惩罚-100分[1](@ref)
    # 时间惩罚（避免拖延）
    reward -= 0.05 * (prev_info['time'] - info['time'])  # 每秒-3分（0.05 * 60帧）
    return reward




# SIMPLE_MOVEMENT 封装动作:
# [['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']]

env = gym_super_mario_bros.make('SuperMarioBros-v0')
# env = JoypadSpace(env, SIMPLE_MOVEMENT)
# env = JoypadSpace(env, COMPLEX_MOVEMENT)

CUSTOM_MOVEMENT = [
    ['NOOP'],
    ['left'],
    ['right'],
    ['A'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
]
env = JoypadSpace(env, actions=CUSTOM_MOVEMENT)

# 预处理游戏信息
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False) # 灰度化 (240, 256)
# env = ResizeObservation(env, (84, 84)) # 缩放到 (84, 84, 1)
env = ResizeObservationSqueeze(env, (84, 84)) # 缩放到 (84, 84)
env = FrameStack(env, num_stack=4) # 堆叠4帧为 (4, 84, 84)

env.seed(1)
obs = env.reset() # (240, 256, 3) 图像帧

n_states = env.observation_space.shape[0] # 4
n_actions = env.action_space.n # 7

agent = Agent(n_states, n_actions)

max_episodes = 10000
max_steps = 1000000
step_list = []

for episode in range(max_episodes):
    state = env.reset()
    total_reward = 0
    prev_info = None
    for step in range(max_steps):
        action = agent.choose_action(state)
        env.render()
        next_state, reward, done, info = env.step(action)

        # reward = get_reward(info, prev_info)

        agent.memory_push(state, action, next_state, reward)
        if step % 4 == 0:
            agent.update(episode)

        state = next_state
        # prev_info = info
        total_reward += reward

        if done or step == max_steps - 1:
            step_list.append(step + 1)
            if (episode + 1) % 20 == 0:
                agent.save_checkpoint(episode)
            if (episode + 1) % 100 == 0:
                plot_steps(step_list, episode)

            print(f"Episode {episode}, Steps: {step}, Reward: {total_reward}, Epsilon: {agent.epsilon}")
            break




# info
# {'coins': 0, 'flag_get': False, 'life': 2, 'score': 100, 'stage': 1, 'status': 'small', 'time': 370, 'world': 1, 'x_pos': 426, 'y_pos': 89}