import gymnasium as gym
import numpy as np

class MazeEnv(gym.Env):
    def __init__(self):
        self.state = 0
        pass
    def reset(self):
        self.state = 0
        self.goal = 8
        return self.state
    def step(self, action):
        if action == 0:
            self.state += 1
        elif action == 1:
            self.state -= 1
        elif action == 2:
            self.state += 10
        elif action == 3:
            self.state -= 10

        if self.state == self.goal:
            terminated = True
        else:
            terminated = False

        reward = 1
        truncated = False
        info = {}
        return self.state, reward, terminated, truncated, info


class Agent:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]
        self.pi = np.ones((10, 4)) / 4
    def choose_action(self, state):
        # action = np.random.choice(self.action_space, p=self.pi[state, :])
        action = 0
        return action

env = MazeEnv()
state = env.reset()

agent = Agent()


done = False
action_history = []
state_history = []
cnt = 0
while not done:
    action = agent.choose_action(state)
    state, reward, terminated, truncated, info = env.step(action)

    state_history.append(state)
    action_history.append(action)

    print(action, state)
    done = terminated or truncated
    cnt += 1
    if cnt > 10:
        break