
import gymnasium as gym
import time
import matplotlib.pyplot as plt
from matplotlib import animation



# env = gym.make("Ant-v5", render_mode="human")
env = gym.make("CartPole-v1", render_mode="rgb_array")

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
        # time.sleep(0.01) # 每x秒渲染一帧
    print(f'episode: {episode+1}, total score: {score}')

    print(frames[0].shape)
    print(len(frames))
env.close()


# print(help(gym.Env.render))



# 保存为 video/gif
def display_frames_to_video(frames):
    figsize_w = frames[0].shape[0] / 72.0
    figsize_h = frames[0].shape[1] / 72.0
    plt.figure(figsize=(figsize_w, figsize_h), dpi=72)
    patch = plt.imshow(frames[0])

    def animate(i):
        patch.set_data(frames[i])


    anim = animation.FuncAnimation(plt.gcf(), animate, frames=range(len(frames)), interval=50)
    anim.save('cartpole.gif', writer='pillow')

display_frames_to_video(frames)