from grid_maze_env import GridMazeEnv
from stable_baselines3.common.evaluation import evaluate_policy
import time

env = GridMazeEnv(grid_size=5, render_mode="human")
obs, _ = env.reset()

for _ in range(50):
    action = env.action_space.sample()  # random action
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    time.sleep(0.5)

    if done:
        print("Reached goal!")
        break

env.close()
