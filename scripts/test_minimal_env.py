# D:\wow_ai\scripts\test_minimal_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MinimalEnv(gym.Env):
    metadata = {"render_modes": ["human", None]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = spaces.Discrete(2) # 简单动作空间
        self.observation_space = spaces.Box(0, 1, shape=(4,), dtype=np.float32) # 简单观察空间
        self.render_mode = render_mode
        self.current_step = 0
        print("MinimalEnv initialized.")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        observation = self.observation_space.sample() # 返回随机观察
        info = {}
        print("MinimalEnv reset.")
        return observation, info

    def step(self, action):
        self.current_step += 1
        observation = self.observation_space.sample() # 返回随机观察
        reward = 0.0 # 简单奖励
        terminated = False
        truncated = self.current_step >= 10 # 最多 10 步
        info = {}
        print(f"MinimalEnv step {self.current_step}: action={action}, reward={reward}")
        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        print("MinimalEnv closed.")

# --- 测试代码 ---
if __name__ == "__main__":
    print("Testing MinimalEnv...")
    env = MinimalEnv()
    obs, info = env.reset()
    done = False
    truncated = False
    step = 0
    while not done and not truncated:
        step += 1
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    print(f"MinimalEnv test finished after {step} steps.")
    env.close()