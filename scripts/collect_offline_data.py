# collect_offline_data.py
import traceback
import gymnasium as gym
import numpy as np
import sys
import os
import time
from tqdm import tqdm # 导入 tqdm 用于显示进度条

# --- 动态路径设置 (同 test_env.py) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) if os.path.basename(current_dir) == 'scripts' else current_dir
wow_rl_path = os.path.join(parent_dir, 'wow_rl')
if wow_rl_path not in sys.path:
    sys.path.append(parent_dir)
    print(f"Added {parent_dir} to sys.path")
# --- 路径设置结束 ---

try:
    from wow_rl.envs.wow_click_env import WowClickEnv
    from wow_rl.buffers.replay_buffer import ReplayBuffer
    print("Imported WowClickEnv and ReplayBuffer successfully.")
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

# --- 配置参数 ---
BUFFER_CAPACITY = 30000 # 先收集 30k 步数据 (ChatGPT 建议值)
BUFFER_SAVE_DIR = 'offline_buffer_data_30k' # 数据保存目录名
NUM_EPISODES = 600 # 运行的回合数 (约 600 * 50 = 30k 步)
MAX_STEPS_PER_EPISODE = 50 # 每个回合的最大步数 (与环境默认值一致)

# 环境参数 (确保与创建环境时一致)
DETECTOR_W = r'D:\wow_ai\runs\detect\gwyx_detect_run_s_ep200_neg\weights\best.pt'
ROI_REWARD = (319, 35, 245, 64)
TEMPLATE_PATH = r'D:\wow_ai\data\target_frame_template.png'
MATCH_THRESHOLD = 0.85
# --- 配置结束 ---

print("--- Offline Data Collection Script ---")

# 1. 初始化环境 (不需要渲染)
print("Initializing environment...")
env = WowClickEnv(
    render_mode=None, # !! 收集数据时不需要渲染窗口 !!
    detector_w=DETECTOR_W,
    roi_reward=ROI_REWARD,
    template_path=TEMPLATE_PATH,
    match_threshold=MATCH_THRESHOLD,
    max_steps=MAX_STEPS_PER_EPISODE
)
print("Environment initialized.")

# 2. 初始化 Replay Buffer
# 获取观察空间和动作空间信息
obs_space = env.observation_space
act_space = env.action_space
print(f"Observation Space: {obs_space}, Action Space: {act_space}")

# 检查观察空间形状是否符合预期 (640, 640, 3)
if not (len(obs_space.shape) == 3 and obs_space.shape[2] == 3):
     print(f"Error: Unexpected observation space shape: {obs_space.shape}. Expected (H, W, 3).")
     env.close()
     sys.exit(1)

# 动作空间通常是 Discrete，我们需要其维度 n
action_dim = 1 if isinstance(act_space, gym.spaces.Discrete) else act_space.shape[0]
if isinstance(act_space, gym.spaces.Discrete):
    print(f"Action space is Discrete with {act_space.n} actions.")
else:
    print(f"Action space shape: {act_space.shape}")


print(f"Initializing replay buffer with capacity {BUFFER_CAPACITY}...")
buffer = ReplayBuffer(
    capacity=BUFFER_CAPACITY,
    obs_shape=obs_space.shape,
    action_dim=action_dim, # 对于 Discrete 动作空间，维度是 1
    save_dir=BUFFER_SAVE_DIR
)
# 尝试加载之前的进度
buffer.load()
print(f"Replay buffer initialized. Current size: {len(buffer)}")


# 3. 运行循环收集数据
print(f"Starting data collection for {NUM_EPISODES} episodes...")
collected_steps = 0
try:
    # 使用 tqdm 显示总的回合进度
    for episode in tqdm(range(NUM_EPISODES), desc="Collecting Episodes"):
        # 如果缓冲区满了，提前停止
        if len(buffer) >= BUFFER_CAPACITY:
            print("\nBuffer is full. Stopping collection.")
            break

        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_steps = 0

        # 每个回合最多运行 max_steps 步
        while not done and not truncated:
            # a. 随机选择动作
            action = env.action_space.sample()

            # b. 与环境交互
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated # done 表示回合结束

            # c. 添加到缓冲区 (确保数据类型和形状正确)
            # 我们存储的是原始观察状态 (640x640x3)
            # DreamerV3 后续会自己处理缩放
            buffer.add(obs, action, reward, next_obs, done)

            # d. 更新状态和统计
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            collected_steps += 1

            # 检查缓冲区是否已满
            if len(buffer) >= BUFFER_CAPACITY:
                 print("\nBuffer is full during episode. Stopping collection.")
                 break # 跳出内部循环

        print(f"\nEpisode {episode + 1} finished after {episode_steps} steps. Reward: {episode_reward:.2f}. Buffer size: {len(buffer)}/{BUFFER_CAPACITY}")
        # 每隔一定回合数保存一次进度
        if (episode + 1) % 50 == 0:
             print(f"Saving buffer progress at episode {episode + 1}...")
             buffer.save()

except KeyboardInterrupt:
    print("\nData collection interrupted by user.")
except Exception as e:
    print(f"\nAn error occurred during data collection: {e}")
    traceback.print_exc() # <-- 打印完整的错误堆栈信息
finally:
    # 4. 结束时保存缓冲区
    print("Finalizing data collection...")
    buffer.save()
    env.close()
    print(f"Data collection finished. Total steps collected in this run: {collected_steps}")
    print(f"Total buffer size: {len(buffer)}")
    print(f"Data saved in: {os.path.abspath(BUFFER_SAVE_DIR)}")