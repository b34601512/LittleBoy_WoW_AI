# D:\wow_ai\scripts\test_trained_model.py
# 测试训练好的PPO模型效果
import os
import sys
import time
import numpy as np
from collections import deque

# --- 动态路径设置 ---
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir) # wow_ai
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    project_root = None
# --- 路径设置结束 ---

# 导入必要的库
try:
    from stable_baselines3 import PPO
    from wow_rl.envs.wow_key_env import WowKeyEnv
    import keyboard  # 用于检测F10停止
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所有必要的库: stable_baselines3, keyboard")
    sys.exit(1)

# --- 配置 ---
# 模型路径
MODEL_PATH = os.path.join(project_root, "runs", "ppo_models", "ppo_wow_agent_m1_final.zip")
# 测试回合数
TEST_EPISODES = 10
# 最大步数
MAX_STEPS_PER_EPISODE = 200
# 是否在测试时渲染环境
RENDER = True
# --- 配置结束 ---

def evaluate_model():
    """加载并评估训练好的模型"""
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 未找到模型文件: {MODEL_PATH}")
        return
    
    print(f"加载模型: {MODEL_PATH}")
    
    # 创建环境
    env_kwargs = {
        "roi_reward": (250, 35, 205, 64),
        "match_thresh_select": 0.50,
        "match_thresh_dead": 0.5,
        "max_steps": MAX_STEPS_PER_EPISODE
    }
    
    env = WowKeyEnv(render_mode="human" if RENDER else None, **env_kwargs)
    
    # 加载模型
    try:
        model = PPO.load(MODEL_PATH, env=env)
        print("模型加载成功!")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        env.close()
        return
    
    # 设置F10停止
    stop_flag = [False]
    
    def on_f10_press(e):
        if e.name.lower() == 'f10':
            print("\nF10 已按下，停止测试...")
            stop_flag[0] = True
    
    keyboard.on_press(on_f10_press)
    
    print("\n开始测试 - 按F10停止")
    print("=" * 50)
    print("回合   | 步数  | 奖励  | 击杀  | 拾取  | 面向错误 | 距离错误")
    print("-" * 50)
    
    # 指标记录
    episode_rewards = []
    episode_lengths = []
    kills = 0
    loots = 0
    need_face_count = 0
    need_range_count = 0
    total_steps = 0
    
    # 用于计算每秒执行的步数
    fps_buffer = deque(maxlen=100)
    last_time = time.time()
    
    # 开始测试回合
    for i_episode in range(TEST_EPISODES):
        if stop_flag[0]:
            break
        
        # 重置环境
        obs, info = env.reset()
        episode_reward = 0
        episode_step = 0
        episode_kills = 0
        episode_loots = 0
        episode_face_errors = 0
        episode_range_errors = 0
        
        done = False
        while not done and episode_step < MAX_STEPS_PER_EPISODE:
            if stop_flag[0]:
                break
            
            # 记录FPS
            current_time = time.time()
            fps_buffer.append(1.0 / max(1e-5, current_time - last_time))
            last_time = current_time
            
            # 获取动作掩码 (从info中获取，第一次循环时使用env.reset()返回的info)
            action_mask = info.get('action_mask', None)
            if action_mask is not None:
                print(f"\r可用动作掩码: {action_mask}", end="")
            
            # 使用模型预测动作
            action, _states = model.predict(obs, deterministic=True)
            
            # 将numpy数组转换为Python整数
            action_int = int(action)
            action_name = "未知"
            if hasattr(env, "ACTION_TABLE"):
                action_name = env.ACTION_TABLE.get(action_int, "未知")
            print(f"\r选择动作ID: {action_int}, 名称: {action_name}", end="")
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action_int)
            
            # 更新统计信息
            episode_reward += reward
            episode_step += 1
            total_steps += 1
            
            # 记录事件
            if info.get("sel_flag", 0) == 1 and info.get("is_dead", 0) == 1:
                # 目标死亡(可能是击杀)
                if info.get("reward_this_step", 0) >= 2.0:  # 击杀奖励约为2.0
                    episode_kills += 1
                    kills += 1
            
            if info.get("loot_success_this_frame", 0) == 1:
                # 成功拾取
                episode_loots += 1
                loots += 1
            
            if info.get("need_face", False):
                episode_face_errors += 1
                need_face_count += 1
            
            if info.get("need_range", False):
                episode_range_errors += 1
                need_range_count += 1
            
            # 检查是否结束回合
            done = terminated or truncated
            
            # 可选: 显示当前动作和奖励
            action_name = info.get("action_name", "未知")
            print(f"\r当前动作: {action_name}, 奖励: {reward:.2f}, FPS: {np.mean(fps_buffer):.1f}", end="")
        
        # 记录回合结果
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_step)
        
        # 显示回合结果
        print(f"\r回合 {i_episode+1:2d} | {episode_step:4d} | {episode_reward:5.2f} | {episode_kills:4d} | {episode_loots:4d} | {episode_face_errors:8d} | {episode_range_errors:8d}")
    
    # 计算并显示总体性能
    print("=" * 50)
    if episode_rewards:
        print(f"平均奖励: {np.mean(episode_rewards):.2f}")
        print(f"平均步数: {np.mean(episode_lengths):.2f}")
        print(f"总击杀数: {kills}")
        print(f"总拾取数: {loots}")
        print(f"拾取/击杀比: {loots/max(1,kills):.2f}")
        
        # 计算错误率
        face_error_rate = need_face_count / max(1, total_steps)
        range_error_rate = need_range_count / max(1, total_steps)
        print(f"面向错误率: {face_error_rate:.2f}")
        print(f"距离错误率: {range_error_rate:.2f}")
    
    # 清理
    keyboard.unhook_all()
    env.close()
    print("\n测试结束")

if __name__ == "__main__":
    evaluate_model() 