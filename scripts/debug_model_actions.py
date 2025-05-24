# D:\wow_ai\scripts\debug_model_actions.py
# 调试PPO模型在不同状态下的动作选择
import os
import sys
import time
import numpy as np
import torch

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
# 模型路径 - 可以使用重置后的模型
USE_RESET_MODEL = False  # 改为False使用现有模型
if USE_RESET_MODEL:
    MODEL_PATH = os.path.join(project_root, "runs", "ppo_models", "ppo_wow_agent_m1_bc_reset.zip")
else:
    MODEL_PATH = os.path.join(project_root, "runs", "ppo_models", "ppo_wow_agent_m1_final.zip")
    
# 日志路径
LOG_DIR = os.path.join(project_root, "runs", "debug_logs")
# --- 配置结束 ---

def debug_model_actions():
    """分析模型在不同状态下的动作选择"""
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
        "render_mode": "human"
    }
    
    env = WowKeyEnv(**env_kwargs)
    
    # 加载模型
    # 确定设备
    device = torch.device("cpu") # 或者 torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device} for model and observations in debug script.")

    try:
        print(f"正在加载模型: {MODEL_PATH}...")
        # 加载模型时指定设备，如果模型已在GPU上训练并保存
        # 如果不确定，先加载到CPU，再视情况 .to(device)
        model = PPO.load(MODEL_PATH, env=env, device="auto") # SB3的auto通常能处理
        model.policy.to(device) # 确保策略网络在指定设备上
        
        print("模型加载成功!")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        print("详细错误信息:")
        import traceback
        traceback.print_exc()
        
        # 检查模型文件是否损坏
        try:
            import zipfile
            with zipfile.ZipFile(MODEL_PATH, 'r') as zip_ref:
                print(f"模型文件ZIP内容: {zip_ref.namelist()}")
        except Exception as zip_error:
            print(f"无法读取模型文件内容: {zip_error}")
            
        env.close()
        return
    
    # 设置F10停止
    stop_flag = [False]
    
    def on_f10_press(e):
        if e.name.lower() == 'f10':
            print("\nF10 已按下，停止调试...")
            stop_flag[0] = True
    
    keyboard.on_press(on_f10_press)
    
    print("\n开始调试 - 按F10停止")
    print("=" * 50)
    print("按空格键暂停，查看模型在当前状态下的所有动作概率")
    print("-" * 50)
    
    # 重置环境
    obs, info = env.reset()
    
    # 用于统计各动作的使用频率
    action_counts = {}
    for i in range(env.action_space.n):
        action_counts[i] = 0
    
    # 记录前一个sel_flag和is_dead状态
    prev_sel_flag = 0
    prev_is_dead = 0
    
    # 循环运行，直到按F10停止
    step = 0
    while not stop_flag[0]:
        step += 1
        
        # 检查键盘输入
        if keyboard.is_pressed('space'):
            print("\n暂停 - 查看所有动作概率")
            
            try:
                # 将字典中的每个numpy数组转换为torch张量，并移到正确的设备
                torch_obs = {}
                for key, value in obs.items():
                    if key == "frame":
                        # 确保frame的通道顺序是正确的 (B,C,H,W)
                        value_tensor = torch.tensor(value).float().permute(2, 0, 1).unsqueeze(0).to(device)
                    else:
                        value_tensor = torch.tensor(value).float().unsqueeze(0).to(device)
                    torch_obs[key] = value_tensor
                
                # 获取所有动作的原始概率
                with torch.no_grad():
                    action_probs = model.policy.get_distribution(torch_obs).distribution.probs.detach().numpy()
                
                # 显示所有动作的概率
                print("\n当前状态下的动作概率:")
                for i in range(env.action_space.n):
                    action_name = env.ACTION_TABLE.get(i, f"动作 {i}")
                    print(f"  {action_name}: {action_probs[0][i]:.4f}")
            except Exception as e:
                print(f"获取动作概率时出错: {e}")
                import traceback
                traceback.print_exc()
            
            # 等待用户按Enter继续
            input("\n按 Enter 继续...")
        
        # 获取当前状态信息
        sel_flag = info.get("sel_flag", 0)
        is_dead = info.get("is_dead", 0)
        
        # 检测状态变化
        if sel_flag != prev_sel_flag or is_dead != prev_is_dead:
            print(f"\n状态变化: sel_flag: {prev_sel_flag} -> {sel_flag}, is_dead: {prev_is_dead} -> {is_dead}")
            prev_sel_flag = sel_flag
            prev_is_dead = is_dead
        
        # 获取动作掩码
        mask = np.ones(env.action_space.n, dtype=bool)
        if not (sel_flag and is_dead):      # 只有选中且死亡才允许拾取
            mask[2] = 0  # 禁用]键
        if is_dead:                         # 尸体阶段禁止攻击
            mask[0] = mask[7] = 0  # 禁用F和Tab
        # 完全禁用Tab键
        mask[7] = 0
        info["action_mask"] = mask
        
        # 使用模型预测动作，并使用动作掩码
        # 确保obs被预处理并传递到与模型相同的设备
        processed_obs = model.policy.obs_to_tensor(obs)[0].to(device) # 获取并处理obs
        action, _states = model.predict(processed_obs, action_masks=mask, deterministic=True)
        action_int = int(action)
        
        # 更新动作计数
        action_counts[action_int] = action_counts.get(action_int, 0) + 1
        
        # 显示选择的动作信息
        action_name = env.ACTION_TABLE.get(action_int, f"动作 {action_int}")
        print(f"\r步骤 {step}: 选择 {action_name} (ID: {action_int}), 掩码: {mask}, 状态: sel={sel_flag} dead={is_dead}", end="")
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action_int)
        
        # 如果回合结束，重置环境
        if terminated or truncated:
            print("\n回合结束，重置环境")
            obs, info = env.reset()
            
        # 稍微延迟以便观察
        time.sleep(0.05)
    
    # 显示动作使用统计
    print("\n\n动作使用统计:")
    total_actions = sum(action_counts.values())
    for action_id, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        action_name = env.ACTION_TABLE.get(action_id, f"动作 {action_id}")
        percentage = (count / total_actions) * 100 if total_actions > 0 else 0
        print(f"  {action_name}: {count} 次 ({percentage:.1f}%)")
    
    # 清理
    keyboard.unhook_all()
    env.close()
    print("\n调试结束")

if __name__ == "__main__":
    debug_model_actions()       