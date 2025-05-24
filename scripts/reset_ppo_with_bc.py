# D:\wow_ai\scripts\reset_ppo_with_bc.py
# 使用BC模型重新初始化PPO模型
import os
import sys
import torch
import numpy as np
from datetime import datetime

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
    from stable_baselines3.common.policies import ActorCriticPolicy
    from wow_rl.envs.wow_key_env import WowKeyEnv
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所有必要的库: stable_baselines3")
    sys.exit(1)

# --- 配置 ---
# BC模型路径
BC_MODEL_PATH = os.path.join(project_root, "runs", "bc_models", "bc_policy_simple.pth")
# PPO模型保存路径
PPO_MODEL_PATH = os.path.join(project_root, "runs", "ppo_models", "ppo_wow_agent_m1_bc_reset.zip")
# 日志目录
LOG_DIR = os.path.join(project_root, "runs", "ppo_logs_m1_reset")
# --- 配置结束 ---

def reset_ppo_with_bc():
    """使用BC模型重新初始化PPO模型"""
    print(f"创建环境...")
    
    # 创建环境
    env_kwargs = {
        "roi_reward": (250, 35, 205, 64),
        "match_thresh_select": 0.50,
        "match_thresh_dead": 0.5
    }
    
    env = WowKeyEnv(**env_kwargs)
    
    # 检查BC模型文件是否存在
    if not os.path.exists(BC_MODEL_PATH):
        print(f"错误: 未找到BC模型文件: {BC_MODEL_PATH}")
        return
    
    print(f"加载BC模型: {BC_MODEL_PATH}")
    
    # 创建PPO模型并初始化权重
    try:
        # 创建新的PPO模型
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.8,
            clip_range=0.25,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=LOG_DIR,
            policy_kwargs={"net_arch": [dict(pi=[64, 64], vf=[64, 64])]}
        )
        
        # 加载BC模型权重
        bc_weights = torch.load(BC_MODEL_PATH)
        
        # 只复制与模型策略共享的权重
        # 这需要了解BC模型和PPO模型的确切结构
        # 这里提供一个简化版本
        state_dict = model.policy.state_dict()
        
        # 假设BC模型结构与PPO的MLP策略类似
        for name, param in bc_weights.items():
            if name in state_dict:
                if state_dict[name].shape == param.shape:
                    state_dict[name].copy_(param)
                    print(f"复制权重: {name}")
                else:
                    print(f"形状不匹配，跳过: {name} - PPO: {state_dict[name].shape}, BC: {param.shape}")
            else:
                print(f"在PPO模型中找不到参数: {name}")
        
        # 保存初始化后的PPO模型
        model.save(PPO_MODEL_PATH)
        print(f"已保存重置的PPO模型到: {PPO_MODEL_PATH}")
        
    except Exception as e:
        print(f"初始化模型时出错: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return
    
    # 清理
    env.close()
    print("\n重置完成")

if __name__ == "__main__":
    reset_ppo_with_bc() 