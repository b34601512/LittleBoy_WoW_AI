# D:\\wow_ai\\scripts\\train_ppo_wow.py
# (Version: PPO trains from scratch, CustomCombinedExtractor defined for MultiInputPolicy)
import gymnasium as gym
import os
import time
import torch
import torch.nn as nn
import stable_baselines3 # Base import

# --- Stable Baselines3 Imports ---
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN # For Custom Extractor
from gymnasium.spaces.utils import flatdim # For Custom Extractor
# --- End SB3 Imports ---

# --- Project Specific Imports ---
import sys
import traceback # For better error printing
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir) # wow_ai
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added project root '{project_root}' to sys.path for WowKeyEnv import.")
    from wow_rl.envs.wow_key_env import WowKeyEnv # Ensure this is the latest version
    print("WowKeyEnv imported successfully.")
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import WowKeyEnv. Ensure it's in the PYTHONPATH.")
    print(f"ImportError: {e}"); traceback.print_exc(); sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR during WowKeyEnv import: {e}"); traceback.print_exc(); sys.exit(1)
# --- End Project Imports ---

# --- 尝试导入可能会需要的其他库 ---
try:
    import keyboard  # 导入keyboard库用于等待触发键
    HAVE_KEYBOARD = True
except ImportError:
    HAVE_KEYBOARD = False
    print("警告: 未找到keyboard库，无法使用按键触发功能")

try:
    import pygetwindow as gw  # 用于获取和激活窗口
    HAVE_WINDOW_UTILS = True
    print("已加载pygetwindow库，可以激活游戏窗口")
except ImportError:
    HAVE_WINDOW_UTILS = False
    print("未找到pygetwindow库，将无法自动激活游戏窗口")
# --- 导入结束 ---

def wait_for_trigger_key(trigger_key='f8'):
    """等待用户按下触发键"""
    if not HAVE_KEYBOARD:
        print("警告: 无法使用按键触发功能，因为未安装keyboard库")
        input("按回车继续...")
        return
        
    print(f"等待按下 {trigger_key.upper()} 键来开始训练...")
    print("请先切换到游戏窗口，然后按下触发键")
    
    # 定义一个标志变量，表示是否已按下触发键
    key_pressed = False
    
    # 定义回调函数，当触发键被按下时设置标志
    def on_trigger_key_press(e):
        nonlocal key_pressed
        if e.name.lower() == trigger_key.lower():
            key_pressed = True
    
    # 设置按键监听
    keyboard.on_press(on_trigger_key_press)
    
    # 等待按键被按下
    while not key_pressed:
        time.sleep(0.1)
    
    # 移除监听器
    keyboard.unhook_all()
    
    print(f"\\n{trigger_key.upper()} 已按下，开始训练！")
    # 等待一小段时间，让用户有时间释放按键
    time.sleep(0.3)

def activate_wow_window():
    """尝试激活魔兽世界窗口"""
    if not HAVE_WINDOW_UTILS:
        print("无法激活WoW窗口，pygetwindow库未安装")
        return False
    
    try:
        # 尝试查找魔兽世界窗口
        wow_windows = [w for w in gw.getAllTitles() if 'World of Warcraft' in w]
        if not wow_windows:
            # 尝试其他可能的名称
            wow_windows = [w for w in gw.getAllTitles() if 'Warcraft' in w]
            
        if wow_windows:
            print(f"找到WoW窗口: {wow_windows[0]}")
            wow_window = gw.getWindowsWithTitle(wow_windows[0])[0]
            wow_window.activate()
            time.sleep(0.5)  # 给一点时间让窗口激活
            return True
        else:
            print("未找到WoW窗口，请确保游戏正在运行")
            return False
    except Exception as e:
        print(f"无法激活WoW窗口: {e}")
        return False

# --- 自定义训练控制回调 ---
class TrainingControlCallback(BaseCallback):
    """回调函数，监控F9暂停/恢复，F10停止并保存训练"""
    
    def __init__(self, verbose=0):
        super(TrainingControlCallback, self).__init__(verbose)
        self.paused = False
        self.stop_requested = False
        self.pause_key = 'f9'
        self.stop_key = 'f10'
        self.keyboard_available = HAVE_KEYBOARD
        self._keyboard_hooked = False # Track if keyboard listeners are active
        
    def _on_training_start(self) -> None:
        """训练开始时设置键盘监听"""
        if not self.keyboard_available:
            print(f"警告: keyboard库不可用，无法使用 {self.pause_key.upper()} 暂停/恢复 和 {self.stop_key.upper()} 停止功能。")
            return
        
        if self._keyboard_hooked: # Avoid multiple hooks if training is resumed
            return

        print(f"训练控制: 按 {self.pause_key.upper()} 暂停/恢复 | 按 {self.stop_key.upper()} 停止并保存模型")
        
        # 设置键盘钩子
        def on_key_press(e):
            key_name = e.name.lower()
            if key_name == self.pause_key:
                self.paused = not self.paused
                status = "暂停" if self.paused else "恢复"
                print(f"\\n训练已{status}! 再次按 {self.pause_key.upper()} 键 {('恢复' if self.paused else '暂停')}训练...")
                if self.paused and self.stop_requested: # If we pause while a stop is pending, clear stop
                    print(f"注意：暂停操作取消了先前的停止请求。")
                    self.stop_requested = False
            elif key_name == self.stop_key:
                if not self.stop_requested: # Prevent multiple stop messages
                    print(f"\\n{self.stop_key.upper()} 已按下，请求停止训练并保存模型...")
                    print("训练将在当前批次完成后安全停止。")
                    self.stop_requested = True
                    if self.paused: # If paused, unpause to allow training to stop gracefully
                        self.paused = False 
                        print("已自动恢复训练以便完成停止流程。")

        keyboard.on_press(on_key_press)
        self._keyboard_hooked = True
    
    def _on_step(self) -> bool:
        """每步检查是否暂停或请求停止"""
        if self.stop_requested:
            return False # 返回False以停止训练

        while self.paused and self.keyboard_available:
            if self.stop_requested: # Check again in case F10 is pressed during pause
                return False 
            time.sleep(0.1)  # 暂停时轻微睡眠以减少CPU使用
        return True # 返回True以继续训练
        
    def _on_training_end(self) -> None:
        """训练结束时清理钩子"""
        if self.keyboard_available and self._keyboard_hooked:
            keyboard.unhook_all()
            self._keyboard_hooked = False
            print("训练控制键盘监听已移除。")

# --- Custom Feature Extractor Definition (as discussed with ChatGPT O4) ---
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # Calculate features_dim before calling super().__init__
        total_dim = 0
        extractors_temp = {} # Temporary dict to build ModuleDict later
        for key, subspace in observation_space.spaces.items():
            if key == "frame":
                # For images, use NatureCNN. You can set features_dim for its output.
                cnn = NatureCNN(subspace, features_dim=128) # Example: 128 features from CNN
                extractors_temp[key] = cnn
                total_dim += cnn.features_dim
            else:
                # For other spaces (Box, MultiBinary), flatten them.
                flatten = nn.Flatten()
                extractors_temp[key] = flatten
                total_dim += flatdim(subspace)
        
        super().__init__(observation_space, features_dim=total_dim) # Pass correct total_dim
        self.extractors = nn.ModuleDict(extractors_temp) # Now create the ModuleDict

    def forward(self, observations: dict) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)
# --- End Custom Feature Extractor ---

def make_env(rank=0, seed=0, render_during_training=False):
    """Utility function for multiprocessed env."""
    def _init():
        render_mode = "human" if rank == 0 and render_during_training else None
        # 修复env_kwargs，移除不存在的参数
        env_kwargs = {
            "roi_reward": (250, 35, 205, 64),
            "match_thresh_select": 0.50,
            "match_thresh_dead": 0.5,
            "max_steps": 200
        }
        
        # 初始化前尝试激活游戏窗口
        if rank == 0 and render_during_training:
            activate_wow_window()
            
        env = WowKeyEnv(render_mode=render_mode, **env_kwargs)
        env = Monitor(env) # Wrap with Monitor for SB3 logging
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == '__main__':
    print("=" * 60)
    print(" 魔兽世界AI代理 - PPO训练脚本 (M-1阶段) ")
    print("=" * 60)
    print("本脚本使用PPO算法训练AI代理在魔兽世界中执行击杀和拾取任务。")
    print("\\n主要功能键:")
    print("  F8:  (在提示后) 开始训练。请确保已切换到游戏窗口。")
    print("  F9:  在训练过程中，暂停或恢复训练。")
    print("  F10: 请求停止训练。模型将在当前数据收集完成后保存并退出。")
    print("  Ctrl+C: 强制中断训练 (模型会尝试保存)。")
    print("-" * 60)

    # --- 询问用户是否准备好开始训练 ---
    print("\\nPPO训练准备开始")
    print("请确保:")
    print("1. 魔兽世界已经运行并且可见")
    print("2. 角色处于可以开始战斗的位置")
    print("3. 游戏UI和控制设置已经配置好")
    print("=" * 50)
    
    reply = input("你准备好开始训练了吗? (y/n): ")
    if reply.lower() != 'y':
        print("训练已取消")
        sys.exit(0)
    
    # --- 显示当前窗口以帮助调试 ---
    if HAVE_WINDOW_UTILS:
        print("\\n当前窗口列表:")
        all_titles = gw.getAllTitles()
        for title in all_titles:
            if title.strip():  # 忽略空标题
                print(f"  - {title}")
        
    # --- Configuration ---
    TRAIN_NEW_MODEL = True  # True: Start new training; False: Load and continue
    LOAD_MODEL_PATH = "D:/wow_ai/runs/ppo_models/ppo_wow_agent_m1_stage" # Path without .zip for SB3 load
                                                                            # (used if TRAIN_NEW_MODEL = False)
    TOTAL_TIMESTEPS = 500000   # 增加到50万步，约可训练2500个回合，大约2-3小时
    LOG_DIR = os.path.join(project_root, "runs", "ppo_logs_m1_stage") 
    MODEL_SAVE_NAME_PREFIX = "ppo_wow_agent_m1" # 使用M-1标识
    MODEL_SAVE_PATH_FINAL = os.path.join(project_root, "runs", "ppo_models", f"{MODEL_SAVE_NAME_PREFIX}_final")
    CHECKPOINT_SAVE_PATH = os.path.join(project_root, "runs", "ppo_models_checkpoints_m1") 
    CHECKPOINT_SAVE_FREQ = 5000 # Save every 5000 steps (relative to one env)
    
    RENDER_FIRST_ENV_DURING_TRAINING = False  # Set to True if you want to see the first environment during training
    NUM_ENVIRONMENTS = 1  # Number of parallel environments
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # --- End Configuration ---

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH_FINAL), exist_ok=True)
    os.makedirs(CHECKPOINT_SAVE_PATH, exist_ok=True)

    print("Creating vectorized environment...")
    if NUM_ENVIRONMENTS == 1:
        env = DummyVecEnv([lambda: make_env(render_during_training=RENDER_FIRST_ENV_DURING_TRAINING)()])
    else: # Not recommended for this GUI-interactive environment initially
        env = SubprocVecEnv([make_env(i, render_during_training=(i==0 and RENDER_FIRST_ENV_DURING_TRAINING)) for i in range(NUM_ENVIRONMENTS)])
    print("Vectorized environment created.")

    # --- Define policy_kwargs using the CustomCombinedExtractor ---
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        # net_arch defines the network layers *after* the features_extractor_class output
        # If CustomCombinedExtractor outputs, e.g., 128 (CNN) + 7 (others) = 135 features,
        # these layers will take that 135-dim vector as input.
        net_arch=dict(pi=[128, 64], vf=[128, 64]) # 增加网络容量
    )
    # --- End policy_kwargs definition ---

    model = None # Initialize model variable
    if TRAIN_NEW_MODEL:
        print("Training a new PPO model from scratch using CustomCombinedExtractor...")
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=0.0003,      
            n_steps=2048, # Number of steps to run for each environment per update
            batch_size=64,           
            n_epochs=10,             
            gamma=0.99,              
            gae_lambda=0.8,         # 按ChatGPT建议：从0.95降到0.8，让奖励更局部化，击杀奖励回溯到F动作
            clip_range=0.25,  # 从0.2增加到0.25，提高探索空间        
            ent_coef=0.02,  # 从0.01增加到0.02，增强探索性          
            vf_coef=0.5,             
            max_grad_norm=0.5,       
            tensorboard_log=LOG_DIR,
            verbose=1,               
            device=device            
        )
        print("PPO model instance created (random initialization).")
    else:
        load_path_full = LOAD_MODEL_PATH + ".zip"
        if os.path.exists(load_path_full):
            print(f"Loading existing PPO model from: {load_path_full}")
            # When loading, SB3 uses the saved policy_kwargs.
            # If CustomCombinedExtractor class definition is available (as it is in this script),
            # it should load correctly.
            model = PPO.load(
                load_path_full, 
                env=env, 
                device=device, 
                tensorboard_log=LOG_DIR
                # No need to pass policy_kwargs here if it was saved with the model correctly
            )
            print("Model loaded. Continuing training...")
        else:
            print(f"ERROR: Model file not found at {load_path_full}. Cannot load model.")
            sys.exit(1)
            
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_SAVE_FREQ // NUM_ENVIRONMENTS, 1),
        save_path=CHECKPOINT_SAVE_PATH,
        name_prefix=MODEL_SAVE_NAME_PREFIX + "_checkpoint"
    )
    
    # 训练控制回调 (暂停/恢复/停止)
    training_control_callback = TrainingControlCallback(verbose=1)

    print(f"Starting PPO training for {TOTAL_TIMESTEPS} timesteps...")
    print(f"TensorBoard logs will be saved to: {LOG_DIR}")
    if RENDER_FIRST_ENV_DURING_TRAINING and NUM_ENVIRONMENTS > 0:
        print("The first environment's window will be rendered during training (可以看到训练过程)")
    
    # 等待用户按下F8触发训练开始
    wait_for_trigger_key('f8')
    
    start_train_time = time.time()
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, training_control_callback],  # 使用多个回调
            log_interval=1, # Log training info every episode
            progress_bar=True
        )
    except KeyboardInterrupt: # Catches Ctrl+C
        print("\\nTraining interrupted by user (Ctrl+C). Saving model before exiting...")
    except Exception as e: # Catches other errors during model.learn()
        print(f"\\nAn error occurred during training: {e}")
        traceback.print_exc()
        print("Attempting to save model before exiting due to error...")
    finally:
        # Ensure keyboard listeners are cleaned up regardless of how training ends
        if training_control_callback.keyboard_available and training_control_callback._keyboard_hooked:
            keyboard.unhook_all() # Ensure unhook is called
            training_control_callback._keyboard_hooked = False
            print("Keyboard listeners for training control have been unhooked in finally block.")

        if model is not None: # Ensure model exists before saving
            print(f"\\nSaving final model to {MODEL_SAVE_PATH_FINAL}.zip ...")
            model.save(MODEL_SAVE_PATH_FINAL)
            print("Model saved.")
        else:
            print("No model instance was created or loaded, nothing to save.")
            
        end_train_time = time.time()
        training_duration = end_train_time - start_train_time
        print(f"Training finished. Total duration: {training_duration:.2f} seconds ({training_duration/3600:.2f} hours).")
        
        if env is not None: # Ensure env exists before closing
            env.close()
        print("Environment closed. Training script finished.")