# D:\wow_ai\scripts\train_ppo_wow.py
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
from stable_baselines3.common.callbacks import CheckpointCallback
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
        # Ensure these kwargs match your WowKeyEnv's __init__ and are optimal
        env_kwargs = {
            "roi_reward": (250, 35, 205, 64), 
            "template_path_select": r'D:\wow_ai\data\target_frame_template.png',
            "match_thresh_select": 0.50, 
            "template_path_dead": r'D:\wow_ai\data\template_target_dead.png',
            "match_thresh_dead": 0.5, # Using the lower threshold that worked
            "max_steps": 200 # Max steps per episode
        }
        env = WowKeyEnv(render_mode=render_mode, **env_kwargs)
        env = Monitor(env) # Wrap with Monitor for SB3 logging
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == '__main__':
    # --- Configuration ---
    TRAIN_NEW_MODEL = False  # True: Start new training; False: Load and continue
    LOAD_MODEL_PATH = "D:/wow_ai/runs/ppo_models/ppo_wow_agent_initial_test" # Path without .zip for SB3 load
                                                                            # (used if TRAIN_NEW_MODEL = False)
    TOTAL_TIMESTEPS = 100000   # Start with a small number for initial testing (e.g., 10k-50k)
    LOG_DIR = os.path.join(project_root, "runs", "ppo_logs_from_scratch") 
    MODEL_SAVE_NAME_PREFIX = "ppo_wow_agent_scratch" # Prefix for saved models
    MODEL_SAVE_PATH_FINAL = os.path.join(project_root, "runs", "ppo_models", f"{MODEL_SAVE_NAME_PREFIX}_final")
    CHECKPOINT_SAVE_PATH = os.path.join(project_root, "runs", "ppo_models_checkpoints") 
    CHECKPOINT_SAVE_FREQ = 5000 # Save every 5000 steps (relative to one env)
    
    RENDER_FIRST_ENV_DURING_TRAINING = False # Set to True to watch the agent learn (slower)
    NUM_ENVIRONMENTS = 1 
    
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
        net_arch=dict(pi=[64, 64], vf=[64, 64]) 
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
            gae_lambda=0.95,         
            clip_range=0.2,          
            ent_coef=0.0,            
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
            
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_SAVE_FREQ // NUM_ENVIRONMENTS, 1),
        save_path=CHECKPOINT_SAVE_PATH,
        name_prefix=MODEL_SAVE_NAME_PREFIX + "_checkpoint"
    )

    print(f"Starting PPO training for {TOTAL_TIMESTEPS} timesteps...")
    print(f"TensorBoard logs will be saved to: {LOG_DIR}")
    if RENDER_FIRST_ENV_DURING_TRAINING and NUM_ENVIRONMENTS > 0:
        print("The first environment's window will be rendered during training (can be SLOW).")
    
    start_train_time = time.time()
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            log_interval=1, # Log training info every episode
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C). Saving model before exiting...")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        traceback.print_exc()
        print("Attempting to save model before exiting due to error...")
    finally:
        if model is not None: # Ensure model exists before saving
            print(f"\nSaving final model to {MODEL_SAVE_PATH_FINAL}.zip ...")
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