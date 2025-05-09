# D:\wow_ai\scripts\test_env_key.py (Manual Input Version)
import gymnasium as gym
import numpy as np
import cv2
import sys
import os
import time
import traceback

# --- 动态路径设置 ---
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir) # wow_ai
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added project root '{project_root}' to sys.path")
except NameError:
    project_root = None
# --- 路径设置结束 ---

try:
    from wow_rl.envs.wow_key_env import WowKeyEnv
    print("Imported WowKeyEnv successfully.")
except ImportError as e:
    print(f"ImportError: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during WowKeyEnv import: {e}")
    traceback.print_exc()
    sys.exit(1)

print("Creating WowKeyEnv instance...")
try:
    # WowKeyEnv 内部 RS, send_key 仍禁用, ES 已启用
    env = WowKeyEnv(render_mode="human", max_steps=30) # 设置一个合适的测试步数
    print("Environment created.")
except Exception as e:
    print(f"Error creating WowKeyEnv: {e}")
    traceback.print_exc()
    sys.exit(1)

print("Resetting environment...")
try:
    obs, info = env.reset() 
    print("Environment reset complete. Starting simulation loop...")
except Exception as e:
    print(f"Error during env.reset(): {e}")
    traceback.print_exc()
    if 'env' in locals(): env.close()
    sys.exit(1)

total_reward = 0
step_count = 0

print("\n--- Starting MANUAL STEP Test Loop (ErrorMessageSensor Enabled) ---")
print("INSTRUCTIONS:")
print("1. For each step, the script will prompt you to enter an action (0-4).")
print("   ACTION MAPPING:", env.action_names) # 打印动作映射
print("2. Before entering an action, MANUALLY set up the game state (e.g., trigger error messages).")
print("3. After entering an action, observe the 'WowKeyEnv Feed' window and console output.")
print("4. Press 'q' then Enter at the action prompt to quit early.")
print("------------------------------------------------------------------")

try:
    for i in range(env.max_steps):
        step_count = i + 1
        print(f"\n--- Step {step_count}/{env.max_steps} ---")
        
        action_input = input(f"Enter action (0-4, or 'q' to quit) for step {step_count}: ")
        
        if action_input.lower() == 'q':
            print("Quitting test loop...")
            break
            
        try:
            action = int(action_input)
            if action not in env.action_space: raise ValueError("Action out of bounds")
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 4, or 'q'. Skipping step.")
            continue # 跳过这个无效输入，进入下一个循环

        action_name = env.action_names.get(action, f"action_{action}")
        print(f"Action selected (manual): {action} ({action_name})")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        sel_flag = info.get('sel_flag', 'N/A') # 这个还是假的(0)
        yolo_prob_target = info.get('yolo_prob_target', 'N/A') # 这个还是假的(0)
        need_face = info.get('need_face', 'N/A') # 这个现在应该来自ErrorMessageSensor
        need_range = info.get('need_range', 'N/A') # 这个现在应该来自ErrorMessageSensor
        no_target_error = info.get('no_target_error', 'N/A') # 这个现在应该来自ErrorMessageSensor
        
        print(f"Step result: Reward={reward:.2f}, Sel={sel_flag}(Fake), P(Gwyx)={yolo_prob_target:.4f}(Fake), "
              f"FaceErr={need_face}, RangeErr={need_range}, NoTgtErr={no_target_error}, "
              f"Term={terminated}, Trunc={truncated}")

        if terminated or truncated:
            print(f"Episode finished after {step_count} steps.")
            break
        
        # 在这里不加额外的waitKey或sleep，依赖 WowKeyEnv.render() 中的 waitKey(30)

except KeyboardInterrupt: # 处理 Ctrl+C
    print("\nTest interrupted by user (Ctrl+C).")
except Exception as e:
    print(f"\nError during simulation loop at step {step_count}:")
    traceback.print_exc()
finally:
    print(f"\n--- Test finished ---")
    print(f"Total steps executed: {step_count}")
    print(f"Total reward collected: {total_reward:.2f}")
    try:
        env.close() 
        print("Environment closed.")
    except Exception as e:
        print(f"Error closing environment: {e}")
    print("Cleanup finished.")