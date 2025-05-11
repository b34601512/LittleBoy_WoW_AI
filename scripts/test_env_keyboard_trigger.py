# D:\wow_ai\scripts\test_env_keyboard_trigger.py
# (Version: Prints env.step() direct returns, uses 'keyboard' lib)
import cv2
import numpy as np
import time
import sys
import os
import traceback
import keyboard # Using 'keyboard' library
# import threading # Not needed with 'keyboard' library for this setup

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
    from wow_rl.envs.wow_key_env import WowKeyEnv # Using the latest WowKeyEnv
    print("Imported WowKeyEnv successfully.")
except ImportError as e:
    print(f"ImportError: {e}"); traceback.print_exc(); sys.exit(1)
except Exception as e:
    print(f"Import Error: {e}"); traceback.print_exc(); sys.exit(1)

# --- Global variables for keyboard callbacks ---
trigger_analysis = False # For F10
action_to_execute = None # For number keys 0-4
quit_flag = False      # For ESC

# --- keyboard library callback functions ---
def handle_f10():
    global trigger_analysis
    if not trigger_analysis: # Prevent rapid re-triggering if an analysis is ongoing
        print("\n[Callback] F10 detected by keyboard lib!")
        trigger_analysis = True

def handle_action_key(action_num_str): # Receives the key as a string
    global action_to_execute
    try:
        action_num = int(action_num_str)
        if 0 <= action_num <= 4: # Assuming 5 actions (0-4)
             print(f"\n[Callback] Number key {action_num} detected by keyboard lib!")
             action_to_execute = action_num
        # else:
             # print(f"\n[Callback] Ignored number key: {action_num_str}")
    except ValueError:
        # print(f"\n[Callback] Non-numeric key pressed, mapped to action: {action_num_str}")
        pass # Ignore non-numeric keys if a char was passed

def handle_esc():
    global quit_flag
    print("\n[Callback] ESC detected by keyboard lib!")
    quit_flag = True

# --- Register keyboard hooks ---
try:
    print("Registering keyboard hooks (using 'keyboard' library)...")
    keyboard.add_hotkey('f10', handle_f10, suppress=False) # suppress=False allows F10 to work elsewhere too
    for i in range(5): # Actions 0 through 4
        keyboard.add_hotkey(str(i), lambda i_val=i: handle_action_key(str(i_val)), suppress=False)
    keyboard.add_hotkey('esc', handle_esc, suppress=False)
    print("Keyboard hooks registered.")
except Exception as e:
    print(f"ERROR registering keyboard hooks: {e}")
    print("Please ensure the script is run with ADMINISTRATOR PRIVILEGES.")
    # sys.exit(1) # Allow to continue to see if it works without admin for some users

# --- Main test function ---
def run_keyboard_test():
    global trigger_analysis, action_to_execute, quit_flag
    print("Creating WowKeyEnv instance...")
    env = None
    try:
        env = WowKeyEnv(render_mode="human", max_steps=30) # WowKeyEnv now integrates is_dead logic
        print("Environment created.")
        print("Performing initial reset...")
        obs, info_reset = env.reset() # Capture initial info
        print("Initial reset complete.")
        print(f"Initial Info from env.reset(): {info_reset}")
    except Exception as e:
        print(f"Error creating or resetting WowKeyEnv: {e}"); traceback.print_exc(); return

    print("\n--- MANUAL STEP Test Loop (Prints env.step() returns) ---")
    print("INSTRUCTIONS:")
    print("1. Ensure the WoW game window is ACTIVE (has focus).")
    print("2. Press F10 to grab the current screen and analyze the state (prints 'F10 Analysis Result').")
    print("3. Press number keys 0-4 to execute the corresponding action.")
    print(f"   ACTION MAPPING: {env.action_names}")
    print("   (After an action, its direct reward/term/trunc will be printed).")
    print("4. Press ESC to quit the script.")
    print("---------------------------------------------------")
    time.sleep(1) # Give a moment to read instructions

    step_count_display = 0

    try:
        while not quit_flag:
            current_step_reward = 0.0
            current_step_terminated = False
            current_step_truncated = False
            current_step_info_from_step = {} # Info directly from this step's env.step()

            # Process F10 analysis trigger
            if trigger_analysis:
                print("\n--- F10 Triggered: Analyzing current state ---")
                # F10 analysis uses a fresh grab, independent of env's internal last_grabbed_frame
                frame_for_f10_analysis = env.grab.grab()
                analysis_result = env.analyze_state_from_frame(frame_for_f10_analysis)
                if analysis_result:
                    print("F10 Analysis Result (independent of step):")
                    for key, value in analysis_result.items():
                        print(f"  {key}: {value}")
                else:
                    print("F10 State analysis failed or returned None.")
                trigger_analysis = False # Reset flag

            # Process action execution trigger
            if action_to_execute is not None:
                step_count_display +=1
                action = action_to_execute
                action_name = env.action_names.get(action, f"action_{action}")
                print(f"\n--- Step {step_count_display} (Action key {action} - {action_name}) ---")
                
                obs, current_step_reward, current_step_terminated, current_step_truncated, current_step_info_from_step = env.step(action)
                
                print(f"Action {action} ({action_name}) executed.")
                print(f"  >> env.step() RETURN: Reward={current_step_reward:.2f}, Terminated={current_step_terminated}, Truncated={current_step_truncated}")
                # print(f"  >> env.step() INFO: {current_step_info_from_step}") # Optional: very verbose

                if current_step_terminated or current_step_truncated:
                    print(f"  EPISODE ENDED by env.step()! Term={current_step_terminated}, Trunc={current_step_truncated}")
                    print("  Environment should be reset by the RL agent in a real training loop.")
                    print("  In this script, you might need to press F10 to see final state or ESC to quit.")
                    # For this test script, if terminated, we might want to break or prompt for reset
                    # quit_flag = True # Optionally quit after episode ends

                action_to_execute = None # Reset action flag
            
            # Render based on the latest info available to the env (updated by step or reset)
            if env.render_mode == "human":
                # Pass the info from the last env.step() or env.reset() to render
                # If no step was taken, info_reset or env.last_analysis_result could be used
                info_to_display = current_step_info_from_step if current_step_info_from_step else env.current_analysis # 使用 env.current_analysis
                env.render(info_for_display=info_to_display)
            
            time.sleep(0.02) # Main loop sleep, keep it short

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Stopping...")
        quit_flag = True
    except Exception as e:
        print(f"\nError in main loop:"); traceback.print_exc()
    finally:
        print("\n--- Test finished ---")
        try:
            print("Unregistering keyboard hooks...")
            keyboard.unhook_all()
        except Exception as e:
            print(f"Error unhooking keyboard: {e}")
        
        if env is not None:
            try: env.close(); print("Environment closed.")
            except Exception as e: print(f"Error closing environment: {e}")
        print("Cleanup finished.")

if __name__ == "__main__":
    print("--- Keyboard Triggered Environment Test (v_step_return_print) ---")
    print("!!! IMPORTANT: Ensure this script is run with ADMINISTRATOR PRIVILEGES !!!")
    run_keyboard_test()