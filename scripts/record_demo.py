# D:\wow_ai\scripts\record_demo.py
# (Version with extended actions including healing skills V and T)
import cv2
import numpy as np
import time
import sys
import os
import traceback
import keyboard # Using 'keyboard' library for press/release events
import pickle # For saving the recorded data

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
    from wow_rl.envs.wow_key_env import WowKeyEnv # For state analysis and screen grabbing
    print("Imported WowKeyEnv successfully.")
except ImportError as e:
    print(f"ImportError: {e}"); traceback.print_exc(); sys.exit(1)

# --- Configuration ---
OUTPUT_FILENAME = "wow_demo_data_with_heals.pkl" # New filename for this recording session
RECORDING_SAMPLE_RATE_HZ = 5 # For continuous actions like movement
DATA_SAVE_PATH = os.path.join(project_root, "data", OUTPUT_FILENAME)


# --- Define our "Expert Action Set" and Key Mappings ---
ACTION_MAP = {
    "no_op": 0,         # Explicit No Operation
    "tab": 1,
    "f": 2,             # Main Attack
    "g": 3,             # Select Previous Target / Corpse
    "r": 4,             # Another skill
    "w": 5,             # Forward
    "s": 6,             # Backward
    "a": 7,             # Strafe Left
    "d": 8,             # Strafe Right
    "space": 9,         # Jump
    "shift+tab": 10,    # Select Previous Enemy
    "v": 11,            # Rejuvenation (Heal 1)
    "t": 12,            # Healing Touch (Heal 2)
    # Add more keys as needed, e.g., "1": 13, "2": 14 for action bar slots
}
# Reverse map for easy printing of action names (optional)
ACTION_ID_TO_NAME = {v: k for k, v in ACTION_MAP.items()}

# --- Global state for recording ---
is_recording = False
recorded_data = [] 
active_movement_keys = set() # Not strictly used in the current simplified movement detection
quit_flag = False # Main loop exit flag

# --- Functions for Keyboard Callbacks (Hotkeys) ---
def start_stop_recording():
    global is_recording, recorded_data
    is_recording = not is_recording
    if is_recording:
        # Check if file exists and prompt if user wants to overwrite or append (for simplicity, we overwrite)
        if os.path.exists(DATA_SAVE_PATH):
            print(f"Warning: File {DATA_SAVE_PATH} already exists. Starting a new recording will overwrite it upon saving.")
        recorded_data.clear() 
        print(f"\n[RECORDING STARTED] - Target file: {DATA_SAVE_PATH}")
        print("Press F8 again to PAUSE/RESUME, F9 to STOP and SAVE.")
    else:
        print("\n[RECORDING PAUSED] Press F8 to RESUME.")

def save_and_quit():
    global is_recording, quit_flag
    was_recording = is_recording # Store current state
    is_recording = False # Ensure recording stops
    
    print("\n[STOP & SAVE TRIGGERED]")
    if not recorded_data and was_recording:
        print("No data was recorded in this session to save.")
    elif not recorded_data and not was_recording:
        print("Recording was not active, no data to save.")
    else:
        print(f"Attempting to save {len(recorded_data)} data points to {DATA_SAVE_PATH}...")
        try:
            with open(DATA_SAVE_PATH, "wb") as f:
                pickle.dump(recorded_data, f)
            print(f"Successfully saved data to {DATA_SAVE_PATH}")
        except Exception as e:
            print(f"Error saving data: {e}"); traceback.print_exc()
            
    quit_flag = True # Signal main loop to exit
    print("Exiting recording script...")


# --- Main Recording Logic ---
def record_demonstrations():
    global is_recording, recorded_data, active_movement_keys, quit_flag

    print("Initializing WowKeyEnv for screen grabbing and state analysis...")
    env_util = None
    try:
        env_util = WowKeyEnv(render_mode=None) # We only need its utility methods
        print("WowKeyEnv instance created for utility.")
    except Exception as e: print(f"Error creating WowKeyEnv: {e}"); traceback.print_exc(); return

    try:
        print("Registering global hotkeys...")
        keyboard.add_hotkey('f8', start_stop_recording, suppress=True)
        keyboard.add_hotkey('f9', save_and_quit, suppress=True)
        print("  - F8: Start/Pause/Resume Recording")
        print("  - F9: Stop Recording, Save Data & Exit Script")
        print("Hotkeys registered. Press F8 to begin recording.")
    except Exception as e:
        print(f"ERROR registering global hotkeys: {e}. Ensure script has Admin privileges."); return

    last_continuous_sample_time = time.time()
    print("\n--- Demonstration Recording Loop ---")
    print("Ensure WoW game window is ACTIVE for correct screen capture and key recording.")
    
    last_recorded_action_time = 0
    min_action_interval = 0.05 # To prevent hyper-fast duplicate action recordings

    while not quit_flag:
        current_time = time.time()
        action_to_record_this_iteration = None
        action_key_pressed_this_iteration = None

        if is_recording:
            # 1. Check for instantaneous (event-based) action keys
            # Order matters if keys can be pressed simultaneously (e.g., Shift+Tab vs Tab)
            if keyboard.is_pressed('shift') and keyboard.is_pressed('tab'):
                action_key_pressed_this_iteration = "shift+tab"
            elif keyboard.is_pressed('tab'): action_key_pressed_this_iteration = "tab"
            elif keyboard.is_pressed('f'): action_key_pressed_this_iteration = "f"
            elif keyboard.is_pressed('g'): action_key_pressed_this_iteration = "g"
            elif keyboard.is_pressed('r'): action_key_pressed_this_iteration = "r"
            elif keyboard.is_pressed('v'): action_key_pressed_this_iteration = "v"
            elif keyboard.is_pressed('t'): action_key_pressed_this_iteration = "t"
            elif keyboard.is_pressed('space'): action_key_pressed_this_iteration = "space"
            
            if action_key_pressed_this_iteration:
                action_to_record_this_iteration = ACTION_MAP.get(action_key_pressed_this_iteration)

            # 2. Check for continuous movement keys (sampled)
            # If an event action already occurred, we might skip movement for this exact instant
            # or decide on a priority. For now, let's allow movement to be sampled too.
            current_movement_key = None
            if keyboard.is_pressed('w'): current_movement_key = 'w'
            elif keyboard.is_pressed('s'): current_movement_key = 's'
            elif keyboard.is_pressed('a'): current_movement_key = 'a'
            elif keyboard.is_pressed('d'): current_movement_key = 'd'

            if current_movement_key and (current_time - last_continuous_sample_time >= 1.0 / RECORDING_SAMPLE_RATE_HZ):
                # If no event action, movement is the action.
                # If there was an event action, we might decide to record event OR movement, or both if state changes fast.
                # For simplicity, if an event action (like F) happened, we prioritize it.
                # If only movement, record movement.
                if action_to_record_this_iteration is None:
                    action_to_record_this_iteration = ACTION_MAP.get(current_movement_key)
                    action_key_pressed_this_iteration = current_movement_key # For logging
                    last_continuous_sample_time = current_time # Reset timer only if movement is sampled
            
            # 3. Record if an action was identified (and not too soon after the last one)
            if action_to_record_this_iteration is not None and \
               (current_time - last_recorded_action_time > min_action_interval):
                frame = env_util.grab.grab()
                if frame is not None and frame.size > 0:
                    state_dict = env_util.analyze_state_from_frame(frame)
                    if state_dict:
                        recorded_data.append((state_dict, action_to_record_this_iteration))
                        action_name_for_log = ACTION_ID_TO_NAME.get(action_to_record_this_iteration, "Unknown")
                        print(f"REC ({len(recorded_data)}): State + Action {action_to_record_this_iteration} (Key: {action_key_pressed_this_iteration} -> {action_name_for_log})")
                        last_recorded_action_time = current_time
                else:
                    print("Warning: Frame grab failed during recording.")
            
            # 4. If no specific key pressed, consider recording No_Op (optional, can make data noisy)
            # elif (current_time - last_continuous_sample_time >= 1.0 / RECORDING_SAMPLE_RATE_HZ):
            #     # Only record No_Op if enough time has passed since *any* last recording
            #     if current_time - last_recorded_action_time > (1.0 / RECORDING_SAMPLE_RATE_HZ):
            #         frame = env_util.grab.grab()
            #         if frame is not None and frame.size > 0:
            #             state_dict = env_util.analyze_state_from_frame(frame)
            #             if state_dict:
            #                 recorded_data.append((state_dict, ACTION_MAP["no_op"]))
            #                 print(f"REC ({len(recorded_data)}): State + Action {ACTION_MAP['no_op']} (No_Op)")
            #                 last_continuous_sample_time = current_time
            #                 last_recorded_action_time = current_time


        time.sleep(0.01) # Main loop sleep

    # --- Cleanup ---
    print("Unregistering all keyboard hotkeys...")
    keyboard.unhook_all() 
    if env_util:
        env_util.close() # Close the env utility instance
    print("Recording script loop finished.")

if __name__ == "__main__":
    print("--- WoW Demonstration Recording Script (v_with_heals) ---")
    print("!!! IMPORTANT: Run this script with ADMINISTRATOR PRIVILEGES for global hotkeys !!!")
    print("Ensure WoW is in the desired window mode (e.g., Maximized Windowed).")
    record_demonstrations()