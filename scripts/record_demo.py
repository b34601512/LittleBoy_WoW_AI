# D:\wow_ai\scripts\record_demo.py
# (Version: Records Paladin actions including buffs/heals, AND image frames)
import cv2
import numpy as np
import time
import sys
import os
import traceback
import keyboard # Using 'keyboard' library
import pickle

# --- 动态路径设置 ---
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir) # wow_ai
    if project_root not in sys.path: sys.path.insert(0, project_root); print(f"Added '{project_root}' to sys.path")
except NameError: project_root = None
# --- 路径设置结束 ---

try:
    from wow_rl.envs.wow_key_env import WowKeyEnv # For state analysis and screen grabbing
    print("Imported WowKeyEnv successfully.")
except ImportError as e: print(f"ImportError: {e}"); traceback.print_exc(); sys.exit(1)

# --- Configuration ---
OUTPUT_FILENAME = "wow_paladin_demo_v1.pkl"
DATA_SAVE_PATH = os.path.join(project_root, "data", OUTPUT_FILENAME)
RECORDING_SAMPLE_RATE_HZ = 5 # For continuous movement sampling
FRAME_WIDTH_BC, FRAME_HEIGHT_BC = 54, 96 # Width, Height for saved frames (aligns with WowKeyEnv obs)

# --- Define our "Expert Action Set" and Key Mappings ---
ACTION_MAP = {
    "no_op": 0,
    "tab": 1,
    "f": 2,             # Main Attack (Seal/Judge Macro)
    "g": 3,             # Select Previous Target / Corpse
    "r": 4,             # Placeholder for another skill
    "w": 5,             # Forward
    "shift+1": 6,       # Strength Buff
    "shift+2": 7,       # Seal Buff
    "v": 8,             # Heal (Holy Light / Flash of Light)
    "space": 9,         # Jump
    "a": 10,            # Strafe Left
    "d": 11,            # Strafe Right
    "s": 12,            # Backward
    # Add "shift+tab": X if needed later
}
ACTION_ID_TO_NAME = {v: k for k, v in ACTION_MAP.items()}
MAX_ACTION_ID = max(ACTION_MAP.values()) # Should be 12 if map above is used

# --- Global state for recording ---
is_recording = False
recorded_data = [] # List to store (frame_array, structured_state_dict, action_id)
quit_flag = False

# --- Functions for Keyboard Callbacks (Hotkeys) ---
def start_stop_recording():
    global is_recording, recorded_data
    is_recording = not is_recording
    if is_recording:
        if os.path.exists(DATA_SAVE_PATH): print(f"Warning: {DATA_SAVE_PATH} exists. New recording will overwrite upon saving.")
        recorded_data.clear() 
        print(f"\n[RECORDING STARTED] -> {DATA_SAVE_PATH}\n  Press F8 to PAUSE/RESUME, F9 to STOP & SAVE.")
    else: print("\n[RECORDING PAUSED] Press F8 to RESUME.")

def save_and_quit():
    global is_recording, quit_flag, recorded_data, DATA_SAVE_PATH
    was_recording = is_recording; is_recording = False
    print("\n[STOP & SAVE TRIGGERED]")
    if recorded_data:
        print(f"Attempting to save {len(recorded_data)} data points to {DATA_SAVE_PATH}...")
        try:
            with open(DATA_SAVE_PATH, "wb") as f: pickle.dump(recorded_data, f)
            print(f"Successfully saved data to {DATA_SAVE_PATH}")
        except Exception as e: print(f"Error saving data: {e}"); traceback.print_exc()
    else: print("No data was recorded to save.")
    quit_flag = True; print("Exiting recording script.")

# --- Key press detection and action mapping ---
# Stores the action to be recorded for the current frame, prioritizing event keys
# This will be set by the keyboard library's more precise hotkey system.
action_to_log_event = None 

def set_action_event(key_name):
    global action_to_log_event
    if is_recording: # Only log if actively recording
        action_id = ACTION_MAP.get(key_name)
        if action_id is not None:
            action_to_log_event = action_id # Store the event-based action
            # print(f"[Event Key Pressed]: {key_name} -> Action {action_id}") # Debug

def register_event_hotkeys():
    # Instantaneous actions (Tab, F, G, R, V, Space, Buffs)
    # Using lambda with default argument to capture correct key_name
    keyboard.add_hotkey("tab", lambda: set_action_event("tab"), suppress=False)
    keyboard.add_hotkey("f", lambda: set_action_event("f"), suppress=False)
    keyboard.add_hotkey("g", lambda: set_action_event("g"), suppress=False)
    keyboard.add_hotkey("r", lambda: set_action_event("r"), suppress=False)
    keyboard.add_hotkey("v", lambda: set_action_event("v"), suppress=False)
    keyboard.add_hotkey("space", lambda: set_action_event("space"), suppress=False)
    keyboard.add_hotkey("shift+1", lambda: set_action_event("shift+1"), suppress=False)
    keyboard.add_hotkey("shift+2", lambda: set_action_event("shift+2"), suppress=False)
    # Add Shift+Tab if needed: keyboard.add_hotkey("shift+tab", lambda: set_action_event("shift+tab"), suppress=False)


# --- Main Recording Logic ---
def record_demonstrations():
    global is_recording, recorded_data, quit_flag, action_to_log_event

    print("Initializing WowKeyEnv for screen grabbing and state analysis...")
    env_util = None
    try:
        env_util = WowKeyEnv(render_mode=None) 
        print("WowKeyEnv instance created for utility.")
    except Exception as e: print(f"Error creating WowKeyEnv: {e}"); traceback.print_exc(); return

    try:
        print("Registering global F8/F9/ESC hotkeys...")
        keyboard.add_hotkey('f8', start_stop_recording, suppress=True)
        keyboard.add_hotkey('f9', save_and_quit, suppress=True)
        keyboard.add_hotkey('esc', save_and_quit, suppress=True) # Allow ESC to also save & quit
        print("  - F8: Start/Pause/Resume Recording")
        print("  - F9 or ESC: Stop Recording, Save Data & Exit Script")
        register_event_hotkeys() # Register a wider range of keys for actual data
        print("Hotkeys registered. Press F8 to begin recording.")
    except Exception as e: print(f"ERROR registering global hotkeys: {e}. Admin privileges?"); return

    last_continuous_sample_time = time.time()
    print("\n--- Demonstration Recording Loop ---")
    print("Ensure WoW game window is ACTIVE for correct screen capture and key recording.")
    
    while not quit_flag:
        current_time = time.time()
        action_this_iteration = None
        log_action_name = "No_Op_Default" # Default if nothing specific happens

        if is_recording:
            # 1. Prioritize event-based actions already set by hotkeys
            if action_to_log_event is not None:
                action_this_iteration = action_to_log_event
                log_action_name = ACTION_ID_TO_NAME.get(action_this_iteration, "UnknownEvent")
                action_to_log_event = None # Consume the event

            # 2. Check for continuous movement keys (sampled)
            # If no event action, check movement.
            if action_this_iteration is None:
                movement_key_pressed = None
                if keyboard.is_pressed('w'): movement_key_pressed = 'w'
                elif keyboard.is_pressed('s'): movement_key_pressed = 's'
                elif keyboard.is_pressed('a'): movement_key_pressed = 'a'
                elif keyboard.is_pressed('d'): movement_key_pressed = 'd'
                
                if movement_key_pressed and (current_time - last_continuous_sample_time >= 1.0 / RECORDING_SAMPLE_RATE_HZ):
                    action_this_iteration = ACTION_MAP.get(movement_key_pressed)
                    log_action_name = ACTION_ID_TO_NAME.get(action_this_iteration, "UnknownMove")
                    last_continuous_sample_time = current_time
            
            # 3. If any action (event or movement) was identified, record it.
            if action_this_iteration is not None:
                frame = env_util.grab.grab()
                if frame is not None and frame.size > 0:
                    # Get structured state (sel_flag, is_dead, errors, etc.)
                    structured_state_dict = env_util.analyze_state_from_frame(frame)
                    
                    # Get and resize the frame for BC model input
                    try:
                        frame_for_bc = cv2.resize(frame, (FRAME_WIDTH_BC, FRAME_HEIGHT_BC), interpolation=cv2.INTER_AREA)
                    except Exception as e_resize:
                        print(f"Error resizing frame for BC: {e_resize}. Skipping this data point.")
                        frame_for_bc = None # Indicate error

                    if structured_state_dict and frame_for_bc is not None:
                        # Save: (image_frame, structured_state_dict, action_id)
                        recorded_data.append((frame_for_bc, structured_state_dict, action_this_iteration))
                        print(f"REC ({len(recorded_data)}): Frame({frame_for_bc.shape}), State, Action {action_this_iteration} ({log_action_name})")
                else:
                    print("Warning: Frame grab failed during recording.")
            
            # 4. Optional: Record No_Op if no other action and enough time passed
            # elif (current_time - last_continuous_sample_time >= 1.0 / RECORDING_SAMPLE_RATE_HZ):
            # (Consider if No_Op data is truly valuable or just noise)

        time.sleep(0.01) # Main loop sleep

    print("Unregistering all keyboard hotkeys...")
    keyboard.unhook_all()
    if env_util: env_util.close()
    print("Recording script loop finished.")

if __name__ == "__main__":
    print("--- WoW Paladin Demonstration Recording Script (v_frames_and_heals) ---")
    print("!!! IMPORTANT: Run this script with ADMINISTRATOR PRIVILEGES for global hotkeys !!!")
    print(f"Output will be saved to: {DATA_SAVE_PATH}")
    record_demonstrations()