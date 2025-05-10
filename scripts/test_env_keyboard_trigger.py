# D:\wow_ai\scripts\test_env_keyboard_trigger.py (Using 'keyboard' library)
import cv2
import numpy as np
import time
import sys
import os
import traceback
import keyboard # <--- 导入 keyboard 库
# import threading # 不再需要 threading

# --- 动态路径设置 ---
# ... (保持不变) ...
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir) # wow_ai
    if project_root not in sys.path: sys.path.insert(0, project_root); print(f"Added project root '{project_root}' to sys.path")
except NameError: project_root = None
# --- 路径设置结束 ---

try:
    # 使用我们上一版本（包含 startWindowThread）的 WowKeyEnv
    from wow_rl.envs.wow_key_env import WowKeyEnv 
    print("Imported WowKeyEnv successfully.")
except ImportError as e: print(f"ImportError: {e}"); traceback.print_exc(); sys.exit(1)
except Exception as e: print(f"Import Error: {e}"); traceback.print_exc(); sys.exit(1)

# --- 全局变量用于回调函数与主循环通信 ---
trigger_analysis = False # F10 是否被按下
action_to_execute = None # 数字键对应的动作
quit_flag = False      # ESC 是否被按下

# --- keyboard 库的回调函数 ---
def handle_f10():
    global trigger_analysis
    print("\n[Callback] F10 detected!")
    trigger_analysis = True

def handle_action_key(action_num):
    global action_to_execute
    # 检查是否是有效动作编号 (0-4)
    if 0 <= action_num <= 4:
         print(f"\n[Callback] Number key {action_num} detected!")
         action_to_execute = action_num
    else:
         print(f"\n[Callback] Ignored number key: {action_num}")


def handle_esc():
    global quit_flag
    print("\n[Callback] ESC detected!")
    quit_flag = True

# --- 注册键盘事件钩子 ---
# 注意：这里直接注册，不需要单独线程
try:
    print("Registering keyboard hooks...")
    keyboard.add_hotkey('f10', handle_f10)
    # 为数字键 0 到 4 注册钩子
    for i in range(5):
         # 使用 lambda 来传递正确的动作编号给处理函数
         keyboard.add_hotkey(str(i), lambda i=i: handle_action_key(i))
    keyboard.add_hotkey('esc', handle_esc)
    print("Keyboard hooks registered.")
except Exception as e:
    print(f"ERROR registering keyboard hooks: {e}")
    print("Please ensure the script is run with administrator privileges.")
    sys.exit(1)

# --- 主测试函数 ---
def run_keyboard_test():
    global trigger_analysis, action_to_execute, quit_flag
    print("Creating WowKeyEnv instance...")
    env = None
    try:
        env = WowKeyEnv(render_mode="human") # WowKeyEnv 内部 RS/ES/send_key 仍是启用状态
        print("Environment created.")
        print("Performing initial reset...")
        env.reset()
        print("Initial reset complete.")
    except Exception as e: print(f"Error creating or resetting WowKeyEnv: {e}"); traceback.print_exc(); return

    print("\n--- Keyboard Trigger Test Loop (using 'keyboard' lib) ---")
    print("INSTRUCTIONS:")
    print("1. Ensure the WoW game window is ACTIVE (has focus).")
    print("2. Press F10 to grab the current screen and analyze the state.")
    print("3. Press number keys 0-4 to execute the corresponding action.")
    print("4. Press ESC to quit the script.")
    print("---------------------------------------------------")

    try:
        while not quit_flag:
            # 检查是否有 F10 触发
            if trigger_analysis:
                print("--- F10 Triggered: Analyzing current state ---")
                current_frame = env.grab.grab()
                analysis_result = env.analyze_state_from_frame(current_frame) # 调用分析
                if analysis_result:
                    print("Analysis Result:")
                    for key, value in analysis_result.items(): print(f"  {key}: {value}")
                else: print("State analysis failed.")
                trigger_analysis = False # 重置标志

            # 检查是否有动作要执行
            if action_to_execute is not None:
                action = action_to_execute
                action_name = env.action_names.get(action, f"action_{action}")
                print(f"--- Number Key Triggered: Executing action {action} ({action_name}) ---")
                # 调用 env.step (只发送按键和截图)
                # 注意：我们暂时不关心 step 返回的 obs/reward 等，因为分析由F10触发
                env.step(action) 
                print(f"Action {action} executed. Press F10 to see the new state.")
                action_to_execute = None # 重置标志

            # 主循环调用 env.render() 来显示窗口并处理其事件
            # render 内部有 waitKey(30)，会给CPU一些喘息时间
            if env.render_mode == "human":
                env.render()
            
            # 如果没有渲染，需要加个短暂 sleep 避免空转
            time.sleep(0.01) 

            # (Optional) Check if ESC was pressed directly by render's waitKey
            # This might be unreliable with the keyboard library hooks active
            # key = cv2.waitKey(1) & 0xFF
            # if key == 27:
            #    print("ESC detected by waitKey in main loop.")
            #    quit_flag = True


    except KeyboardInterrupt: print("\nCtrl+C detected. Stopping...")
    except Exception as e: print(f"\nError in main loop:"); traceback.print_exc()
    finally:
        print("\n--- Test finished ---")
        try:
            print("Unregistering keyboard hooks...")
            keyboard.unhook_all() # 注销所有钩子
        except Exception as e: print(f"Error unhooking keyboard: {e}")
        
        if env is not None:
            try: env.close(); print("Environment closed.")
            except Exception as e: print(f"Error closing environment: {e}")
        print("Cleanup finished.")

if __name__ == "__main__":
    print("--- Keyboard Triggered Environment Test (using 'keyboard' lib) ---")
    print("!!! IMPORTANT: Ensure this script is run with ADMINISTRATOR PRIVILEGES !!!")
    # input("Press Enter to start...")
    run_keyboard_test()