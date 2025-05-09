# D:\wow_ai\scripts\test_env_keyboard_trigger.py
import cv2
import numpy as np
import time
import sys
import os
import traceback
from pynput import keyboard # 导入pynput用于监听
import threading # 用于后台监听

# --- 动态路径设置 ---
# ... (和之前脚本一样) ...
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir) # wow_ai
    if project_root not in sys.path: sys.path.insert(0, project_root); print(f"Added project root '{project_root}' to sys.path")
except NameError: project_root = None
# --- 路径设置结束 ---

try:
    from wow_rl.envs.wow_key_env import WowKeyEnv # 导入修改后的 WowKeyEnv
    print("Imported WowKeyEnv successfully.")
except ImportError as e: print(f"ImportError: {e}"); traceback.print_exc(); sys.exit(1)
except Exception as e: print(f"Import Error: {e}"); traceback.print_exc(); sys.exit(1)

# --- 全局变量用于线程间通信 ---
f10_pressed = threading.Event() # F10按键事件
f11_action = None # F11按键触发的动作
f11_event = threading.Event() # F11按键事件
esc_pressed = threading.Event() # ESC按键事件
listener_thread = None # 监听器线程
key_listener = None # pynput监听器对象

# --- 键盘按键处理函数 ---
def on_press(key):
    global f11_action, f11_event, f10_pressed, esc_pressed
    try:
        # print(f'Key {key} pressed') # 调试用
        if key == keyboard.Key.f10:
            print("\n[Listener] F10 detected!")
            f10_pressed.set() # 设置F10事件
        elif key == keyboard.Key.esc:
            print("\n[Listener] ESC detected!")
            esc_pressed.set() # 设置ESC事件
            return False # 停止监听器
        # --- 可以添加F11或其他按键来触发动作 ---
        # 例如，我们让数字键 0-4 触发对应的 env.step()
        elif hasattr(key, 'char') and key.char in ['0', '1', '2', '3', '4']:
            action_num = int(key.char)
            print(f"\n[Listener] Number key {action_num} detected!")
            f11_action = action_num # 记录要执行的动作
            f11_event.set() # 设置动作执行事件

    except AttributeError:
        # 处理特殊按键（如果需要）
        pass
    except Exception as e:
        print(f"Error in on_press: {e}")

def start_keyboard_listener():
    global key_listener
    print("[Main Thread] Starting keyboard listener...")
    # 创建并启动监听器
    # non_blocking=True 可能不适用于所有平台，这里使用标准方式
    # 需要一个单独的线程来运行监听器，防止阻塞主线程
    with keyboard.Listener(on_press=on_press) as kl:
         key_listener = kl # 保存监听器对象引用
         kl.join() # 等待监听器停止（当on_press返回False时）
    print("[Listener Thread] Keyboard listener stopped.")


# --- 主测试函数 ---
def run_keyboard_test():
    global listener_thread, f10_pressed, f11_action, f11_event, esc_pressed
    print("Creating WowKeyEnv instance...")
    env = None
    try:
        # 使用 render_mode="human" 可以在屏幕上看到ROI框
        env = WowKeyEnv(render_mode="human") 
        print("Environment created.")
    except Exception as e:
        print(f"Error creating WowKeyEnv: {e}"); traceback.print_exc(); return

    # 启动键盘监听线程
    listener_thread = threading.Thread(target=start_keyboard_listener, daemon=True)
    listener_thread.start()
    time.sleep(1) # 等待监听器启动

    print("\n--- Keyboard Trigger Test Loop ---")
    print("INSTRUCTIONS:")
    print("1. Ensure the WoW game window is ACTIVE (has focus).")
    print("2. Press F10 to grab the current screen and analyze the state.")
    print("3. Press number keys 0-4 to execute the corresponding action:")
    print("     0: Tab, 1: Shift+Tab, 2: G, 3: F(Attack), 4: No_Op")
    print("4. Press ESC to quit the script.")
    print("   (You might need to press ESC multiple times if focus is lost)")
    print("---------------------------------------------------")

    try:
        while not esc_pressed.is_set():
            # 检查是否有F10事件
            if f10_pressed.wait(timeout=0.05): # 非阻塞等待0.05秒
                print("--- F10 Triggered: Analyzing current state ---")
                current_frame = env.grab.grab() # 获取当前帧
                analysis_result = env.analyze_state_from_frame(current_frame) # 分析状态
                if analysis_result:
                    print("Analysis Result:")
                    for key, value in analysis_result.items():
                        print(f"  {key}: {value}")
                else:
                    print("State analysis failed.")
                f10_pressed.clear() # 清除事件，等待下一次按下

            # 检查是否有动作执行事件 (F11 或 数字键)
            if f11_event.wait(timeout=0.05):
                 if f11_action is not None:
                     print(f"--- Number Key Triggered: Executing action {f11_action} ---")
                     # 执行动作 (env.step现在只发送按键和截图)
                     env.step(f11_action) 
                     print(f"Action {f11_action} executed. Press F10 to see the new state.")
                     f11_action = None # 重置动作
                 f11_event.clear() # 清除事件

            # 保持窗口响应 (如果 env.render() 不足以保持响应，可以取消注释下面这行)
            # cv2.waitKey(1) # 极短等待，允许处理窗口消息

            # 让主循环稍微休息一下，避免CPU空转过高
            # time.sleep(0.01) 

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Stopping...")
        esc_pressed.set() # 触发退出
    except Exception as e:
        print(f"\nError in main loop:")
        traceback.print_exc()
    finally:
        print("\n--- Test finished ---")
        if key_listener is not None:
             print("Attempting to stop keyboard listener...")
             # keyboard.Listener.stop() # 可以尝试调用 stop 方法
             # 或者依赖 esc_pressed 和 on_press 返回 False
        if listener_thread is not None:
             listener_thread.join(timeout=1.0) # 等待监听线程结束
             if listener_thread.is_alive(): print("Warning: Listener thread did not exit.")
        
        if env is not None:
             try: env.close(); print("Environment closed.")
             except Exception as e: print(f"Error closing environment: {e}")
        
        print("Cleanup finished.")


if __name__ == "__main__":
    print("--- Keyboard Triggered Environment Test ---")
    # 确保脚本有管理员权限可能有助于键盘钩子工作，但通常不需要
    # input("Press Enter to start...") # 可选：给时间准备
    run_keyboard_test()