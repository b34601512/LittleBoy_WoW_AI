# wow_rl/utils/interactor_keys.py
import pydirectinput
import time

# --- 按键映射字典 ---
# 动作编号 -> lambda 函数来执行按键
# 使用 pydirectinput 来发送指令
KEY_FUNCTIONS = {
    0: lambda: pydirectinput.press('tab'),          # 动作 0: 按 Tab (选中最近敌人)
    1: lambda: (pydirectinput.keyDown('shift'),     # 动作 1: 按 Shift+Tab
                time.sleep(0.02),                 # 短暂等待确保 shift 按下
                pydirectinput.press('tab'),
                time.sleep(0.02),
                pydirectinput.keyUp('shift')),
    2: lambda: pydirectinput.press('g'),            # 动作 2: 按 G (上一个敌对)
    3: lambda: pydirectinput.press('f'),            # 动作 3: 按 F (攻击目标 - 假设 F 是攻击键)
    4: lambda: None                                 # 动作 4: 什么都不做
}
# --- 按键映射结束 ---

def send_key(action: int, wait: float = 0.05):
    """
    根据动作编号执行对应的键盘指令。
    Args:
        action (int): 动作编号 (0-4)。
        wait (float): 执行按键后等待的时间 (秒)。
    """
    if action in KEY_FUNCTIONS:
        try:
            # print(f"Sending Key Action: {action}") # 用于调试
            action_func = KEY_FUNCTIONS[action]
            action_func() # 执行对应的 lambda 函数
            if action_func is not None: # 如果不是空动作，才等待
                time.sleep(wait)
        except Exception as e:
            print(f"Error sending key for action {action}: {e}")
    else:
        print(f"Warning: Invalid action number {action} received in send_key.")

# 可以添加一个简单的测试函数
if __name__ == '__main__':
    print("Testing Interactor Keys...")
    print("Pressing Tab (Action 0) in 3 seconds...")
    time.sleep(3)
    send_key(0)
    print("Tab sent.")
    time.sleep(1)
    print("Pressing Shift+Tab (Action 1) in 3 seconds...")
    time.sleep(3)
    send_key(1)
    print("Shift+Tab sent.")
    time.sleep(1)
    print("Pressing G (Action 2) in 3 seconds...")
    time.sleep(3)
    send_key(2)
    print("G sent.")
    time.sleep(1)
    print("Pressing F (Action 3) in 3 seconds...")
    time.sleep(3)
    send_key(3)
    print("F sent.")
    time.sleep(1)
    print("Doing Nothing (Action 4) in 3 seconds...")
    time.sleep(3)
    send_key(4)
    print("None sent.")
    print("Test finished.")