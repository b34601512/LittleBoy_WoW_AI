# D:\wow_ai\wow_rl\utils\interactor_keys.py
try:
    import keyboard  # 尝试导入keyboard库
    USE_KEYBOARD = True
    print("使用keyboard库进行按键操作")
except ImportError:
    import pydirectinput
    USE_KEYBOARD = False
    print("未能加载keyboard库，回退到pydirectinput")

import time

# 确保pydirectinput在游戏窗口中生效的设置
if not USE_KEYBOARD:
    pydirectinput.PAUSE = 0.01  # 减少按键之间的默认暂停时间
    pydirectinput.FAILSAFE = False

# 使用了解决方案对方向键进行修复
KEY_MAPPINGS = {
    'left': 'left',
    'right': 'right',
    'up': 'up',
    'down': 'down',
    # 添加更多映射，如必要
}

# 定义使用keyboard库的按键函数
def press_key_with_keyboard(key):
    if key == 'left':
        keyboard.press_and_release('left')
    elif key == 'right':
        keyboard.press_and_release('right')
    elif key == ']':
        keyboard.press_and_release(']')
    else:
        keyboard.press_and_release(key)

# Updated ACTION_TABLE based on the 8 core actions
KEY_FUNCTIONS = {
    0: lambda: press_key_with_keyboard('f') if USE_KEYBOARD else pydirectinput.press('f'),
    1: lambda: press_key_with_keyboard('g') if USE_KEYBOARD else pydirectinput.press('g'),
    2: lambda: press_key_with_keyboard(']') if USE_KEYBOARD else pydirectinput.press(']'),
    3: lambda: press_key_with_keyboard('left') if USE_KEYBOARD else pydirectinput.press(KEY_MAPPINGS['left']),
    4: lambda: press_key_with_keyboard('right') if USE_KEYBOARD else pydirectinput.press(KEY_MAPPINGS['right']),
    5: lambda: press_key_with_keyboard('w') if USE_KEYBOARD else pydirectinput.press('w'),
    6: lambda: None,  # No_Op
    7: lambda: press_key_with_keyboard('tab') if USE_KEYBOARD else pydirectinput.press('tab'),
}

# 添加键名映射字典
KEY_MAP = {
    'f': 0,
    'g': 1,
    ']': 2,
    'left': 3,
    'right': 4,
    'w': 5,
    None: 6,
    'tab': 7
}

def send_key(action, wait: float = 0.05, debug: bool = False):
    """
    Executes the keyboard command corresponding to the action number or key name.
    Args:
        action: The action number (0-7) or key name ('f', 'g', etc.).
        wait (float): Time to wait after sending the key (in seconds).
        debug (bool): Whether to print debug information.
    """
    # 如果action是字符串，尝试将其转换为动作编号
    original_action = action
    if isinstance(action, str):
        if debug:
            print(f"Debug[send_key]: 接收到按键 '{action}'")
        if action in KEY_MAP:
            action = KEY_MAP[action]
            if debug:
                print(f"Debug[send_key]: 映射到动作 {action}")
        else:
            print(f"Warning: 无效的按键名 '{action}'")
            return
    
    # 现在action应该是一个整数
    if action in KEY_FUNCTIONS:
        try:
            if debug:
                try:
                    action_name = list(KEY_MAP.keys())[list(KEY_MAP.values()).index(action)]
                    print(f"Debug[send_key]: 执行动作 {action} ({action_name})")
                except:
                    print(f"Debug[send_key]: 执行动作 {action}")
                
            action_func = KEY_FUNCTIONS[action]
            if action_func is not None:  # Check if it's a valid function (not None for No_Op)
                action_func()  # Execute the lambda function
                if debug:
                    print(f"Debug[send_key]: 按键执行完成 {action}")
                if action != 6:  # Don't sleep for No_Op
                    time.sleep(wait)
        except Exception as e:
            print(f"Error sending key for action {original_action}->{action}: {e}")
    else:
        print(f"Warning: 无效的动作编号 {action}")

# 测试键盘库是否正常工作
if USE_KEYBOARD:
    try:
        print("测试keyboard库是否工作正常...")
        print("keyboard库可访问")
    except Exception as e:
        print(f"警告: keyboard可能无法正常工作: {e}")
else:
    try:
        print("测试pydirectinput是否工作正常...")
        # 只是测试API调用，不实际发送按键
        pydirectinput.position()
        print("pydirectinput API可访问")
    except Exception as e:
        print(f"警告: pydirectinput可能无法正常工作: {e}")

if __name__ == '__main__':
    # Quick test for key functions
    print("测试按键 (将依次按下 F, G, ], Left, Right, W, Tab)")
    print("请切换到文本编辑器或游戏窗口来查看按键效果")
    time.sleep(3)
    for i in range(8): # Test all actions 0-7
        if i == 6: # No_Op
            print(f"动作 {i}: No_Op (无操作)")
            time.sleep(1)
            continue
        print(f"发送动作 {i}...")
        send_key(i, wait=0.5, debug=True)
        time.sleep(1) # Wait a bit longer between different key tests
    print("按键测试完成")
    
    # 测试字符串键名
    print("\n使用键名测试按键")
    for key in ['f', 'g', ']', 'left', 'right', 'w', 'tab']:
        print(f"发送按键 '{key}'...")
        send_key(key, wait=0.5, debug=True)
        time.sleep(1)
    print("键名测试完成")