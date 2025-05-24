# D:\wow_ai\scripts\wow_direct_test.py
import time
import sys
import os
# import keyboard # keyboard 会在下面的try-except中导入
import traceback

try:
    import pygetwindow as gw
    HAVE_WINDOW_UTILS = True
except ImportError:
    HAVE_WINDOW_UTILS = False
    print("警告: 未找到pygetwindow库，无法自动激活窗口")

# 定义HAVE_KEYBOARD
try:
    import keyboard # 在这里导入
    HAVE_KEYBOARD = True
except ImportError:
    HAVE_KEYBOARD = False
    # 此处可以不打印，因为main函数中会有统一的检查和提示

def activate_wow_window():
    """尝试激活魔兽世界窗口"""
    if not HAVE_WINDOW_UTILS:
        print("无法激活窗口: pygetwindow库未安装或导入失败。")
        return False
    
    try:
        print("搜索WoW窗口...")
        wow_windows = [w for w in gw.getAllTitles() if 'World of Warcraft' in w]
        if not wow_windows:
            wow_windows = [w for w in gw.getAllTitles() if 'Warcraft' in w]
            
        if wow_windows:
            target_window_title = wow_windows[0]
            print(f"找到窗口: {target_window_title}")
            wow_window = gw.getWindowsWithTitle(target_window_title)
            if wow_window:
                wow_window = wow_window[0]
                # 尝试最小化再恢复，某些情况下有助于获取焦点
                # if wow_window.isMinimized:
                #     wow_window.restore()
                # elif wow_window.isMaximized:
                #     wow_window.minimize()
                #     wow_window.maximize()
                # else:
                #     wow_window.minimize()
                #     wow_window.restore()
                wow_window.activate()
                time.sleep(0.1) # 给操作系统一点反应时间
                if wow_window.isActive:
                    print(f"窗口 '{target_window_title}' 已成功激活.")
                    return True
                else:
                    print(f"警告: 尝试激活窗口 '{target_window_title}' 后，它仍未处于激活状态。请手动点击游戏窗口。")
                    return False
            else:
                print(f"未找到标题为 '{target_window_title}' 的窗口实例。")
                return False
        else:
            print("未找到WoW窗口。请确保游戏正在运行并且窗口标题包含'World of Warcraft'或'Warcraft'。")
            return False
    except Exception as e:
        print(f"激活窗口时出错: {e}")
        return False

def press_key(key, delay_after_press=0.1):
    """按下并释放指定的键，然后在返回前等待指定的延迟时间"""
    try:
        print(f"  正在发送按键: '{key}'")
        keyboard.press_and_release(key)
        print(f"  按键 '{key}' 发送完毕. 等待 {delay_after_press} 秒...")
        time.sleep(delay_after_press)
        return True
    except Exception as e:
        print(f"  发送按键 '{key}' 时出错: {e}")
        return False

def wait_for_trigger_key(trigger_key='f8', action_message="开始执行"):
    """等待用户按下触发键"""
    print(f"\n请先手动切换到魔兽世界游戏窗口！")
    print(f"然后返回此控制台，看到提示后按下 {trigger_key.upper()} 键来 {action_message}...")
    
    key_pressed_flag = [False]
    
    def on_trigger_key_press(e):
        if e.name.lower() == trigger_key.lower():
            key_pressed_flag[0] = True
    
    hook = None
    if HAVE_KEYBOARD:
        hook = keyboard.on_press(on_trigger_key_press)
    else:
        input(f"(keyboard库不可用) 请按回车键以模拟 {trigger_key.upper()} 并 {action_message}...")
        key_pressed_flag[0] = True # 模拟按键
        return # 直接返回，不进入下面的循环
        
    while not key_pressed_flag[0]:
        time.sleep(0.1)
    
    if hook and HAVE_KEYBOARD:
        keyboard.unhook(hook)
    
    print(f"\n{trigger_key.upper()} 已按下，{action_message}！")
    time.sleep(0.3)

_stop_sequence_requested = False # 全局标志，用于F10停止
_test_paused = False # 全局标志，用于F9暂停
_keyboard_hook_active = False # 跟踪键盘钩子是否活动

def _handle_test_keys(e):
    """处理F9和F10的全局回调函数"""
    global _stop_sequence_requested, _test_paused
    key_name = e.name.lower()

    if key_name == 'f9':
        _test_paused = not _test_paused
        status = "暂停" if _test_paused else "恢复"
        print(f"\n测试序列已{status}! 再次按 F9 键 {('恢复' if _test_paused else '暂停')}...")
        if _test_paused and _stop_sequence_requested:
            print("注意: 暂停操作取消了先前的停止请求。")
            _stop_sequence_requested = False

    elif key_name == 'f10':
        if not _stop_sequence_requested:
            print(f"\nF10 已按下，请求停止按键序列...")
            _stop_sequence_requested = True
            if _test_paused: # 如果暂停了，自动恢复以允许停止
                _test_paused = False
                print("已自动恢复测试序列以便完成停止流程。")

def setup_test_key_hooks():
    """设置测试期间的F9/F10按键钩子"""
    global _keyboard_hook_active
    if not _keyboard_hook_active and HAVE_KEYBOARD:
        keyboard.on_press(_handle_test_keys)
        _keyboard_hook_active = True
        print("\n测试控制快捷键已激活:")
        print("  F9: 暂停/恢复序列")
        print("  F10: 停止序列")
    elif not HAVE_KEYBOARD:
        print("警告: keyboard库不可用，F9/F10快捷键将无法工作。")

def remove_test_key_hooks():
    """移除测试期间的F9/F10按键钩子"""
    global _keyboard_hook_active
    if _keyboard_hook_active and HAVE_KEYBOARD:
        try:
            keyboard.unhook_all() # 尝试移除所有钩子
        except Exception as e:
            # keyboard库有时在没有活动钩子时调用unhook_all会出错，或者权限问题
            print(f"移除键盘监听时出现轻微问题 (可忽略): {e}")
        _keyboard_hook_active = False
        print("测试控制键盘监听已移除。")

def main():
    global _stop_sequence_requested, _test_paused
    _stop_sequence_requested = False # 重置标志
    _test_paused = False # 重置标志

    print("=" * 60)
    print(" 魔兽世界直接按键测试脚本 ")
    print("=" * 60)
    print("本脚本用于直接向魔兽世界窗口发送按键序列，以测试按键交互是否正常。")
    print("重要提示: 请在提示后，先手动点击激活魔兽世界游戏窗口，然后再按F8键开始测试。")
    print("\n主要功能键:")
    print("  F8:  (在提示后) 开始执行预设的按键序列。")
    print("  F9:  在按键序列执行过程中，暂停或恢复序列。")
    print("  F10: 在按键序列执行过程中，请求停止当前的按键序列。")
    print("  Ctrl+C: 强制中断测试脚本。")
    print("-" * 60)
    
    if not HAVE_KEYBOARD:
        print("警告: 未找到keyboard库或导入失败，F8/F9/F10快捷键功能将不可用。测试将尝试使用回车键继续。")

    if HAVE_WINDOW_UTILS:
        print("\n当前所有窗口 (用于调试，非自动激活): ")
        all_titles = gw.getAllTitles()
        if all_titles:
            for i, title in enumerate(all_titles):
                if title.strip():
                    print(f"  {i+1}. {title}")
        else:
            print("  未能获取到任何窗口标题。")
    else:
        print("\n警告: pygetwindow库不可用，无法列出窗口或尝试自动激活窗口。")
    
    key_sequence = [
        ('w', 2.0),     # 前进 (2秒钟)
        ('s', 2.0),     # 后退 (2秒钟)
        ('space', 1.0), # 跳跃 (按下后等待1秒)
        ('tab', 1.0),   # 选择目标 (按下后等待1秒)
    ]
    print("\n将执行以下测试按键序列:")
    for i, (key_name, delay) in enumerate(key_sequence):
        print(f"  步骤 {i+1}: 按键 '{key_name}', 执行后等待 {delay}秒")
    
    wait_for_trigger_key('f8', action_message="开始按键序列测试")

    print("\n尝试激活魔兽世界窗口...")
    window_activated = activate_wow_window()
    if window_activated:
        print("窗口激活尝试完毕。给系统1秒时间响应焦点切换...")
        time.sleep(1) # 额外等待，确保焦点切换完成
    else:
        print("窗口激活失败或无法进行。请务必手动确保魔兽世界窗口是当前活动窗口！")
        # 即使激活失败，也给用户一个机会手动操作
        # input("按回车键继续尝试发送按键...")

    print("\n开始发送按键到目标窗口...")
    setup_test_key_hooks()

    try:
        for i, (key_name, delay_time) in enumerate(key_sequence):
            print(f"\n[序列步骤 {i+1}/{len(key_sequence)}] 即将执行: '{key_name}'")
            if _stop_sequence_requested:
                print("按键序列已由用户通过F10停止。")
                break
            
            while _test_paused and HAVE_KEYBOARD:
                if _stop_sequence_requested:
                    print("按键序列在暂停期间由用户通过F10停止。")
                    break
                print("...测试已暂停，按F9恢复...")
                time.sleep(0.5) # 暂停时检查频率
            if _stop_sequence_requested:
                break

            print(f"执行: 按键 '{key_name}'")
            if not press_key(key_name, delay_time):
                 print(f"警告: 按键 '{key_name}' 未能成功发送。可能是keyboard库问题。")
            print(f"按键 '{key_name}' 操作及其延时已完成。")
            
            if i < len(key_sequence) -1 :
                time.sleep(0.05)

        if not _stop_sequence_requested:
            print("\n所有按键序列执行完成!")

    except Exception as e:
        print(f"在执行按键序列时发生严重错误: {e}")
        traceback.print_exc()
    finally:
        remove_test_key_hooks()
        _stop_sequence_requested = False
        _test_paused = False
        
    print("\n测试脚本运行结束!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n测试被用户 (Ctrl+C) 中断")
    except Exception as e:
        print(f"\n测试脚本遇到未处理的严重错误: {e}") 
        traceback.print_exc()
    finally:
        # 确保在任何情况下都尝试移除钩子
        if _keyboard_hook_active and HAVE_KEYBOARD: # 检查是否需要移除
            remove_test_key_hooks() 