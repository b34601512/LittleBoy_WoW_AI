import time
import sys
import os
import traceback

# 导入窗口管理库
try:
    import pygetwindow as gw
    HAVE_WINDOW_UTILS = True
except ImportError:
    HAVE_WINDOW_UTILS = False
    print("警告: 未找到pygetwindow库，无法自动激活窗口")

# 优先使用pydirectinput以便更好地兼容游戏
try:
    import pydirectinput
    HAVE_DIRECT_INPUT = True
except ImportError:
    HAVE_DIRECT_INPUT = False
    print("警告: 未找到pydirectinput库，将尝试使用keyboard库")
    
# 备用方案使用keyboard
try:
    import keyboard
    HAVE_KEYBOARD = True
except ImportError:
    HAVE_KEYBOARD = False
    print("警告: 未找到keyboard库，无法进行按键模拟")

def activate_wow_window():
    """尝试激活魔兽世界窗口并返回窗口对象"""
    if not HAVE_WINDOW_UTILS:
        print("无法激活窗口: pygetwindow库未安装或导入失败。")
        return None
    
    try:
        print("搜索WoW窗口...")
        # 先搜索英文标题
        wow_windows = [w for w in gw.getAllTitles() if 'World of Warcraft' in w]
        # 如果没有找到，尝试搜索简化的英文标题
        if not wow_windows:
            wow_windows = [w for w in gw.getAllTitles() if 'Warcraft' in w]
        # 如果还没找到，尝试搜索中文标题
        if not wow_windows:
            wow_windows = [w for w in gw.getAllTitles() if '魔兽世界' in w]
            
        if wow_windows:
            target_window_title = wow_windows[0]
            print(f"找到窗口: {target_window_title}")
            wow_window = gw.getWindowsWithTitle(target_window_title)
            if wow_window:
                wow_window = wow_window[0]
                # 激活窗口的同时确保它不是最小化的
                if wow_window.isMinimized:
                    wow_window.restore()
                    time.sleep(0.5)  # 给恢复窗口一些时间
                
                # 交替最小化再激活，在某些系统上更可靠
                try:
                    wow_window.minimize()
                    time.sleep(0.3)
                    wow_window.restore()
                    time.sleep(0.3)
                except Exception as e:
                    print(f"窗口交替操作失败 (非致命): {e}")
                
                # 最终激活
                wow_window.activate()
                time.sleep(0.5)  # 给激活窗口更多时间
                
                # 验证窗口是否成功激活
                try:
                    if wow_window.isActive:
                        print(f"窗口 '{target_window_title}' 已成功激活.")
                        # 打印窗口位置信息，帮助调试
                        print(f"窗口位置: 左={wow_window.left}, 上={wow_window.top}, 宽={wow_window.width}, 高={wow_window.height}")
                        return wow_window
                    else:
                        print(f"警告: 尝试激活窗口失败。请手动点击游戏窗口。")
                        return None
                except Exception as e:
                    print(f"检查窗口激活状态时出错 (非致命): {e}")
                    return wow_window  # 仍然返回找到的窗口
            else:
                print(f"未找到标题为 '{target_window_title}' 的窗口实例。")
                return None
        else:
            print("未找到WoW窗口。请确保游戏正在运行并且窗口标题包含'World of Warcraft'、'Warcraft'或'魔兽世界'。")
            return None
    except Exception as e:
        print(f"激活窗口时出错: {e}")
        traceback.print_exc()
        return None

def press_key(key, delay_after_press=0.1, retry_attempts=2):
    """按下并释放指定的键，支持重试机制"""
    success = False
    error_msg = None
    
    # 首先尝试使用DirectInput（优先）
    if HAVE_DIRECT_INPUT:
        for attempt in range(retry_attempts + 1):
            if attempt > 0:
                print(f"  重试DirectInput按键 ({attempt}/{retry_attempts})...")
            try:
                print(f"  正在通过DirectInput发送按键: '{key}'")
                # 对于特殊键名进行转换
                direct_key = key
                if key == 'space':
                    direct_key = 'spacebar'  # pydirectinput使用spacebar而不是space
                
                pydirectinput.press(direct_key)
                print(f"  DirectInput按键 '{key}' 发送完毕. 等待 {delay_after_press} 秒...")
                time.sleep(delay_after_press)
                success = True
                break
            except Exception as e:
                error_msg = str(e)
                print(f"  通过DirectInput发送按键 '{key}' 时出错: {e}")
                if attempt < retry_attempts:
                    time.sleep(0.2)  # 重试前短暂等待
                else:
                    print(f"  DirectInput发送失败，将尝试备用方法...")
    
    # 如果DirectInput失败或不可用，尝试使用keyboard库
    if not success and HAVE_KEYBOARD:
        try:
            print(f"  正在通过keyboard库发送按键: '{key}'")
            keyboard.press_and_release(key)
            print(f"  keyboard按键 '{key}' 发送完毕. 等待 {delay_after_press} 秒...")
            time.sleep(delay_after_press)
            success = True
        except Exception as e:
            error_msg = str(e)
            print(f"  通过keyboard库发送按键 '{key}' 时也出错: {e}")
    
    if not success:
        print(f"  所有按键方法都失败了。无法发送按键 '{key}'. 错误: {error_msg}")
    
    return success

def wait_for_trigger_key(trigger_key='f8', action_message="开始执行"):
    """等待用户按下触发键"""
    print(f"\n请先手动切换到魔兽世界游戏窗口！")
    print(f"然后返回此控制台，看到提示后按下 {trigger_key.upper()} 键来 {action_message}...")
    
    if HAVE_KEYBOARD:
        key_pressed_flag = [False]
        
        def on_trigger_key_press(e):
            if e.name.lower() == trigger_key.lower():
                key_pressed_flag[0] = True
        
        hook = keyboard.on_press(on_trigger_key_press)
        
        while not key_pressed_flag[0]:
            time.sleep(0.1)
        
        keyboard.unhook(hook)
        print(f"\n{trigger_key.upper()} 已按下，{action_message}！")
    else:
        input(f"(keyboard库不可用) 请按回车键以模拟 {trigger_key.upper()} 并 {action_message}...")
        print(f"\n回车已按下，{action_message}！")
    
    time.sleep(0.3)

_stop_sequence_requested = False  # 全局标志，用于F10停止
_test_paused = False  # 全局标志，用于F9暂停
_keyboard_hook_active = False  # 跟踪键盘钩子是否活动

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
            if _test_paused:  # 如果暂停了，自动恢复以允许停止
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
            keyboard.unhook_all()  # 尝试移除所有钩子
        except Exception as e:
            # keyboard库有时在没有活动钩子时调用unhook_all会出错，或者权限问题
            print(f"移除键盘监听时出现轻微问题 (可忽略): {e}")
        _keyboard_hook_active = False
        print("测试控制键盘监听已移除。")

def show_input_methods_status():
    """显示当前可用的输入方法状态"""
    print("\n当前输入方法状态:")
    print(f"  DirectInput: {'可用' if HAVE_DIRECT_INPUT else '不可用'}")
    print(f"  Keyboard库: {'可用' if HAVE_KEYBOARD else '不可用'}")
    print(f"  窗口管理: {'可用' if HAVE_WINDOW_UTILS else '不可用'}")
    
    if not HAVE_DIRECT_INPUT:
        print("\n强烈建议安装pydirectinput库以提高按键可靠性:")
        print("  pip install pydirectinput")
    
    if not HAVE_WINDOW_UTILS:
        print("\n建议安装pygetwindow库以启用窗口自动激活:")
        print("  pip install pygetwindow")

def main():
    global _stop_sequence_requested, _test_paused
    _stop_sequence_requested = False  # 重置标志
    _test_paused = False  # 重置标志

    print("=" * 60)
    print(" 魔兽世界直接按键测试脚本 (改进版) ")
    print("=" * 60)
    print("本脚本用于直接向魔兽世界窗口发送按键序列，以测试按键交互是否正常。")
    print("重要提示: 请在提示后，先手动点击激活魔兽世界游戏窗口，然后再按F8键开始测试。")
    print("\n主要功能键:")
    print("  F8:  (在提示后) 开始执行预设的按键序列。")
    print("  F9:  在按键序列执行过程中，暂停或恢复序列。")
    print("  F10: 在按键序列执行过程中，请求停止当前的按键序列。")
    print("  Ctrl+C: 强制中断测试脚本。")
    print("-" * 60)
    
    # 显示输入方法状态
    show_input_methods_status()
    
    if not HAVE_KEYBOARD and not HAVE_DIRECT_INPUT:
        print("\n严重警告: 未找到任何可用的按键模拟库，无法进行测试！")
        print("请安装至少一种按键模拟库:")
        print("  pip install pydirectinput  # 推荐")
        print("  pip install keyboard")
        return
    
    if HAVE_WINDOW_UTILS:
        print("\n当前所有窗口 (用于调试): ")
        all_titles = gw.getAllTitles()
        if all_titles:
            filtered_titles = [title for title in all_titles if title.strip()]
            for i, title in enumerate(filtered_titles[:20]):  # 只显示前20个非空窗口标题
                print(f"  {i+1}. {title}")
            if len(filtered_titles) > 20:
                print(f"  ... 还有 {len(filtered_titles) - 20} 个窗口未显示")
        else:
            print("  未能获取到任何窗口标题。")
    
    # 定义测试按键序列: (按键, 按下后等待时间)
    key_sequence = [
        ('f', 0.8),      # 攻击 (0.8秒钟)
        ('g', 0.8),      # 互动/选尸 (0.8秒钟)
        (']', 0.8),      # 拾取 (0.8秒钟)
        ('space', 0.8),  # 跳跃 (0.8秒钟)
        ('w', 1.0),      # 前进 (1秒钟)
    ]
    
    print("\n将执行以下测试按键序列:")
    for i, (key_name, delay) in enumerate(key_sequence):
        print(f"  步骤 {i+1}: 按键 '{key_name}', 执行后等待 {delay}秒")
    
    # 等待用户确认开始测试
    wait_for_trigger_key('f8', action_message="开始按键序列测试")
    
    # 尝试激活WoW窗口
    print("\n尝试激活魔兽世界窗口...")
    wow_window = activate_wow_window()
    if wow_window:
        print("窗口激活尝试完成。等待1.5秒确保焦点切换...")
        time.sleep(1.5)  # 给系统和游戏更多时间响应焦点切换
    else:
        print("窗口自动激活失败或无法进行。")
        print("请手动点击魔兽世界窗口使其成为活动窗口，然后按回车键继续...")
        input("按回车键继续...")
    
    # 开始测试键盘钩子
    setup_test_key_hooks()
    
    try:
        print("\n开始发送按键到目标窗口...")
        for i, (key_name, delay_time) in enumerate(key_sequence):
            print(f"\n[序列步骤 {i+1}/{len(key_sequence)}] 即将执行: '{key_name}'")
            if _stop_sequence_requested:
                print("按键序列已由用户通过F10停止。")
                break
            
            # 处理暂停逻辑
            while _test_paused and HAVE_KEYBOARD:
                if _stop_sequence_requested:
                    print("按键序列在暂停期间由用户通过F10停止。")
                    break
                print("...测试已暂停，按F9恢复...")
                time.sleep(0.5)  # 暂停时检查频率
            if _stop_sequence_requested:
                break
            
            # 发送按键前再次确认窗口激活
            if HAVE_WINDOW_UTILS and wow_window:
                try:
                    if not wow_window.isActive:
                        print(f"警告: 魔兽世界窗口不是活动窗口。尝试重新激活...")
                        wow_window.activate()
                        time.sleep(0.5)
                except Exception as e:
                    print(f"重新检查/激活窗口时出错: {e}")
            
            # 执行按键
            print(f"执行: 按键 '{key_name}'")
            if not press_key(key_name, delay_time):
                print(f"警告: 按键 '{key_name}' 未能成功发送，继续下一个按键...")
            else:
                print(f"按键 '{key_name}' 操作及其延时已完成。")
            
            # 不同按键间的额外短暂延迟
            if i < len(key_sequence) - 1:
                time.sleep(0.1)
        
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
        if _keyboard_hook_active and HAVE_KEYBOARD:
            remove_test_key_hooks() 