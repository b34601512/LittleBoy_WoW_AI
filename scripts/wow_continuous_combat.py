import time
import sys
import os
import traceback
import random

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
                
                # 尝试激活窗口
                try:
                    wow_window.activate()
                    time.sleep(0.5)  # 给激活窗口更多时间
                except Exception as e:
                    print(f"窗口激活操作失败 (非致命): {e}")
                
                try:
                    if wow_window.isActive:
                        print(f"窗口 '{target_window_title}' 已成功激活.")
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

def press_key(key, delay_after_press=0.1):
    """按下并释放指定的键"""
    success = False
    
    # 首先尝试使用DirectInput（优先）
    if HAVE_DIRECT_INPUT:
        try:
            print(f"  发送按键: '{key}'")
            # 对于特殊键名进行转换
            direct_key = key
            if key == 'space':
                direct_key = 'spacebar'  # pydirectinput使用spacebar而不是space
            
            pydirectinput.press(direct_key)
            time.sleep(delay_after_press)
            success = True
        except Exception as e:
            print(f"  通过DirectInput发送按键 '{key}' 时出错: {e}")
    
    # 如果DirectInput失败或不可用，尝试使用keyboard库
    if not success and HAVE_KEYBOARD:
        try:
            keyboard.press_and_release(key)
            time.sleep(delay_after_press)
            success = True
        except Exception as e:
            print(f"  通过keyboard库发送按键 '{key}' 时也出错: {e}")
    
    if not success:
        print(f"  所有按键方法都失败了。无法发送按键 '{key}'.")
    
    return success

def execute_combat_sequence():
    """执行一个完整的战斗序列"""
    # 攻击目标 (按F键)
    print("执行: 攻击目标 (F)")
    press_key('f', delay_after_press=0.2)
    
    # 等待攻击冷却和战斗进行
    wait_time = random.uniform(0.8, 1.2)  # 随机等待时间，让行为更自然
    time.sleep(wait_time)
    
    # 有一定概率尝试互动(G)和拾取(])操作
    if random.random() < 0.3:  # 30%概率
        print("执行: 互动/选尸 (G)")
        press_key('g', delay_after_press=0.2)
        time.sleep(random.uniform(0.4, 0.8))
        
        print("执行: 拾取 (])")
        press_key(']', delay_after_press=0.2)
        time.sleep(random.uniform(0.4, 0.8))

_stop_requested = False  # 全局标志，用于F10停止
_combat_paused = False   # 全局标志，用于F9暂停
_keyboard_hook_active = False  # 跟踪键盘钩子是否活动

def _handle_control_keys(e):
    """处理F9和F10的全局回调函数"""
    global _stop_requested, _combat_paused
    key_name = e.name.lower()

    if key_name == 'f9':
        _combat_paused = not _combat_paused
        status = "暂停" if _combat_paused else "恢复"
        print(f"\n战斗循环已{status}! 再次按 F9 键 {('恢复' if _combat_paused else '暂停')}...")

    elif key_name == 'f10':
        if not _stop_requested:
            print(f"\nF10 已按下，正在停止战斗循环...")
            _stop_requested = True
            if _combat_paused:  # 如果暂停了，自动恢复以便正常结束循环
                _combat_paused = False
                print("已自动恢复以便正常结束战斗循环。")

def setup_control_key_hooks():
    """设置F9/F10控制按键钩子"""
    global _keyboard_hook_active
    if not _keyboard_hook_active and HAVE_KEYBOARD:
        keyboard.on_press(_handle_control_keys)
        _keyboard_hook_active = True
        print("\n控制快捷键已激活:")
        print("  F9: 暂停/恢复战斗循环")
        print("  F10: 停止战斗循环")
        print("  Ctrl+C: 强制中断脚本")
    elif not HAVE_KEYBOARD:
        print("警告: keyboard库不可用，F9/F10快捷键将无法工作。")

def remove_control_key_hooks():
    """移除F9/F10控制按键钩子"""
    global _keyboard_hook_active
    if _keyboard_hook_active and HAVE_KEYBOARD:
        try:
            keyboard.unhook_all()
        except Exception as e:
            print(f"移除键盘监听时出现轻微问题 (可忽略): {e}")
        _keyboard_hook_active = False
        print("键盘监听已移除。")

def wait_for_start_signal(trigger_key='f8'):
    """等待用户按下开始键"""
    print(f"\n请先手动切换到魔兽世界游戏窗口！")
    print(f"然后返回此控制台，看到提示后按下 {trigger_key.upper()} 键来开始战斗循环...")
    
    if HAVE_KEYBOARD:
        key_pressed_flag = [False]
        
        def on_trigger_key_press(e):
            if e.name.lower() == trigger_key.lower():
                key_pressed_flag[0] = True
        
        hook = keyboard.on_press(on_trigger_key_press)
        
        while not key_pressed_flag[0]:
            time.sleep(0.1)
        
        keyboard.unhook(hook)
        print(f"\n{trigger_key.upper()} 已按下，开始战斗循环！")
    else:
        input(f"(keyboard库不可用) 请按回车键开始战斗循环...")
        print(f"\n回车已按下，开始战斗循环！")
    
    time.sleep(0.3)

def main():
    global _stop_requested, _combat_paused
    _stop_requested = False  # 重置标志
    _combat_paused = False   # 重置标志

    print("=" * 60)
    print(" 魔兽世界自动战斗循环脚本 ")
    print("=" * 60)
    print("本脚本将自动循环执行战斗操作，模拟基本战斗循环。")
    print("默认操作:")
    print("  F键: 攻击目标")
    print("  G键: 选择尸体/互动")
    print("  ]键: 拾取")
    print("-" * 60)

    # 检查必要的库
    if not HAVE_KEYBOARD and not HAVE_DIRECT_INPUT:
        print("\n严重警告: 未找到任何可用的按键模拟库，无法运行脚本！")
        print("请安装必要的库:")
        print("  pip install pydirectinput pygetwindow keyboard")
        return
    
    # 等待用户确认开始
    wait_for_start_signal('f8')
    
    # 尝试激活WoW窗口
    print("\n尝试激活魔兽世界窗口...")
    wow_window = activate_wow_window()
    if not wow_window:
        print("窗口自动激活失败或无法进行。")
        print("请手动点击魔兽世界窗口使其成为活动窗口，然后按回车键继续...")
        input("按回车键继续...")
    
    # 设置控制键钩子
    setup_control_key_hooks()
    
    try:
        print("\n开始自动战斗循环，按F9暂停，按F10停止...")
        combat_cycle = 0
        
        while not _stop_requested:
            combat_cycle += 1
            
            # 处理暂停逻辑
            while _combat_paused and HAVE_KEYBOARD:
                if _stop_requested:
                    print("战斗循环在暂停期间被停止。")
                    break
                print("战斗循环已暂停，按F9恢复...")
                time.sleep(0.5)
            
            if _stop_requested:
                break
            
            # 确保窗口处于激活状态
            if HAVE_WINDOW_UTILS and wow_window:
                try:
                    if not wow_window.isActive:
                        print(f"警告: 魔兽世界窗口不是活动窗口。尝试重新激活...")
                        wow_window.activate()
                        time.sleep(0.5)
                except Exception:
                    pass  # 忽略窗口激活错误，继续尝试发送按键
            
            # 执行战斗序列
            print(f"\n循环 #{combat_cycle}")
            execute_combat_sequence()
            
            # 随机等待一小段时间，避免机械式的重复行为
            time.sleep(random.uniform(0.2, 0.5))
            
        print("\n战斗循环已停止!")
        
    except KeyboardInterrupt:
        print("\n战斗循环被用户 (Ctrl+C) 中断")
    except Exception as e:
        print(f"\n运行时发生错误: {e}")
        traceback.print_exc()
    finally:
        remove_control_key_hooks()
        _stop_requested = False
        _combat_paused = False
    
    print("\n脚本运行结束!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n脚本被用户 (Ctrl+C) 中断")
    except Exception as e:
        print(f"\n脚本遇到未处理的严重错误: {e}")
        traceback.print_exc()
    finally:
        # 确保在任何情况下都尝试移除钩子
        if _keyboard_hook_active and HAVE_KEYBOARD:
            remove_control_key_hooks() 