# D:\wow_ai\scripts\test_key_interaction.py
import os
import sys
import time
import cv2

# 添加项目根目录到路径
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added project root '{project_root}' to sys.path")
except NameError:
    print("Warning: Unable to determine project root automatically")

# 导入项目模块
from wow_rl.utils.screen_grabber import ScreenGrabber
from wow_rl.utils.interactor_keys import send_key

def main():
    """测试屏幕捕获和键盘交互的主函数"""
    print("========== 交互测试脚本 ==========")
    print("该脚本将测试屏幕捕获和键盘交互功能")
    print("注意: 请确保魔兽世界窗口正在运行并且位于前台")
    
    # 1. 测试屏幕捕获
    print("\n--- 测试屏幕捕获 ---")
    sg = ScreenGrabber()
    
    # 显示捕获的屏幕
    print("正在捕获屏幕...")
    frame = sg.grab()
    if frame is not None and frame.size > 0:
        print(f"成功捕获屏幕，分辨率: {frame.shape}")
        # 显示缩小的图像
        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow("Captured Screen", display_frame)
        cv2.waitKey(1000)  # 显示1秒
        cv2.destroyAllWindows()
    else:
        print("错误: 无法捕获屏幕")
        return
    
    # 2. 测试键盘交互
    print("\n--- 测试键盘交互 ---")
    print("将按顺序发送以下按键: w, f, g, ], tab, left, right")
    print("请切换到魔兽世界窗口以观察按键效果")
    
    input("按回车键开始键盘测试...")
    
    # 按键测试序列
    key_sequence = ['w', 'f', 'g', ']', 'tab', 'left', 'right']
    
    for key in key_sequence:
        print(f"发送按键: {key}")
        send_key(key, wait=0.1, debug=True)
        time.sleep(1)  # 按键之间间隔1秒
    
    print("\n测试完成! 如果在魔兽世界中看到角色对按键有反应，则交互功能正常。")

if __name__ == "__main__":
    main() 