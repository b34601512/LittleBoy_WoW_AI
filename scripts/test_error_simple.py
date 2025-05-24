# scripts/test_error_simple.py
# 简化的错误识别测试脚本
import cv2
import sys
import os
import time
import keyboard
import numpy as np

# 添加项目根目录到sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from wow_rl.utils.screen_grabber import ScreenGrabber
    from wow_rl.utils.error_sensor import ErrorMessageSensor
    print("✅ 成功导入ScreenGrabber和ErrorMessageSensor")
    print(f"🎯 识别阈值: 0.51")
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)

# 测试清单
TEST_CHECKLIST = [
    "1. 你没有目标 (no_target) - 没有选中任何目标时按F键",
    "2. 距离太远 (range) - 选中很远的目标按F键", 
    "3. 你距离太远 (too_far_away) - 选中很远目标按F键(不同措辞)",
    "4. 必须面对目标 (face) - 选中目标,背对目标按F键",
    "5. 你面朝错误的方向 (facing_wrong_way) - 背对目标按F键(不同措辞)",
    "6. 法术还没有准备好 (spell_not_ready) - 技能冷却中按F键",
    "7. 你不能攻击那个目标 (cant_attack_target) - 选中友方NPC按F键",
    "8. 现在没有可以攻击的目标 (no_attackable_target) - 没有可攻击目标按F键",
    "9. 你死了 (player_dead) - 死亡状态按F键",
    "10. 你不能在死亡状态时攻击 (cannot_attack_while_dead) - 死亡状态按F键"
]

class SimpleErrorTester:
    def __init__(self):
        self.grabber = ScreenGrabber()
        self.error_sensor = ErrorMessageSensor()
        self.running = True
        
        # 设置F10退出键
        keyboard.add_hotkey('f10', self.stop_test, suppress=True)
        
        # 创建显示窗口
        cv2.namedWindow("错误识别监控", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("错误识别监控", 1000, 800)
        
    def stop_test(self):
        """F10键停止测试"""
        print("\n🛑 F10已按下，停止测试")
        self.running = False
        
    def run_test(self):
        """运行测试"""
        print("🚀 错误识别测试程序")
        print("📋 测试清单:")
        for test in TEST_CHECKLIST:
            print(f"   {test}")
        print("\n" + "="*80)
        print("🎮 请在游戏中按照清单依次测试各种错误")
        print("👀 观察控制台输出的匹配值")
        print("✅ 当匹配值 ≥ 0.51 时，错误会被识别")
        print("⌨️  按F10结束测试")
        print("="*80)
        
        frame_count = 0
        last_detection = {}  # 记录每种错误上次检测的状态
        
        while self.running:
            try:
                # 抓取屏幕
                frame = self.grabber.grab()
                if frame is None:
                    time.sleep(0.1)
                    continue
                    
                frame_count += 1
                
                # 检测错误（会在控制台输出匹配值）
                error_flags = self.error_sensor.detect(frame)
                
                # 检查是否有新的错误被识别
                current_errors = [identifier for identifier, flag in error_flags.items() if flag == 1]
                
                # 显示当前检测状态
                if current_errors:
                    for error in current_errors:
                        if error not in last_detection or not last_detection[error]:
                            print(f"\n🎉 检测到错误: {error}")
                            last_detection[error] = True
                else:
                    # 重置检测状态
                    for identifier in error_flags.keys():
                        last_detection[identifier] = False
                
                # 创建状态显示图像
                self.create_status_display(frame, error_flags, frame_count)
                
                # 控制帧率
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ 测试出错: {e}")
                time.sleep(0.5)
        
        self.show_final_results()
        
    def create_status_display(self, frame, error_flags, frame_count):
        """创建状态显示图像"""
        try:
            # 调整图像大小
            display_frame = cv2.resize(frame, (800, 600))
            
            # 创建信息面板
            info_panel = np.zeros((200, 800, 3), dtype=np.uint8)
            
            # 显示标题
            cv2.putText(info_panel, f"Error Recognition Test - Frame: {frame_count}", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示当前检测状态
            y_offset = 50
            detected_count = sum(error_flags.values())
            cv2.putText(info_panel, f"Detected Errors: {detected_count}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示具体检测到的错误
            y_offset = 80
            col1_x, col2_x = 10, 400
            row_height = 25
            
            error_list = list(error_flags.items())
            for i, (identifier, flag) in enumerate(error_list):
                x_pos = col1_x if i < 5 else col2_x
                y_pos = y_offset + (i % 5) * row_height
                
                color = (0, 255, 0) if flag == 1 else (100, 100, 100)
                status = "✓" if flag == 1 else "✗"
                text = f"{status} {identifier}"
                
                cv2.putText(info_panel, text, (x_pos, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 组合显示
            combined = np.vstack([display_frame, info_panel])
            cv2.imshow("错误识别监控", combined)
            cv2.waitKey(1)
            
        except Exception as e:
            print(f"显示错误: {e}")
            
    def show_final_results(self):
        """显示最终结果"""
        print("\n" + "="*80)
        print("📊 测试结果总结")
        print("="*80)
        
        print("✅ 加载的错误模板:")
        for identifier in self.error_sensor.templates.keys():
            print(f"   {identifier}")
            
        print(f"\n📈 模板总数: {len(self.error_sensor.templates)}")
        print("💡 注意: 请根据控制台输出的匹配值判断识别效果")
        print("   匹配值 ≥ 0.51 表示识别成功")
        print("="*80)

if __name__ == "__main__":
    try:
        tester = SimpleErrorTester()
        tester.run_test()
    except KeyboardInterrupt:
        print("\n⏹️  用户中断测试")
    finally:
        keyboard.unhook_all()
        cv2.destroyAllWindows()
        print("🔚 测试程序结束") 