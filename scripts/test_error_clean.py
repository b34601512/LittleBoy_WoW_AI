# scripts/test_error_clean.py
# 清晰显示的错误识别测试脚本
import cv2
import sys
import os
import time
import keyboard
import numpy as np
from collections import defaultdict

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

class CleanErrorTester:
    def __init__(self):
        self.grabber = ScreenGrabber()
        self.error_sensor = ErrorMessageSensor()
        self.running = True
        self.max_values = defaultdict(float)  # 记录每种错误的最高匹配值
        self.last_clear_time = time.time()
        
        # 设置F10退出键
        keyboard.add_hotkey('f10', self.stop_test, suppress=True)
        
    def stop_test(self):
        """F10键停止测试"""
        print("\n🛑 F10已按下，停止测试")
        self.running = False
        
    def clear_screen(self):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def run_test(self):
        """运行测试"""
        self.clear_screen()
        print("🚀 错误识别测试程序 - 清晰版本")
        print("🎯 识别阈值: 0.51")
        print("⌨️  按F10结束测试")
        print("="*80)
        
        frame_count = 0
        last_detected = set()
        
        while self.running:
            try:
                # 抓取屏幕
                frame = self.grabber.grab()
                if frame is None:
                    time.sleep(0.1)
                    continue
                    
                frame_count += 1
                
                # 每3秒清屏一次，避免滚动
                current_time = time.time()
                if current_time - self.last_clear_time > 3:
                    self.clear_screen()
                    print("🚀 错误识别测试程序 - 清晰版本")
                    print("🎯 识别阈值: 0.51")
                    print("⌨️  按F10结束测试")
                    print("="*80)
                    self.last_clear_time = current_time
                
                # 检测错误（静默版本）
                error_flags = self._quiet_detect(frame)
                
                # 检查检测结果
                current_detected = set([k for k, v in error_flags.items() if v == 1])
                
                # 显示状态
                print(f"📊 帧数: {frame_count} | 检测到的错误: {len(current_detected)}")
                
                # 显示最高匹配值（只显示前5个最重要的）
                important_errors = ['no_target', 'no_attackable_target', 'range', 'too_far_away', 'face']
                for error in important_errors:
                    max_val = self.max_values.get(error, 0.0)
                    status = "✅ 识别成功" if max_val >= 0.51 else f"❌ {max_val:.4f}"
                    print(f"  {error:20s}: {status}")
                
                # 显示新检测到的错误
                new_detected = current_detected - last_detected
                if new_detected:
                    for error in new_detected:
                        print(f"\n🎉 新检测到: {error}")
                
                lost_detected = last_detected - current_detected
                if lost_detected:
                    for error in lost_detected:
                        print(f"🔚 结束检测: {error}")
                
                last_detected = current_detected
                
                # 控制帧率
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ 测试出错: {e}")
                time.sleep(0.5)
        
        self.show_final_results()
        
    def _quiet_detect(self, frame):
        """静默版本的detect方法，不输出debug信息"""
        error_flags = {identifier: 0 for identifier in self.error_sensor.templates.keys()}
        
        try:
            frame_h, frame_w = frame.shape[:2]
            roi_x_start, roi_y_start = self.error_sensor.x, self.error_sensor.y
            roi_x_end, roi_y_end = min(self.error_sensor.x + self.error_sensor.w, frame_w), min(self.error_sensor.y + self.error_sensor.h, frame_h)

            if roi_x_end <= roi_x_start or roi_y_end <= roi_y_start:
                return error_flags

            image_roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

            if image_roi.size == 0:
                return error_flags

            for identifier, template_img in self.error_sensor.templates.items():
                if template_img.shape[0] > image_roi.shape[0] or template_img.shape[1] > image_roi.shape[1]:
                    continue

                result = cv2.matchTemplate(image_roi, template_img, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                # 更新最高匹配值
                self.max_values[identifier] = max(self.max_values[identifier], max_val)

                if max_val >= self.error_sensor.match_threshold:
                    error_flags[identifier] = 1

        except Exception as e:
            print(f"ERROR in quiet detect: {e}")
        
        return error_flags
        
    def show_final_results(self):
        """显示最终结果"""
        self.clear_screen()
        print("\n" + "="*80)
        print("📊 最终测试结果")
        print("="*80)
        
        print("🎯 各错误类型最高匹配值:")
        for identifier, max_val in sorted(self.max_values.items()):
            status = "✅ 成功识别" if max_val >= 0.51 else "❌ 未识别"
            print(f"  {identifier:25s}: {max_val:.4f} | {status}")
            
        successful = sum(1 for v in self.max_values.values() if v >= 0.51)
        total = len(self.max_values)
        
        print(f"\n📈 总体结果: {successful}/{total} 项成功识别")
        
        if successful > 0:
            print("🎉 有错误成功识别！")
        else:
            print("⚠️  没有错误被识别，可能需要调整模板或阈值")
            
        print("="*80)

if __name__ == "__main__":
    try:
        tester = CleanErrorTester()
        tester.run_test()
    except KeyboardInterrupt:
        print("\n⏹️  用户中断测试")
    finally:
        keyboard.unhook_all()
        print("🔚 测试程序结束") 