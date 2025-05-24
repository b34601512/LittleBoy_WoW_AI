# scripts/test_error_static.py
# 静态显示的错误识别测试脚本 - 不滚动
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
    time.sleep(1)
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)

class StaticErrorTester:
    def __init__(self):
        self.grabber = ScreenGrabber()
        self.error_sensor = ErrorMessageSensor()
        self.running = True
        self.max_values = defaultdict(float)
        self.detected_errors = set()
        
        # 设置F10退出键
        keyboard.add_hotkey('f10', self.stop_test, suppress=True)
        
    def stop_test(self):
        """F10键停止测试"""
        self.running = False
        
    def clear_screen(self):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def display_status(self, frame_count):
        """显示当前状态 - 固定格式不滚动"""
        self.clear_screen()
        
        print("🚀 错误识别测试程序 - 静态版本")
        print("🎯 识别阈值: 0.51 | ⌨️ 按F10结束测试")
        print("="*80)
        print(f"📊 已处理帧数: {frame_count}")
        print("-"*80)
        
        # 显示测试指引
        print("📋 请在游戏中测试以下操作:")
        print("  1. 没有选中目标时按F键 → 应触发'现在没有可以攻击的目标'")
        print("  2. 选中很远目标按F键 → 应触发'距离太远'类错误")
        print("  3. 背对目标按F键 → 应触发'面对目标'类错误")
        print("-"*80)
        
        # 显示各错误类型状态
        print("🎯 错误识别状态:")
        important_errors = [
            ('no_attackable_target', '现在没有可以攻击的目标'),
            ('no_target', '你没有目标'),
            ('range', '距离太远'),
            ('too_far_away', '你距离太远'),
            ('face', '必须面对目标'),
            ('facing_wrong_way', '你面朝错误的方向'),
            ('cant_attack_target', '你不能攻击那个目标'),
            ('spell_not_ready', '法术还没有准备好'),
            ('player_dead', '你死了'),
            ('cannot_attack_while_dead', '你不能在死亡状态时攻击')
        ]
        
        for error_id, chinese_name in important_errors:
            max_val = self.max_values.get(error_id, 0.0)
            if max_val >= 0.51:
                status = "✅ 已识别"
                color_indicator = "🟢"
            elif max_val > 0.3:
                status = f"🟡 {max_val:.3f}"
                color_indicator = "🟡"
            else:
                status = f"❌ {max_val:.3f}"
                color_indicator = "🔴"
                
            print(f"  {color_indicator} {chinese_name:25s}: {status}")
        
        print("-"*80)
        
        # 显示已检测到的错误
        if self.detected_errors:
            print("🎉 已成功检测到的错误:")
            for error in sorted(self.detected_errors):
                chinese_name = dict(important_errors).get(error, error)
                print(f"  ✅ {chinese_name}")
        else:
            print("⏳ 等待检测错误... (请在游戏中按F键触发各种错误)")
            
        print("="*80)
        
    def run_test(self):
        """运行测试"""
        frame_count = 0
        last_display_time = 0
        
        # 初始显示
        self.display_status(frame_count)
        
        while self.running:
            try:
                # 抓取屏幕
                frame = self.grabber.grab()
                if frame is None:
                    time.sleep(0.1)
                    continue
                    
                frame_count += 1
                
                # 检测错误（静默版本）
                error_flags = self._quiet_detect(frame)
                
                # 检查是否有新的错误被检测到
                current_detected = set([k for k, v in error_flags.items() if v == 1])
                new_detected = current_detected - self.detected_errors
                
                # 如果有新检测到的错误，或者每5秒更新一次显示
                current_time = time.time()
                should_update = (
                    new_detected or 
                    current_time - last_display_time > 5 or
                    any(self.max_values[k] > 0.3 for k in self.max_values if k not in self.detected_errors)
                )
                
                if should_update:
                    self.detected_errors.update(new_detected)
                    self.display_status(frame_count)
                    last_display_time = current_time
                    
                    # 如果有新检测，稍微暂停一下让用户看清
                    if new_detected:
                        time.sleep(1)
                
                # 控制帧率
                time.sleep(0.2)
                
            except Exception as e:
                print(f"❌ 测试出错: {e}")
                time.sleep(1)
        
        self.show_final_results()
        
    def _quiet_detect(self, frame):
        """静默版本的detect方法"""
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
            pass  # 静默处理错误
        
        return error_flags
        
    def show_final_results(self):
        """显示最终结果"""
        self.clear_screen()
        print("\n" + "="*80)
        print("📊 最终测试结果")
        print("="*80)
        
        successful = 0
        total = 0
        
        important_errors = [
            ('no_attackable_target', '现在没有可以攻击的目标'),
            ('no_target', '你没有目标'),
            ('range', '距离太远'),
            ('too_far_away', '你距离太远'),
            ('face', '必须面对目标'),
            ('facing_wrong_way', '你面朝错误的方向'),
            ('cant_attack_target', '你不能攻击那个目标'),
            ('spell_not_ready', '法术还没有准备好'),
            ('player_dead', '你死了'),
            ('cannot_attack_while_dead', '你不能在死亡状态时攻击')
        ]
        
        for error_id, chinese_name in important_errors:
            total += 1
            max_val = self.max_values.get(error_id, 0.0)
            if max_val >= 0.51:
                successful += 1
                status = "✅ 成功识别"
            else:
                status = "❌ 未识别"
            print(f"  {chinese_name:25s}: {max_val:.4f} | {status}")
            
        print("-"*80)
        print(f"📈 识别结果: {successful}/{total} 项成功")
        
        if successful >= 5:
            print("🎉 大部分错误已能识别，可以开始PPO训练！")
        elif successful > 0:
            print("🟡 部分错误能识别，建议优化模板后再训练")
        else:
            print("❌ 没有错误被识别，需要检查模板和阈值设置")
            
        print("="*80)
        input("按回车键退出...")

if __name__ == "__main__":
    try:
        tester = StaticErrorTester()
        tester.run_test()
    except KeyboardInterrupt:
        print("\n⏹️  用户中断测试")
    finally:
        keyboard.unhook_all()
        print("🔚 测试程序结束") 