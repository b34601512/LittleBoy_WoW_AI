# scripts/test_error_recognition_step_by_step.py
# 逐步测试错误识别系统 - 一次测试一个错误类型
import cv2
import sys
import os
import time
import keyboard

# 添加项目根目录到sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from wow_rl.utils.screen_grabber import ScreenGrabber
    from wow_rl.utils.error_sensor import ErrorMessageSensor
    print("✅ 成功导入ScreenGrabber和ErrorMessageSensor")
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)

# 测试清单 - 按优先级排序
TEST_CHECKLIST = [
    {
        "id": 1,
        "error_type": "no_target",
        "chinese_name": "你没有目标",
        "test_method": "没有选中任何目标时按F键",
        "expected": "应该出现'你没有目标'或类似提示"
    },
    {
        "id": 2, 
        "error_type": "range",
        "chinese_name": "距离太远",
        "test_method": "选中很远的目标，按F键攻击",
        "expected": "应该出现'距离太远'提示"
    },
    {
        "id": 3,
        "error_type": "too_far_away", 
        "chinese_name": "你距离太远",
        "test_method": "选中很远的目标，按F键攻击(不同措辞)",
        "expected": "应该出现'你距离太远'提示"
    },
    {
        "id": 4,
        "error_type": "face",
        "chinese_name": "必须面对目标", 
        "test_method": "选中目标，背对目标按F键",
        "expected": "应该出现'必须面对目标'提示"
    },
    {
        "id": 5,
        "error_type": "facing_wrong_way",
        "chinese_name": "你面朝错误的方向",
        "test_method": "选中目标，背对目标按F键(不同措辞)",
        "expected": "应该出现'你面朝错误的方向'提示"
    },
    {
        "id": 6,
        "error_type": "spell_not_ready",
        "chinese_name": "法术还没有准备好",
        "test_method": "技能冷却中时按F键",
        "expected": "应该出现'法术还没有准备好'提示"
    },
    {
        "id": 7,
        "error_type": "cant_attack_target",
        "chinese_name": "你不能攻击那个目标",
        "test_method": "选中友方NPC或玩家，按F键",
        "expected": "应该出现'你不能攻击那个目标'提示"
    },
    {
        "id": 8,
        "error_type": "no_attackable_target",
        "chinese_name": "现在没有可以攻击的目标",
        "test_method": "没有可攻击目标时按F键",
        "expected": "应该出现'现在没有可以攻击的目标'提示"
    },
    {
        "id": 9,
        "error_type": "player_dead",
        "chinese_name": "你死了",
        "test_method": "角色死亡状态时按F键",
        "expected": "应该出现'你死了'提示"
    },
    {
        "id": 10,
        "error_type": "cannot_attack_while_dead",
        "chinese_name": "你不能在死亡状态时攻击",
        "test_method": "角色死亡状态时按F键攻击",
        "expected": "应该出现死亡状态攻击限制提示"
    }
]

class StepByStepTester:
    def __init__(self):
        self.grabber = ScreenGrabber()
        self.error_sensor = ErrorMessageSensor()
        self.current_test_index = 0
        self.results = {}
        self.running = True
        self.test_success = False
        
        # 设置F10退出键
        keyboard.add_hotkey('f10', self.stop_test, suppress=True)
        
    def stop_test(self):
        """F10键停止测试"""
        print("\n🛑 F10已按下，停止测试")
        self.running = False
        
    def display_current_test(self):
        """显示当前测试项目"""
        if self.current_test_index >= len(TEST_CHECKLIST):
            print("\n🎉 所有测试项目已完成！")
            return False
            
        test = TEST_CHECKLIST[self.current_test_index]
        print(f"\n" + "="*80)
        print(f"📋 测试项目 {test['id']}/10: {test['chinese_name']}")
        print(f"🔧 错误类型: {test['error_type']}")
        print(f"🎮 测试方法: {test['test_method']}")
        print(f"🎯 预期结果: {test['expected']}")
        print("="*80)
        print("🕹️  请切换到游戏中按照上述方法操作")
        print("👀 观察下方匹配值，当匹配值 ≥ 0.51 时表示识别成功")
        print("⌨️  按F10结束当前测试并查看结果")
        print("-"*80)
        return True
        
    def run_current_test(self):
        """运行当前测试"""
        test = TEST_CHECKLIST[self.current_test_index]
        error_type = test['error_type']
        
        print(f"🔍 开始监控错误类型: {error_type}")
        print("📊 实时匹配值:")
        
        max_match_value = 0.0
        detection_count = 0
        frame_count = 0
        
        while self.running and not self.test_success:
            try:
                # 抓取屏幕
                frame = self.grabber.grab()
                if frame is None:
                    time.sleep(0.1)
                    continue
                    
                frame_count += 1
                
                # 检测错误
                error_flags = self.error_sensor.detect(frame)
                
                # 获取当前测试错误类型的匹配值
                # 从debug输出中解析匹配值（这里需要修改error_sensor来返回匹配值）
                current_match = self.get_match_value_for_type(error_type)
                max_match_value = max(max_match_value, current_match)
                
                # 检查是否识别成功
                if error_flags.get(error_type, 0) == 1:
                    detection_count += 1
                    print(f"\n🎉 识别成功! 匹配值: {current_match:.4f} ≥ 0.51")
                    print(f"📈 最高匹配值: {max_match_value:.4f}")
                    print(f"🔢 识别次数: {detection_count}")
                    self.test_success = True
                    self.results[error_type] = {
                        'success': True,
                        'max_match_value': max_match_value,
                        'detection_count': detection_count,
                        'frames_tested': frame_count
                    }
                    break
                else:
                    # 显示当前匹配值（每10帧显示一次避免刷屏）
                    if frame_count % 10 == 0:
                        print(f"📊 {error_type}: {current_match:.4f} (最高: {max_match_value:.4f}) 帧数: {frame_count}")
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ 测试出错: {e}")
                time.sleep(0.5)
        
        # 如果测试结束但未成功识别
        if not self.test_success and not self.running:
            print(f"\n⚠️  测试被手动停止")
            print(f"📈 最高匹配值: {max_match_value:.4f}")
            print(f"🔢 帧数: {frame_count}")
            self.results[error_type] = {
                'success': False,
                'max_match_value': max_match_value,
                'detection_count': detection_count,
                'frames_tested': frame_count
            }
            
    def get_match_value_for_type(self, error_type):
        """从最新的检测中获取特定错误类型的匹配值"""
        # 这是一个简化版本，实际应该修改error_sensor返回详细匹配值
        # 目前返回0作为占位符，实际需要修改error_sensor.detect方法
        return 0.0
        
    def show_test_result(self):
        """显示当前测试结果"""
        test = TEST_CHECKLIST[self.current_test_index]
        error_type = test['error_type']
        result = self.results.get(error_type, {})
        
        print(f"\n📊 测试结果 - {test['chinese_name']}")
        print(f"✅ 识别成功: {'是' if result.get('success', False) else '否'}")
        print(f"📈 最高匹配值: {result.get('max_match_value', 0):.4f}")
        print(f"🔢 识别次数: {result.get('detection_count', 0)}")
        print(f"🎬 测试帧数: {result.get('frames_tested', 0)}")
        
        if result.get('success', False):
            print("🎉 该项测试通过！")
        else:
            print("❌ 该项测试未通过，可能需要:")
            print("   1. 调整模板图片")
            print("   2. 调整识别阈值") 
            print("   3. 检查游戏内操作是否正确触发了错误")
            
    def next_test(self):
        """进入下一个测试"""
        self.current_test_index += 1
        self.test_success = False
        self.running = True
        
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 错误识别系统逐步测试程序")
        print("📝 将测试以下10种错误类型的识别能力")
        print("⌨️  每个测试中按F10可结束当前测试进入下一个")
        
        while self.current_test_index < len(TEST_CHECKLIST):
            if not self.display_current_test():
                break
                
            self.run_current_test()
            self.show_test_result()
            
            if self.current_test_index < len(TEST_CHECKLIST) - 1:
                input("\n⏭️  按回车键继续下一个测试...")
                self.next_test()
            else:
                break
                
        self.show_final_summary()
        
    def show_final_summary(self):
        """显示最终测试总结"""
        print("\n" + "="*80)
        print("📊 最终测试总结")
        print("="*80)
        
        passed_tests = 0
        failed_tests = 0
        
        for i, test in enumerate(TEST_CHECKLIST):
            error_type = test['error_type']
            result = self.results.get(error_type, {})
            success = result.get('success', False)
            max_match = result.get('max_match_value', 0)
            
            status = "✅ 通过" if success else "❌ 失败"
            print(f"{i+1:2d}. {test['chinese_name']:20s} {status} (最高匹配值: {max_match:.4f})")
            
            if success:
                passed_tests += 1
            else:
                failed_tests += 1
                
        print("-"*80)
        print(f"📈 总体结果: {passed_tests}/{len(TEST_CHECKLIST)} 项测试通过")
        print(f"✅ 通过: {passed_tests} 项")
        print(f"❌ 失败: {failed_tests} 项")
        
        if passed_tests == len(TEST_CHECKLIST):
            print("🎉 恭喜！所有错误识别测试都通过了，可以开始PPO训练！")
        else:
            print("⚠️  部分测试未通过，建议先修复识别问题再进行PPO训练")
            
        print("="*80)

if __name__ == "__main__":
    try:
        tester = StepByStepTester()
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n⏹️  用户中断测试")
    finally:
        keyboard.unhook_all()
        cv2.destroyAllWindows()
        print("🔚 测试程序结束") 