# scripts/test_error_templates.py
# 测试所有错误模板是否能正确识别
import cv2
import sys
import os
import time

# 添加项目根目录到sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from wow_rl.utils.screen_grabber import ScreenGrabber
    from wow_rl.utils.error_sensor import ErrorMessageSensor
    print("成功导入ScreenGrabber和ErrorMessageSensor")
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

def test_error_templates():
    print("=== 错误模板测试程序 ===")
    print("此程序将测试所有错误模板是否能正确加载和识别")
    
    # 初始化ErrorMessageSensor
    try:
        error_sensor = ErrorMessageSensor()
        print(f"\nErrorMessageSensor初始化成功")
        print(f"ROI区域: {error_sensor.DEFAULT_ERROR_ROI}")
        print(f"匹配阈值: {error_sensor.DEFAULT_MATCH_THRESHOLD}")
    except Exception as e:
        print(f"ErrorMessageSensor初始化失败: {e}")
        return

    # 显示所有加载的模板
    print(f"\n加载的错误模板数量: {len(error_sensor.templates)}")
    for identifier, template in error_sensor.templates.items():
        print(f"  ✅ {identifier}: 形状 {template.shape}")
    
    # 检查缺失的模板
    missing_templates = []
    for identifier, path in error_sensor.DEFAULT_TEMPLATES.items():
        if identifier not in error_sensor.templates:
            missing_templates.append((identifier, path))
    
    if missing_templates:
        print(f"\n❌ 缺失的模板 ({len(missing_templates)}个):")
        for identifier, path in missing_templates:
            print(f"  - {identifier}: {path}")
    else:
        print(f"\n✅ 所有模板都已成功加载!")

    # 初始化屏幕抓取
    try:
        grabber = ScreenGrabber()
        print("\n屏幕抓取器初始化成功")
    except Exception as e:
        print(f"屏幕抓取器初始化失败: {e}")
        return

    # 创建显示窗口
    cv2.namedWindow("错误检测测试", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("错误检测测试", 800, 600)

    print("\n=== 实时错误检测测试 ===")
    print("请在游戏中触发各种错误信息，观察识别效果:")
    print("- 按F键攻击过远的目标 (距离太远)")
    print("- 背对目标按F键 (面朝错误方向)")
    print("- 没有目标时按F键 (没有目标)")
    print("- 技能冷却中按F键 (法术没准备好)")
    print("- 死亡状态按F键 (死亡相关错误)")
    print("\n按ESC键退出测试")

    frame_count = 0
    detected_errors_history = []

    try:
        while True:
            # 抓取屏幕
            frame = grabber.grab()
            if frame is None:
                time.sleep(0.1)
                continue

            frame_count += 1

            # 检测错误
            error_flags = error_sensor.detect(frame)
            
            # 统计当前检测到的错误
            current_errors = [identifier for identifier, flag in error_flags.items() if flag == 1]
            
            # 记录检测历史
            if current_errors:
                detected_errors_history.extend(current_errors)
            
            # 创建显示图像
            display_frame = cv2.resize(frame, (800, 600))
            
            # 绘制ROI区域
            roi_x, roi_y, roi_w, roi_h = error_sensor.DEFAULT_ERROR_ROI
            cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 255), 2)
            cv2.putText(display_frame, "Error ROI", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # 显示错误信息
            y_offset = 30
            if current_errors:
                cv2.putText(display_frame, f"检测到错误: {len(current_errors)}个", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 30
                for error in current_errors:
                    cv2.putText(display_frame, f"- {error}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    y_offset += 25
            else:
                cv2.putText(display_frame, "未检测到错误", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 显示统计信息
            unique_errors = list(set(detected_errors_history))
            cv2.putText(display_frame, f"帧数: {frame_count}", (10, display_frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"已识别错误类型: {len(unique_errors)}", (10, display_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"总模板数: {len(error_sensor.templates)}", (10, display_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "按ESC退出", (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 显示图像
            cv2.imshow("错误检测测试", display_frame)

            # 检查退出键
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键
                break

    except KeyboardInterrupt:
        print("\n用户中断测试")

    finally:
        cv2.destroyAllWindows()
        
        # 显示测试结果
        print("\n=== 测试结果总结 ===")
        print(f"总处理帧数: {frame_count}")
        unique_errors = list(set(detected_errors_history))
        print(f"成功识别的错误类型 ({len(unique_errors)}个):")
        for error in unique_errors:
            count = detected_errors_history.count(error)
            print(f"  ✅ {error}: 检测到 {count} 次")
        
        # 显示未识别的错误类型
        all_error_types = list(error_sensor.DEFAULT_TEMPLATES.keys())
        undetected_errors = [error for error in all_error_types if error not in unique_errors]
        if undetected_errors:
            print(f"\n未识别的错误类型 ({len(undetected_errors)}个):")
            for error in undetected_errors:
                print(f"  ❌ {error}: 可能需要在游戏中触发或调整模板")
        
        print(f"\n识别率: {len(unique_errors)}/{len(all_error_types)} = {len(unique_errors)/len(all_error_types)*100:.1f}%")

if __name__ == "__main__":
    test_error_templates() 