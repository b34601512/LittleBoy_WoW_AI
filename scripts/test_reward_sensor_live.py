# D:\wow_ai\scripts\test_reward_sensor_live.py
import cv2
import time
import sys
import os
import traceback

# --- 动态路径设置 ---
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added project root '{project_root}' to sys.path")
except NameError:
    project_root = None
# --- 路径设置结束 ---

try:
    from wow_rl.utils.screen_grabber import ScreenGrabber
    from wow_rl.utils.error_sensor import ErrorMessageSensor # 确保导入的是我们新的模板匹配版本
    print("Imported ScreenGrabber and ErrorMessageSensor successfully.")
except ImportError as e:
    print(f"ImportError: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    traceback.print_exc()
    sys.exit(1)

def run_test():
    print("Initializing ScreenGrabber and ErrorMessageSensor (Template Matching Version)...")
    try:
        grabber = ScreenGrabber() 
        error_sensor = ErrorMessageSensor() # 使用默认参数
        print("ScreenGrabber and ErrorMessageSensor initialized.")
    except Exception as e:
        print(f"Error during sensor initialization: {e}")
        traceback.print_exc()
        return

    cv2.namedWindow("Live Feed (Error Sensor - Template)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Feed (Error Sensor - Template)", 480, 270)

    print("\nStarting live error detection test loop (Template Matching)...")
    print("Manually trigger in-game errors like '必须面对目标' or '距离太远'.")
    print(f"Sensor will search in ROI: {error_sensor.DEFAULT_ERROR_ROI}")
    print(f"Templates being used: {list(error_sensor.DEFAULT_TEMPLATES.keys())}")
    print(f"Match threshold: {error_sensor.DEFAULT_MATCH_THRESHOLD}")
    print("Press 'ESC' in the 'Live Feed (Error Sensor - Template)' window to quit.")

    try:
        while True:
            full_frame = grabber.grab()
            if full_frame is None or full_frame.size == 0:
                time.sleep(0.5)
                continue

            error_flags = error_sensor.detect(full_frame)
            display_frame = cv2.resize(full_frame, (480, 270))

            text_lines = []
            if error_flags.get('face', 0) == 1:
                text_lines.append("Error: Need to FACE target! (Template)")
            if error_flags.get('range', 0) == 1:
                text_lines.append("Error: Target too FAR! (Template)")
            if error_flags.get('no_target', 0) == 1:
                text_lines.append("Error: NO TARGET! (Template)")
            
            if not text_lines and any(error_flags.values()): # Should not happen if keys are consistent
                 text_lines.append(f"Other template errors: {error_flags}")
            elif not text_lines:
                 text_lines.append("No critical errors detected by Template.")

            for i, line in enumerate(text_lines):
                cv2.putText(display_frame, line, (10, 20 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if "Error" in line else (0,255,0), 1)

            cv2.imshow("Live Feed (Error Sensor - Template)", display_frame)

            # !!!!! 修改点：退出键改为 ESC !!!!!
            key = cv2.waitKey(100) 
            if key == 27: # 27 is the ASCII for ESC key
                print("ESC pressed, quitting test.")
                break

    except KeyboardInterrupt:
        print("Test interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"An error occurred in the test loop: {e}")
        traceback.print_exc()
    finally:
        print("Closing OpenCV windows...")
        cv2.destroyAllWindows()
        for _ in range(5): cv2.waitKey(1)
        print("Cleanup finished.")

if __name__ == "__main__":
    print("--------------------------------------------------------------------------")
    print("Starting Error Message Sensor Live Test (TEMPLATE MATCHING VERSION).")
    print("Please ensure WoW is running and visible on screen.")
    print("1. MAKE SURE YOU HAVE CREATED TEMPLATE IMAGES and placed them in D:\\wow_ai\\data\\")
    print("   - template_error_face.png")
    print("   - template_error_range.png")
    print("   - template_error_no_target.png")
    print("2. The script will attempt to match these templates in the defined ROI.")
    print("--------------------------------------------------------------------------")
    time.sleep(3)
    run_test()