# D:\wow_ai\scripts\test_reward_sensor_debug.py
import cv2
import numpy as np
import time
import sys
import os
import traceback

# --- 动态路径设置 ---
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir) # wow_ai
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added project root '{project_root}' to sys.path")
except NameError:
    project_root = None
# --- 路径设置结束 ---

try:
    from wow_rl.utils.screen_grabber import ScreenGrabber
except ImportError as e:
    print(f"ImportError: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- RewardSensor 参数 ---
# 使用您提供的坐标计算出的新ROI
REWARD_ROI =(319, 35, 245, 64)  # (x_top_left, y_top_left, width, height)
TEMPLATE_PATH = r'D:\wow_ai\data\target_frame_template.png' # 确保这是您之前能达到高匹配度的模板
MATCH_THRESHOLD = 0.80 # 先用0.8作为阈值，如果匹配度高，可以再调回0.85

def run_reward_sensor_debug():
    print("Initializing ScreenGrabber for RewardSensor debug...")
    try:
        grabber = ScreenGrabber() # 默认全屏
        print("ScreenGrabber initialized.")
    except Exception as e:
        print(f"Error initializing ScreenGrabber: {e}")
        traceback.print_exc()
        return

    print(f"Loading target template from: {TEMPLATE_PATH}")
    target_template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_COLOR)
    if target_template is None:
        print(f"CRITICAL ERROR: Failed to load template image from {TEMPLATE_PATH}")
        return
    
    t_h, t_w = target_template.shape[:2]
    print(f"Target template loaded. Shape: (H={t_h}, W={t_w})")
    cv2.imshow("Target Template", target_template)

    # 使用新的ROI尺寸来尝试调整窗口大小，但OpenCV的resizeWindow可能不总按预期工作
    # 主要还是看实际截取的ROI内容
    cv2.namedWindow("Reward Sensor ROI Feed", cv2.WINDOW_NORMAL) 
    # cv2.resizeWindow("Reward Sensor ROI Feed", REWARD_ROI[2], REWARD_ROI[3]) # 尝试设置窗口大小

    print("\n--- Starting RewardSensor Debug Loop ---")
    print(f"Watching NEW ROI: x={REWARD_ROI[0]}, y={REWARD_ROI[1]}, w={REWARD_ROI[2]}, h={REWARD_ROI[3]}")
    print("INSTRUCTIONS:")
    print("1. Make sure WoW is running and visible.")
    print("2. Manually select an enemy target in the game so the target frame appears at the new ROI location.")
    print("3. Observe the 'Reward Sensor ROI Feed' window and console output.")
    print("4. Check if the template matches and if the match value is high.")
    print("Press 'ESC' in the 'Reward Sensor ROI Feed' window to quit.")
    print("------------------------------------------------------------------")

    try:
        while True:
            full_frame = grabber.grab()
            if full_frame is None or full_frame.size == 0:
                time.sleep(0.1)
                continue

            roi_x, roi_y, roi_w, roi_h = REWARD_ROI
            frame_h_full, frame_w_full = full_frame.shape[:2]
            
            # 安全边界检查
            if roi_x < 0 or roi_y < 0 or roi_x + roi_w > frame_w_full or roi_y + roi_h > frame_h_full:
                print(f"ERROR: ROI {REWARD_ROI} is out of bounds for frame size {full_frame.shape[:2]}. Please check coordinates.")
                cv2.imshow("Reward Sensor ROI Feed", np.zeros((100,100,3), dtype=np.uint8))
                if cv2.waitKey(100) == 27: break
                continue

            roi_crop = full_frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

            if roi_crop.size == 0:
                print("Warning: ROI crop is empty.")
                cv2.imshow("Reward Sensor ROI Feed", np.zeros((100,100,3), dtype=np.uint8))
                if cv2.waitKey(100) == 27: break
                continue
            
            display_roi = roi_crop.copy() # 创建副本用于显示

            if t_h > roi_crop.shape[0] or t_w > roi_crop.shape[1]:
                text = f"Template({t_w}x{t_h}) > ROI({roi_crop.shape[1]}x{roi_crop.shape[0]})"
                cv2.putText(display_roi, text, (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255),1)
                cv2.imshow("Reward Sensor ROI Feed", display_roi)
                if cv2.waitKey(100) == 27: break
                continue

            result = cv2.matchTemplate(roi_crop, target_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            match_color = (0, 0, 255) 
            sel_detected = 0

            if max_val >= MATCH_THRESHOLD:
                match_color = (0, 255, 0) 
                sel_detected = 1
                top_left = max_loc
                bottom_right = (top_left[0] + t_w, top_left[1] + t_h)
                cv2.rectangle(display_roi, top_left, bottom_right, match_color, 2)
            
            text = f"MatchVal: {max_val:.2f} (Thresh:{MATCH_THRESHOLD}) Sel:{sel_detected}"
            cv2.putText(display_roi, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, match_color, 1) # 字体稍大一点
            cv2.imshow("Reward Sensor ROI Feed", display_roi)

            if sel_detected:
                print(f"SUCCESS: Target selected! Match Value: {max_val:.4f}")
            
            key = cv2.waitKey(100) 
            if key == 27: 
                print("ESC pressed, quitting debug loop.")
                break
    
    except KeyboardInterrupt:
        print("\nDebug loop interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\nError during debug loop:")
        traceback.print_exc()
    finally:
        print("Closing OpenCV windows...")
        cv2.destroyAllWindows()
        for i in range(5): cv2.waitKey(1)
        print("Cleanup finished.")

if __name__ == "__main__":
    print("--------------------------------------------------------------------------")
    print("Starting RewardSensor Debug Script with NEW ROI.")
    print("This script will test template matching for the target selection frame.")
    print("--------------------------------------------------------------------------")
    run_reward_sensor_debug()