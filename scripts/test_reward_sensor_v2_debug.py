# D:\wow_ai\scripts\test_reward_sensor_v2_debug.py
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
    from wow_rl.utils.reward_sensor import RewardSensor # 导入我们修改后的 RewardSensor
    print("Imported ScreenGrabber and RewardSensor (v_dead_detect) successfully.")
except ImportError as e:
    print(f"ImportError: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- RewardSensor 参数 (与您 WowKeyEnv 中最终确定的参数一致) ---
# ROI for target selection (and currently for dead status as well)
ROI_REWARD = (250, 35, 205, 64) # !!! 请确保这是您0.99分匹配的ROI !!!
TEMPLATE_PATH_SELECTED = r'D:\wow_ai\data\target_frame_template.png'
MATCH_THRESHOLD_SELECTED = 0.90 # 您为选中状态设置的高阈值

# Parameters for "dead" template
TEMPLATE_PATH_DEAD = r'D:\wow_ai\data\template_target_dead.png'
MATCH_THRESHOLD_DEAD = 0.7  # 初始为0.7，我们可以根据打印的匹配值调整这个

# YOLO parameters (RewardSensor会加载，但我们在这个脚本中主要关注模板匹配)
YOLO_WEIGHT_PATH = r'D:\wow_ai\runs\detect\gwyx_detect_run_s_ep200_neg\weights\best.pt'

def run_sensor_v2_debug():
    print("Initializing ScreenGrabber and RewardSensor (v_dead_detect) for debug...")
    try:
        grabber = ScreenGrabber()
        # 初始化 RewardSensor，它会加载两个模板
        reward_sensor = RewardSensor(
            roi_select=ROI_REWARD,
            template_path_select=TEMPLATE_PATH_SELECTED,
            match_thresh_select=MATCH_THRESHOLD_SELECTED,
            template_path_dead=TEMPLATE_PATH_DEAD,
            match_thresh_dead=MATCH_THRESHOLD_DEAD, # 使用我们定义的死亡阈值
            yolo_weight=YOLO_WEIGHT_PATH # 传递YOLO路径以避免其内部加载错误
        )
        print("ScreenGrabber and RewardSensor initialized.")
    except Exception as e:
        print(f"Error during sensor initialization: {e}")
        traceback.print_exc()
        return

    # 显示两个模板以便参考
    if reward_sensor.template_selected is not None:
        cv2.imshow("Template - Selected", reward_sensor.template_selected)
    if reward_sensor.template_dead is not None:
        cv2.imshow("Template - Dead", reward_sensor.template_dead)

    window_name_feed = "Reward Sensor V2 - ROI Feed"
    cv2.namedWindow(window_name_feed, cv2.WINDOW_NORMAL)
    
    print("\n--- Starting RewardSensor V2 Debug Loop ---")
    print(f"Watching ROI: {ROI_REWARD}")
    print("INSTRUCTIONS:")
    print("1. Make sure WoW is running and visible (MAXIMIZED WINDOW mode).")
    print("2. Manually select an enemy target.")
    print("3. Observe 'ROI Feed' window and console output for 'sel_flag'.")
    print("4. Kill the selected enemy target.")
    print("5. Observe 'ROI Feed' window and console output for 'is_dead_flag'.")
    print("   (Console will show '[RS Debug] Selected/Dead Template MatchVal')")
    print("Press 'ESC' in the 'ROI Feed' window to quit.")
    print("------------------------------------------------------------------")

    try:
        while True:
            full_frame = grabber.grab()
            if full_frame is None or full_frame.size == 0:
                time.sleep(0.1)
                continue

            # 调用修改后的 RewardSensor.analyze()
            # 它现在应该在内部打印 "Selected Template MatchVal" 和 "Dead Template MatchVal"
            sel_flag, is_dead_flag, yolo_logits = reward_sensor.analyze(full_frame)

            # --- 可视化 ROI 和检测结果 ---
            roi_x, roi_y, roi_w, roi_h = ROI_REWARD
            frame_h_full, frame_w_full = full_frame.shape[:2]
            # 安全边界检查 (与之前脚本类似)
            if not (0 <= roi_x < frame_w_full and 0 <= roi_y < frame_h_full and roi_x + roi_w <= frame_w_full and roi_y + roi_h <= frame_h_full):
                 print(f"ERROR: ROI {ROI_REWARD} is out of bounds for frame {full_frame.shape[:2]}.")
                 cv2.imshow(window_name_feed, np.zeros((100,300,3), dtype=np.uint8))
                 if cv2.waitKey(100) == 27: break
                 continue

            roi_crop_display = full_frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w].copy()
            
            text_sel = f"Sel: {sel_flag}"
            color_sel = (0, 255, 0) if sel_flag == 1 else (0, 0, 255)
            cv2.putText(roi_crop_display, text_sel, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_sel, 2)

            text_dead = f"Dead: {is_dead_flag}"
            color_dead = (0, 255, 0) if is_dead_flag == 1 else (0, 0, 255)
            cv2.putText(roi_crop_display, text_dead, (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_dead, 2)
            
            # 显示YOLO的简单指示 (如果sel_flag=1且未死)
            if sel_flag == 1 and is_dead_flag == 0:
                yolo_text = f"P(Gwyx):{yolo_logits[0]:.2f}"
                cv2.putText(roi_crop_display, yolo_text, (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,0),1)

            cv2.imshow(window_name_feed, roi_crop_display)
            # --- 可视化结束 ---

            key = cv2.waitKey(100) # ~10 FPS
            if key == 27: # ESC
                print("ESC key pressed. Exiting loop.")
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
        print("RewardSensor V2 debug finished.")

if __name__ == "__main__":
    print("--------------------------------------------------------------------------")
    print("Starting RewardSensor V2 (Selected & Dead) Debug Script.")
    print("This script will test template matching for target selection AND dead status.")
    print(f"Make sure your RewardSensor.py has the 'is_dead_flag' logic.")
    print(f"And that it prints '[RS Debug] Selected/Dead Template MatchVal' inside analyze().")
    print("--------------------------------------------------------------------------")
    # 等待用户确认或准备
    # input("Press Enter to start once WoW is in MAXIMIZED WINDOW mode and ready...")
    time.sleep(2)
    run_sensor_v2_debug()