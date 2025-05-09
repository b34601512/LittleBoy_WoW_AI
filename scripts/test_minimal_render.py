# D:\wow_ai\scripts\test_minimal_render.py
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
    # 只导入最基础的 ScreenGrabber
    from wow_rl.utils.screen_grabber import ScreenGrabber
    print("Imported ScreenGrabber successfully.")
except ImportError as e:
    print(f"ImportError: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    traceback.print_exc()
    sys.exit(1)

def run_minimal_render_test():
    print("Initializing ScreenGrabber...")
    try:
        grabber = ScreenGrabber() # 使用默认全屏截图
        print("ScreenGrabber initialized.")
    except Exception as e:
        print(f"Error initializing ScreenGrabber: {e}")
        traceback.print_exc()
        return

    window_name = "Minimal Render Test Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 360) # 显示一个适中大小的窗口

    print("\n--- Starting Minimal Render Loop ---")
    print("This will continuously grab the screen and display it.")
    print("Press 'ESC' key IN THIS WINDOW to quit.")
    print("-----------------------------------------")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # 1. 截图
            frame = grabber.grab()

            # 2. 检查截图是否成功
            if frame is None or frame.size == 0:
                print("Warning: Failed to grab frame or frame is empty.")
                # 显示一个黑色图像提示错误
                error_frame = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, "Frame Grab Error!", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow(window_name, error_frame)
                time.sleep(0.1) # 避免在错误状态下快速循环
            else:
                # 3. (可选) 缩小图像以便显示
                display_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                
                # 4. 显示图像
                cv2.imshow(window_name, display_frame)

            # 5. 处理按键事件 (至关重要!)
            key = cv2.waitKey(1) # 等待1毫秒，让OpenCV处理GUI事件并检测按键
            if key == 27: # ESC 键的 ASCII 码是 27
                print("ESC key pressed. Exiting loop.")
                break

            frame_count += 1
            # (可选) 计算并打印FPS
            # if time.time() - start_time >= 1.0: # 每秒打印一次
            #     fps = frame_count / (time.time() - start_time)
            #     print(f"FPS: {fps:.2f}")
            #     frame_count = 0
            #     start_time = time.time()

    except KeyboardInterrupt: # 处理 Ctrl+C
        print("\nInterrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\nError during minimal render loop:")
        traceback.print_exc()
    finally:
        print("Closing OpenCV window...")
        cv2.destroyAllWindows()
        # 确保窗口完全关闭
        for _ in range(5):
            cv2.waitKey(1)
        print("Minimal render test finished.")

if __name__ == "__main__":
    print("--- Minimal Screen Grab and Render Test ---")
    run_minimal_render_test()