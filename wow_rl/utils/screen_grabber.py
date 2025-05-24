# wow_rl/utils/screen_grabber.py
import mss, numpy as np, cv2
import time

class ScreenGrabber:
    def __init__(self, region=(0, 0, 1920, 1080)): # 默认截取整个 1920x1080 屏幕
        self.mon = {'left':region[0], 'top':region[1],
                    'width':region[2], 'height':region[3]}
        self.sct = mss.mss()
        print(f"ScreenGrabber initialized with region: {region}")
        # 测试是否可以成功截图
        try:
            test_frame = self.grab()
            if test_frame is not None and test_frame.size > 0:
                print(f"Initial screen grab successful, shape: {test_frame.shape}")
            else:
                print("WARNING: Initial screen grab returned empty frame")
        except Exception as e:
            print(f"WARNING: Failed to take initial screen capture: {e}")

    def grab(self):
        try:
            frame = np.array(self.sct.grab(self.mon))
            # 移除 alpha 通道（如果存在）并将 BGRA 转换为 BGR
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            print(f"ERROR in screen grab: {e}")
            # 添加短暂延迟，防止错误连续输出
            time.sleep(0.1)
            return None