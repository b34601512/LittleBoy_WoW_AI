# wow_rl/utils/screen_grabber.py
import mss, numpy as np, cv2

class ScreenGrabber:
    def __init__(self, region=(0, 0, 1920, 1080)): # 默认截取整个 1920x1080 屏幕
        self.mon = {'left':region[0], 'top':region[1],
                    'width':region[2], 'height':region[3]}
        self.sct = mss.mss()

    def grab(self):
        frame = np.array(self.sct.grab(self.mon))
        # 移除 alpha 通道（如果存在）并将 BGRA 转换为 BGR
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)