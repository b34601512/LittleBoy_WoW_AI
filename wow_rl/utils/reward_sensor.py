# wow_rl/utils/reward_sensor.py (修正版 v3 - 结合模板匹配和YOLO)
import cv2
import numpy as np
import os
import torch # 导入 torch
from ultralytics import YOLO # 需要导入 YOLO

class RewardSensor:
    # --- 配置参数 (确保这些值是你最终验证有效的) ---
    DEFAULT_ROI = (319, 35, 245, 64)
    DEFAULT_TEMPLATE_PATH = r'D:\wow_ai\data\target_frame_template.png'
    DEFAULT_MATCH_THRESHOLD = 0.85 # 使用我们验证过的可靠阈值
    DEFAULT_YOLO_WEIGHT = r'D:\wow_ai\runs\detect\gwyx_detect_run_s_ep200_neg\weights\best.pt'
    TARGET_CLASS_INDEX = 0 # 假设林精是第 0 类
    # --- 配置结束 ---

    def __init__(self, roi=DEFAULT_ROI, template_path=DEFAULT_TEMPLATE_PATH,
                 match_thresh=DEFAULT_MATCH_THRESHOLD, yolo_weight=DEFAULT_YOLO_WEIGHT,
                 target_idx=TARGET_CLASS_INDEX, device='cuda:0'):

        self.x, self.y, self.w, self.h = roi
        self.match_thresh = match_thresh
        self.target_idx = target_idx
        self.device = device

        self.template = self._load_template(template_path)
        if self.template is None:
            raise ValueError(f"无法加载模板图像: {template_path}")

        try:
            self.det = YOLO(yolo_weight)
            print(f"RewardSensor: YOLO detector loaded from {yolo_weight}")
        except Exception as e:
            print(f"RewardSensor: Error loading YOLO model {yolo_weight}: {e}")
            raise e

        print(f"RewardSensor 使用 ROI {roi}, 模板 {template_path}, 匹配阈值 {match_thresh}, 目标类别 {target_idx} 初始化")

    def _load_template(self, template_path):
        if not os.path.exists(template_path):
            print(f"错误: 模板文件未找到 '{template_path}'")
            return None
        template_img = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if template_img is None:
            print(f"错误: 无法加载模板 '{template_path}'")
            return None
        return template_img

    @torch.no_grad()
    def analyze(self, frame):
        sel_flag = 0
        yolo_logits = np.zeros(2, dtype=np.float32)
        match_score = -1.0

        if self.template is None:
            return sel_flag, yolo_logits

        try:
            frame_h, frame_w = frame.shape[:2]
            roi_x_end = min(self.x + self.w, frame_w)
            roi_y_end = min(self.y + self.h, frame_h)
            roi_x_start = max(0, self.x)
            roi_y_start = max(0, self.y)

            if roi_x_end <= roi_x_start or roi_y_end <= roi_y_start:
                return sel_flag, yolo_logits

            roi_crop = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

            if roi_crop.size == 0 or self.template.shape[0] > roi_crop.shape[0] or self.template.shape[1] > roi_crop.shape[1]:
                return sel_flag, yolo_logits

            result = cv2.matchTemplate(roi_crop, self.template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            match_score = max_val
            if match_score >= self.match_thresh:
                sel_flag = 1

            if sel_flag == 1:
                res = self.det.predict(roi_crop, device=self.device, verbose=False, imgsz=224, conf=0.1)[0]
                if len(res.boxes):
                    best_box_idx = res.boxes.conf.argmax()
                    cls = int(res.boxes.cls[best_box_idx])
                    prob = float(res.boxes.conf[best_box_idx])
                    if cls == self.target_idx:
                        yolo_logits[0] = prob
                    else:
                        yolo_logits[1] = prob

            return sel_flag, yolo_logits.astype(np.float32)

        except Exception as e:
            print(f"Error in RewardSensor.analyze: {e}")
            return sel_flag, yolo_logits.astype(np.float32)