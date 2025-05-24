# D:\wow_ai\wow_rl\utils\error_sensor.py
import cv2
import numpy as np
import os
import traceback

class ErrorMessageSensor:
    DEFAULT_ERROR_ROI = (800, 110, 330, 90) # ROI仍然用来指定大概的搜索区域

    # 定义模板文件路径和对应的标识符
    # 请确保这些路径与你保存的模板文件名一致
    DEFAULT_TEMPLATES = {
        # 原有的错误模板
        "face": r"D:\wow_ai\data\template_error_face.png",
        "range": r"D:\wow_ai\data\template_error_range.png",
        "no_target": r"D:\wow_ai\data\template_error_no_target.png",
        "cant_attack_target": r"D:\wow_ai\data\cant_attack_target.png",
        
        # 新增的错误模板 - 用户最新截图
        "facing_wrong_way": r"D:\wow_ai\data\you_are_facing_the_wrong_way.png",  # 面朝错误方向
        "spell_not_ready": r"D:\wow_ai\data\spell_is_not_ready_yet.png",          # 法术没准备好/技能冷却
        "too_far_away": r"D:\wow_ai\data\you_are_too_far_away.png",              # 你距离太远(与range类似但不同措辞)
        "player_dead": r"D:\wow_ai\data\you_are_dead.png",                       # 玩家死亡状态
        "cannot_attack_while_dead": r"D:\wow_ai\data\you_cannot_attack_while_dead.png",  # 死亡状态不能攻击
        "no_attackable_target": r"D:\wow_ai\data\template_error_no_attackable_target.png"  # 没有可攻击目标
    }
    # 模板匹配的阈值，可能需要根据实际情况微调
    DEFAULT_MATCH_THRESHOLD = 0.51 

    def __init__(self, roi=DEFAULT_ERROR_ROI, templates_info=DEFAULT_TEMPLATES, threshold=DEFAULT_MATCH_THRESHOLD):
        self.x, self.y, self.w, self.h = roi
        self.match_threshold = threshold
        self.templates = {} # 用于存储加载后的模板图像

        print(f"ErrorMessageSensor (Template Matching) initialized. ROI: {roi}, Threshold: {threshold}")
        for identifier, path in templates_info.items():
            if not os.path.exists(path):
                print(f"WARNING: Template file not found for '{identifier}': {path}")
                continue
            template_img = cv2.imread(path, cv2.IMREAD_COLOR) # 或者 cv2.IMREAD_GRAYSCALE 如果模板是灰度
            if template_img is None:
                print(f"WARNING: Failed to load template for '{identifier}' from {path}")
                continue
            self.templates[identifier] = template_img
            print(f"  Loaded template for '{identifier}' from {path}, shape: {template_img.shape}")
        
        if not self.templates:
            print("CRITICAL WARNING: No templates loaded. ErrorMessageSensor will not detect anything.")

    def detect(self, frame):
        error_flags = {identifier: 0 for identifier in self.templates.keys()}
        
        try:
            frame_h, frame_w = frame.shape[:2]
            roi_x_start, roi_y_start = self.x, self.y
            roi_x_end, roi_y_end = min(self.x + self.w, frame_w), min(self.y + self.h, frame_h)

            if roi_x_end <= roi_x_start or roi_y_end <= roi_y_start:
                return error_flags

            image_roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

            if image_roi.size == 0:
                return error_flags

            # cv2.imshow("[Debug] Error Sensor ROI", image_roi) # 用于调试ROI区域

            for identifier, template_img in self.templates.items():
                if template_img.shape[0] > image_roi.shape[0] or template_img.shape[1] > image_roi.shape[1]:
                    # print(f"Skipping template {identifier}, template larger than ROI.")
                    continue

                # 使用 TM_CCOEFF_NORMED 方法进行匹配
                # 其他方法如 TM_SQDIFF_NORMED (值越小越好) 也可以尝试
                result = cv2.matchTemplate(image_roi, template_img, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                print(f"  [ErrorSensor Debug] Template '{identifier}': MaxMatchVal = {max_val:.4f} (Threshold: {self.match_threshold})")

                # print(f"  Template '{identifier}': Max match value = {max_val:.4f}") # 打印匹配值用于调试阈值

                if max_val >= self.match_threshold:
                    error_flags[identifier] = 1
                    print(f"MATCH FOUND (Template)! Identifier: '{identifier}', Value: {max_val:.2f} >= {self.match_threshold}")
                    # 可选：在ROI中绘制匹配位置
                    # t_h, t_w = template_img.shape[:2]
                    # top_left = max_loc
                    # bottom_right = (top_left[0] + t_w, top_left[1] + t_h)
                    # cv2.rectangle(image_roi, top_left, bottom_right, (0,255,0), 2)
                    # cv2.imshow("[Debug] Match Found in ROI", image_roi)
                    break # 如果找到一个匹配，可以假设只有一个错误提示同时出现，提前退出

        except Exception as e:
            print(f"ERROR in ErrorMessageSensor.detect (Template Matching): {e}")
            traceback.print_exc()
        
        return error_flags