# D:\wow_ai\wow_rl\utils\reward_sensor.py
# (Version: Detects red AND neutral target frames, and dead status)
import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
import traceback

class RewardSensor:
    # --- Target Selection ---
    DEFAULT_ROI_SELECT = (250, 35, 205, 64) # Your primary ROI for target frames
    DEFAULT_TEMPLATE_PATH_HOSTILE = r'D:\wow_ai\data\target_frame_template.png' # Red hostile
    DEFAULT_TEMPLATE_PATH_NEUTRAL = r'D:\wow_ai\data\template_target_neutral_selected.png' # Yellow neutral
    DEFAULT_MATCH_THRESHOLD_SELECT = 0.5 # Your effective threshold for both

    # --- Target Dead Status (uses the same ROI_SELECT for now) ---
    DEFAULT_TEMPLATE_PATH_DEAD = r'D:\wow_ai\data\template_target_dead.png'
    DEFAULT_MATCH_THRESHOLD_DEAD = 0.5 # Your effective threshold for dead

    # --- YOLO (Optional, can be disabled if yolo_weight is None) ---
    DEFAULT_YOLO_WEIGHT = r'D:\wow_ai\runs\detect\gwyx_detect_run_s_ep200_neg\weights\best.pt'
    TARGET_CLASS_INDEX = 0 

    def __init__(self, 
                 roi_select=DEFAULT_ROI_SELECT, 
                 template_path_hostile=DEFAULT_TEMPLATE_PATH_HOSTILE,
                 template_path_neutral=DEFAULT_TEMPLATE_PATH_NEUTRAL,
                 match_thresh_select=DEFAULT_MATCH_THRESHOLD_SELECT,
                 template_path_dead=DEFAULT_TEMPLATE_PATH_DEAD,
                 match_thresh_dead=DEFAULT_MATCH_THRESHOLD_DEAD,
                 yolo_weight=DEFAULT_YOLO_WEIGHT, 
                 target_idx=TARGET_CLASS_INDEX,  
                 device='cuda:0'):
        
        self.roi_select_x, self.roi_select_y, self.roi_select_w, self.roi_select_h = roi_select
        self.match_thresh_select = match_thresh_select
        self.match_thresh_dead = match_thresh_dead
        
        self.template_hostile = self._load_template(template_path_hostile, "hostile_selected")
        self.template_neutral = self._load_template(template_path_neutral, "neutral_selected")
        self.template_dead = self._load_template(template_path_dead, "dead")

        self.target_idx = target_idx
        self.device = device
        self.det = None # Initialize YOLO detector as None
        if yolo_weight and os.path.exists(yolo_weight):
            try:
                self.det = YOLO(yolo_weight)
                print(f"RewardSensor: YOLO detector loaded from {yolo_weight}")
            except Exception as e:
                print(f"RewardSensor: Error loading YOLO model {yolo_weight}: {e}. Proceeding without YOLO.")
        else:
            print("RewardSensor: YOLO weight path not provided or not found. Proceeding without YOLO.")

        print(f"RewardSensor Initialized:")
        if self.template_hostile is not None: print(f"  - Hostile Select: ROI={roi_select}, Thresh={match_thresh_select}")
        if self.template_neutral is not None: print(f"  - Neutral Select: ROI={roi_select}, Thresh={match_thresh_select}")
        if self.template_dead is not None: print(f"  - Target Dead: Using ROI={roi_select}, Thresh={match_thresh_dead}")

    def _load_template(self, template_path, template_name="unknown"):
        if not os.path.exists(template_path):
            print(f"WARNING (RewardSensor): Template for '{template_name}' NOT FOUND: {template_path}")
            return None
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if template is None: print(f"WARNING (RewardSensor): Failed to LOAD template '{template_name}' from {template_path}"); return None
        print(f"  Template '{template_name}' loaded from {template_path}, shape: {template.shape}")
        return template

    def _match_template_in_roi(self, roi_crop, template):
        if template is None or roi_crop.size == 0: return 0.0
        if template.shape[0] > roi_crop.shape[0] or template.shape[1] > roi_crop.shape[1]: return 0.0
        result = cv2.matchTemplate(roi_crop, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val

    @torch.no_grad()
    def analyze(self, frame_full):
        sel_flag = 0
        is_dead_flag = 0
        yolo_logits = np.zeros(2, dtype=np.float32)

        try:
            frame_h, frame_w = frame_full.shape[:2]
            rs_x, rs_y, rs_w, rs_h = self.roi_select_x, self.roi_select_y, self.roi_select_w, self.roi_select_h
            roi_x_end = min(rs_x + rs_w, frame_w); roi_y_end = min(rs_y + rs_h, frame_h)
            if roi_x_end <= rs_x or roi_y_end <= rs_y: return 0,0,yolo_logits
            roi_crop = frame_full[rs_y:roi_y_end, rs_x:roi_x_end]
            if roi_crop.size == 0: return 0,0,yolo_logits

            # 1. Check for Dead status first (as it might override selection)
            match_val_dead = self._match_template_in_roi(roi_crop, self.template_dead)
            # print(f"  [RS Inside Analyze] Dead Template MatchVal: {match_val_dead:.4f}") # Keep for debug
            if match_val_dead >= self.match_thresh_dead:
                is_dead_flag = 1
            
            # 2. Check for Target Selection (Hostile OR Neutral)
            # Only consider selected if not confirmed dead by its own template in this frame
            if is_dead_flag == 0: # If not dead, check if selected
                match_val_hostile = self._match_template_in_roi(roi_crop, self.template_hostile)
                match_val_neutral = self._match_template_in_roi(roi_crop, self.template_neutral)
                # print(f"  [RS Inside Analyze] Hostile Select MatchVal: {match_val_hostile:.4f}") # Debug
                # print(f"  [RS Inside Analyze] Neutral Select MatchVal: {match_val_neutral:.4f}") # Debug
                if match_val_hostile >= self.match_thresh_select or \
                   match_val_neutral >= self.match_thresh_select:
                    sel_flag = 1
            
            # 3. YOLO Analysis (Optional: if selected, not dead, and detector exists)
            if sel_flag == 1 and is_dead_flag == 0 and self.det is not None:
                res_yolo = self.det.predict(roi_crop, device=self.device, verbose=False, imgsz=224, conf=0.1)
                if res_yolo and len(res_yolo[0].boxes):
                    best_box_idx = res_yolo[0].boxes.conf.argmax()
                    cls = int(res_yolo[0].boxes.cls[best_box_idx]); prob = float(res_yolo[0].boxes.conf[best_box_idx])
                    if cls == self.target_idx: yolo_logits[0] = prob
                    else: yolo_logits[1] = prob
        except Exception as e:
            print(f"ERROR in RewardSensor.analyze: {e}"); traceback.print_exc()
            return 0, 0, np.zeros(2, dtype=np.float32)
        return sel_flag, is_dead_flag, yolo_logits