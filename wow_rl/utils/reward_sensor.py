# D:\wow_ai\wow_rl\utils\reward_sensor.py
# (Version with integrated 'target dead' template matching)
import cv2
import numpy as np
import os
import torch # RewardSensor v3 uses YOLO
from ultralytics import YOLO # RewardSensor v3 uses YOLO

class RewardSensor:
    # --- Configuration for Target Selection (Red Frame) ---
    DEFAULT_ROI_REWARD = (250, 35, 205, 64) # Your confirmed ROI for target selection
    DEFAULT_TEMPLATE_PATH_SELECTED = r'D:\wow_ai\data\target_frame_template.png' # Your 0.99 score template
    DEFAULT_MATCH_THRESHOLD_SELECTED = 0.90 # Your high threshold for selection

    # --- Configuration for Target Dead Status ---
    # !! IMPORTANT: This ROI might need to be different from DEFAULT_ROI_REWARD !!
    # !! It should specifically cover where the "死亡" text appears within the target frame.
    # !! For now, we'll assume it's within or the same as DEFAULT_ROI_REWARD, but adjust if needed.
    # !! If "死亡" appears in a very different part of the target frame,
    # !! you might need a separate ROI for it, or make DEFAULT_ROI_REWARD large enough to cover both.
    DEFAULT_TEMPLATE_PATH_DEAD = r'D:\wow_ai\data\template_target_dead.png'
    DEFAULT_MATCH_THRESHOLD_DEAD = 0.7 # Threshold for "dead" template, can be adjusted
                                       # "死亡"文字特征可能比较明显，可以先用一个略低于选中的阈值

    # --- Configuration for YOLO (if used for target type identification) ---
    DEFAULT_YOLO_WEIGHT = r'D:\wow_ai\runs\detect\gwyx_detect_run_s_ep200_neg\weights\best.pt'
    TARGET_CLASS_INDEX = 0 # 林精的类别索引 (if YOLO is used)

    def __init__(self, 
                 roi_select=DEFAULT_ROI_REWARD, 
                 template_path_select=DEFAULT_TEMPLATE_PATH_SELECTED,
                 match_thresh_select=DEFAULT_MATCH_THRESHOLD_SELECTED,
                 template_path_dead=DEFAULT_TEMPLATE_PATH_DEAD,
                 match_thresh_dead=DEFAULT_MATCH_THRESHOLD_DEAD,
                 yolo_weight=DEFAULT_YOLO_WEIGHT, # Keep YOLO params for now
                 target_idx=TARGET_CLASS_INDEX,   # Keep YOLO params for now
                 device='cuda:0'
                 ):
        
        self.roi_select_x, self.roi_select_y, self.roi_select_w, self.roi_select_h = roi_select
        self.match_thresh_select = match_thresh_select
        self.match_thresh_dead = match_thresh_dead
        
        self.template_selected = self._load_template(template_path_select, "selected")
        self.template_dead = self._load_template(template_path_dead, "dead")

        # --- YOLO Initialization (from v3) ---
        self.target_idx = target_idx
        self.device = device
        try:
            self.det = YOLO(yolo_weight)
            print(f"RewardSensor: YOLO detector loaded from {yolo_weight}")
        except Exception as e:
            print(f"RewardSensor: Error loading YOLO model {yolo_weight}: {e}")
            # Decide if this should be a fatal error. For now, allow to proceed without YOLO.
            self.det = None 
            print("RewardSensor: Proceeding without YOLO functionality due to loading error.")
        # --- End YOLO Initialization ---

        print(f"RewardSensor Initialized:")
        if self.template_selected is not None:
            print(f"  - Target Selected: ROI={roi_select}, Thresh={match_thresh_select}")
        if self.template_dead is not None:
            print(f"  - Target Dead: Using same ROI as selection (or part of it), Thresh={match_thresh_dead}")


    def _load_template(self, template_path, template_name="unknown"):
        if not os.path.exists(template_path):
            print(f"WARNING (RewardSensor): Template file for '{template_name}' not found: {template_path}")
            return None
        template = cv2.imread(template_path, cv2.IMREAD_COLOR) # Load as color
        if template is None:
            print(f"WARNING (RewardSensor): Failed to load template for '{template_name}' from {template_path}")
            return None
        print(f"  Template '{template_name}' loaded from {template_path}, shape: {template.shape}")
        return template

    @torch.no_grad() # For YOLO part, if active
    def analyze(self, frame_full):
        sel_flag = 0
        is_dead_flag = 0
        yolo_logits = np.zeros(2, dtype=np.float32) # Default YOLO output

        try:
            # --- 1. Target Selection (Red Frame) and Target Dead detection ---
            # We will use the self.roi_select_x,y,w,h for both for now.
            # If "dead" text is far from "selected" frame's main features, this ROI might need adjustment
            # or we might need a separate ROI for the "dead" template.
            
            frame_h, frame_w = frame_full.shape[:2]
            roi_x_start = self.roi_select_x
            roi_y_start = self.roi_select_y
            roi_x_end = min(roi_x_start + self.roi_select_w, frame_w)
            roi_y_end = min(roi_y_start + self.roi_select_h, frame_h)

            if roi_x_end <= roi_x_start or roi_y_end <= roi_y_start:
                # print("Warning (RewardSensor): ROI for selection/dead is invalid.")
                return sel_flag, is_dead_flag, yolo_logits # Return defaults

            roi_crop_for_match = frame_full[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

            if roi_crop_for_match.size == 0:
                # print("Warning (RewardSensor): ROI crop is empty.")
                return sel_flag, is_dead_flag, yolo_logits

            # --- Match for "Target Selected" ---
            if self.template_selected is not None:
                if self.template_selected.shape[0] <= roi_crop_for_match.shape[0] and \
                   self.template_selected.shape[1] <= roi_crop_for_match.shape[1]:
                    result_select = cv2.matchTemplate(roi_crop_for_match, self.template_selected, cv2.TM_CCOEFF_NORMED)
                    _, max_val_select, _, _ = cv2.minMaxLoc(result_select)
                    # print(f"  [RS Debug] Selected Template MatchVal: {max_val_select:.4f}") # Debug
                    if max_val_select >= self.match_thresh_select:
                        sel_flag = 1
                # else: print("Warning (RewardSensor): 'selected' template larger than ROI crop.")


            # --- Match for "Target Dead" (only if a target might be selected or was just selected) ---
            # We can check for "dead" regardless of sel_flag, or only if sel_flag was 1 recently.
            # For simplicity, let's check if the template exists.
            # This will run on the same roi_crop_for_match.
            if self.template_dead is not None:
                if self.template_dead.shape[0] <= roi_crop_for_match.shape[0] and \
                   self.template_dead.shape[1] <= roi_crop_for_match.shape[1]:
                    result_dead = cv2.matchTemplate(roi_crop_for_match, self.template_dead, cv2.TM_CCOEFF_NORMED)
                    _, max_val_dead, _, _ = cv2.minMaxLoc(result_dead)
                    print(f"  [RS Inside Analyze] Dead Template MatchVal: {max_val_dead:.4f} (Thresh: {self.match_thresh_dead})")
                    if max_val_dead >= self.match_thresh_dead:
                        is_dead_flag = 1
                        # If target is dead, it's arguably not "selected" in an actionable way anymore.
                        # We can let sel_flag remain as is, or force it to 0.
                        # For now, let WowKeyEnv handle the implication of is_dead_flag.
                        # sel_flag = 0 # Optional: if dead, then not actively selected
                # else: print("Warning (RewardSensor): 'dead' template larger than ROI crop.")

            # --- 2. YOLO Analysis (from v3 - only if target is selected and NOT dead, and YOLO model exists) ---
            if sel_flag == 1 and is_dead_flag == 0 and self.det is not None:
                # print("DEBUG (RewardSensor): Performing YOLO on ROI_crop")
                # Ensure roi_crop_for_match is suitable for YOLO (e.g., BGR)
                res_yolo = self.det.predict(roi_crop_for_match, device=self.device, verbose=False, imgsz=224, conf=0.1)
                if res_yolo and len(res_yolo[0].boxes):
                    best_box_idx = res_yolo[0].boxes.conf.argmax()
                    cls = int(res_yolo[0].boxes.cls[best_box_idx])
                    prob = float(res_yolo[0].boxes.conf[best_box_idx])
                    if cls == self.target_idx:
                        yolo_logits[0] = prob # Probability of target class (e.g.,林精)
                    else:
                        yolo_logits[1] = prob # Probability of other class
                # else: print("DEBUG (RewardSensor): YOLO found no boxes or empty result.")
            # else:
                # if self.det is None: print("DEBUG (RewardSensor): YOLO detector not available.")
                # if sel_flag == 0 : print("DEBUG (RewardSensor): No target selected, skipping YOLO.")
                # if is_dead_flag == 1: print("DEBUG (RewardSensor): Target is dead, skipping YOLO.")


        except Exception as e:
            print(f"ERROR in RewardSensor.analyze: {e}")
            traceback.print_exc()
            # Return defaults on error to prevent crashes
            return 0, 0, np.zeros(2, dtype=np.float32)

        return sel_flag, is_dead_flag, yolo_logits.astype(np.float32)