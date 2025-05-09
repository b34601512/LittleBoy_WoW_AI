# D:\wow_ai\wow_rl\envs\wow_key_env.py
# (Version for Keyboard Trigger Test - Includes cv2.startWindowThread - SYNTAX FIXED)
import gymnasium as gym
import numpy as np
import cv2
import time
from gymnasium import spaces
import traceback # Import traceback for better error printing

# --- Assuming these paths are correct and classes are defined ---
try:
    from wow_rl.utils.screen_grabber import ScreenGrabber
    from wow_rl.utils.interactor_keys import send_key, KEY_FUNCTIONS
    from wow_rl.utils.reward_sensor import RewardSensor
    from wow_rl.utils.error_sensor import ErrorMessageSensor # Template matching version
except ImportError as e:
    print(f"CRITICAL ERROR in wow_key_env.py: Failed to import utility classes.")
    print(e)
    raise e
# --- End Imports ---

class WowKeyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self,
                 detector_w=r'D:\wow_ai\runs\detect\gwyx_detect_run_s_ep200_neg\weights\best.pt',
                 render_mode=None, max_steps=50,
                 # --- Use your confirmed ROI and Threshold for 0.99 score ---
                 roi_reward=(250, 35, 205, 64), # !!! REPLACE with your exact ROI if different !!!
                 template_path=r'D:\wow_ai\data\target_frame_template.png',
                 match_threshold=0.90, # !!! REPLACE with your optimal threshold (e.g., 0.9, 0.85) !!!
                 # --- End Confirmed Params ---
                 target_class_index=0,
                 roi_error=(800, 110, 330, 90)
                 ):
        super().__init__()
        print("Initializing WowKeyEnv (v_KeyboardTrigger_ThreadedRender)...") # Updated version name

        # Initialize Utilities
        try:
            self.grab = ScreenGrabber()
            print("ScreenGrabber initialized.")
            self.rs = RewardSensor(roi=roi_reward, template_path=template_path, match_thresh=match_threshold,
                                   yolo_weight=detector_w, target_idx=target_class_index)
            print(f"RewardSensor initialized with ROI: {roi_reward}, Match Threshold: {match_threshold}.")
            self.es = ErrorMessageSensor(roi=roi_error)
            print("ErrorMessageSensor (Template Matching) initialized.")
        except Exception as e:
            print(f"ERROR during utility initialization: {e}")
            traceback.print_exc()
            raise e

        # Define Action Space
        self.action_space = spaces.Discrete(len(KEY_FUNCTIONS))
        self.action_names = { 0: 'Tab', 1: 'Shift_Tab', 2: 'G_Key', 3: 'F_Attack', 4: 'No_Op' }
        print(f"Action space: Discrete({len(KEY_FUNCTIONS)}) -> {self.action_names}")

        # Define Observation Space (placeholders)
        self.error_keys = list(ErrorMessageSensor.DEFAULT_TEMPLATES.keys())
        print(f"Error keys (for analysis): {self.error_keys}")
        self.observation_space = spaces.Dict({
            "frame":   spaces.Box(low=0, high=255, shape=(96, 54, 3), dtype=np.uint8),
            "sel_flag":spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            "yolo":    spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            "errors":  spaces.MultiBinary(len(self.error_keys))
        })
        print(f"Observation space structure (placeholders for sensors): {self.observation_space}")

        # Other attributes
        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode
        self.window_name = "WowKeyEnv Feed (Keyboard Trigger)"

        # Initialize and start OpenCV window thread
        if self.render_mode == "human":
            print("Creating OpenCV window...")
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            try:
                cv2.startWindowThread()
                print("Started OpenCV window thread.")
            except Exception as e:
                print(f"Warning: cv2.startWindowThread() failed or not available: {e}")

        self.last_grabbed_frame = None
        print("WowKeyEnv (v_KeyboardTrigger_ThreadedRender) initialized successfully.")


    # _get_obs remains simplified
    def _get_obs(self, full_frame):
        if full_frame is None or full_frame.size == 0:
             return {"frame": np.zeros(self.observation_space["frame"].shape, dtype=np.uint8),"sel_flag": np.array([0], dtype=np.uint8),"yolo": np.zeros(self.observation_space["yolo"].shape, dtype=np.float32),"errors": np.zeros(self.observation_space["errors"].shape, dtype=np.int8)}
        try: frame_resized = cv2.resize(full_frame, (96, 54), interpolation=cv2.INTER_AREA)
        except Exception as e: print(f"Error resizing frame in _get_obs: {e}"); return {"frame": np.zeros(self.observation_space["frame"].shape, dtype=np.uint8),"sel_flag": np.array([0], dtype=np.uint8),"yolo": np.zeros(self.observation_space["yolo"].shape, dtype=np.float32),"errors": np.zeros(self.observation_space["errors"].shape, dtype=np.int8)}
        return {"frame": frame_resized.astype(np.uint8),"sel_flag": np.array([0], dtype=np.uint8),"yolo": np.zeros(self.observation_space["yolo"].shape, dtype=np.float32),"errors": np.zeros(self.observation_space["errors"].shape, dtype=np.int8)}

    # analyze_state_from_frame remains the same
    def analyze_state_from_frame(self, frame_to_analyze):
        if frame_to_analyze is None or frame_to_analyze.size == 0: return None
        analysis_result = {}
        try:
            sel_flag_raw, yolo_logits = self.rs.analyze(frame_to_analyze)
            analysis_result["sel_flag"] = int(sel_flag_raw)
            if isinstance(yolo_logits, np.ndarray) and yolo_logits.size > 0: analysis_result["yolo_prob_target"] = float(yolo_logits[0])
            else: analysis_result["yolo_prob_target"] = 0.0
            error_flags_dict = self.es.detect(frame_to_analyze)
            analysis_result["errors_dict"] = error_flags_dict
            analysis_result['need_face'] = error_flags_dict.get('face', 0) == 1
            analysis_result['need_range'] = error_flags_dict.get('range', 0) == 1
            analysis_result['no_target_error'] = error_flags_dict.get('no_target', 0) == 1
            return analysis_result
        except Exception as e: print(f"ERROR during state analysis: {e}"); traceback.print_exc(); return analysis_result if analysis_result else None

    # step remains simplified
    def step(self, action: int):
        try: send_key(action, wait=0.05)
        except Exception as e: print(f"ERROR sending key for action {action}: {e}")
        try: self.last_grabbed_frame = self.grab.grab()
        except Exception as e: print(f"ERROR grabbing screen in step: {e}"); self.last_grabbed_frame = None
        obs = self._get_obs(self.last_grabbed_frame)
        reward = 0.0; terminated = False; truncated = self.current_step >= self.max_steps; info = {}
        self.current_step += 1
        if self.render_mode == "human": self.render()
        return obs, reward, terminated, truncated, info

    # reset remains the same
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.current_step = 0; print("Environment reset."); time.sleep(0.2)
        try: self.last_grabbed_frame = self.grab.grab()
        except Exception as e: print(f"ERROR grabbing screen in reset: {e}"); self.last_grabbed_frame = None
        observation = self._get_obs(self.last_grabbed_frame); info = {}
        if self.render_mode == "human": self.render()
        return observation, info

    # render method is simplified - SYNTAX FIXED HERE
    def render(self):
         if self.render_mode == "human":
            display_frame = None
            if self.last_grabbed_frame is not None and self.last_grabbed_frame.size > 0:
                try:
                    vis_frame = self.last_grabbed_frame.copy()
                    display_frame = cv2.resize(vis_frame, (640, 360), interpolation=cv2.INTER_AREA)
                    rx, ry, rw, rh = self.rs.x, self.rs.y, self.rs.w, self.rs.h; cv2.rectangle(display_frame, (rx,ry), (rx+rw, ry+rh), (0,255,255), 1); cv2.putText(display_frame, "TargetROI", (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255),1)
                    ex, ey, ew, eh = self.es.x, self.es.y, self.es.w, self.es.h; cv2.rectangle(display_frame, (ex,ey), (ex+ew, ey+eh), (255,0,255), 1); cv2.putText(display_frame, "ErrorROI", (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255),1)
                except Exception as e: print(f"Error preparing frame for rendering: {e}"); display_frame = None

            if display_frame is None:
                display_frame = np.zeros((360, 640, 3), dtype=np.uint8); cv2.putText(display_frame, "Frame Error or Waiting...", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

            try:
                 cv2.imshow(self.window_name, display_frame)
                 # !!!!! 使用正确的 waitKey !!!!!
                 cv2.waitKey(30) # Use waitKey(30)
            except Exception as e:
                 print(f"Error during cv2.imshow or cv2.waitKey: {e}")
                 # !!!!! 正确的嵌套 try-except 来关闭窗口 !!!!!
                 try:
                     cv2.destroyWindow(self.window_name)
                     print("Force closed render window due to error.")
                 except Exception as inner_e:
                     # 如果销毁窗口也失败，就忽略内部错误
                     # print(f"Also failed to destroy window: {inner_e}")
                     pass
                 # !!!!! 嵌套结束 !!!!!

    # close remains the same
    def close(self):
        if self.render_mode == "human":
            print("Closing OpenCV window.")
            try: cv2.destroyWindow(self.window_name); cv2.waitKey(10)
            except Exception as e: print(f"Error destroying window in close: {e}")
        print("WowKeyEnv closed.")