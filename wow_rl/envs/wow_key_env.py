# D:\wow_ai\wow_rl\envs\wow_key_env.py
# (Version: Reduced reward for switching live targets, no post-attack sleep)
import gymnasium as gym
import numpy as np
import cv2
import time
from gymnasium import spaces
import traceback

try:
    from wow_rl.utils.screen_grabber import ScreenGrabber
    from wow_rl.utils.interactor_keys import send_key, KEY_FUNCTIONS
    from wow_rl.utils.reward_sensor import RewardSensor
    from wow_rl.utils.error_sensor import ErrorMessageSensor
except ImportError as e: print(f"CRITICAL ERROR importing utils: {e}"); raise e

class WowKeyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self,
                 detector_w=r'D:\wow_ai\runs\detect\gwyx_detect_run_s_ep200_neg\weights\best.pt',
                 render_mode=None, max_steps=200,
                 roi_reward=(250, 35, 205, 64), template_path_select=r'D:\wow_ai\data\target_frame_template.png',
                 match_thresh_select=0.50, template_path_dead=r'D:\wow_ai\data\template_target_dead.png',
                 match_thresh_dead=0.5, # Using your effective threshold
                 roi_error=(800, 110, 330, 90),
                 error_match_thresh=0.5, target_class_index=0
                 ):
        super().__init__()
        print("Initializing WowKeyEnv (v_ReduceSwitchTargetReward)...") # New version name

        try:
            self.grab = ScreenGrabber()
            self.rs = RewardSensor(roi_select=roi_reward, template_path_select=template_path_select, match_thresh_select=match_thresh_select, template_path_dead=template_path_dead, match_thresh_dead=match_thresh_dead, yolo_weight=detector_w, target_idx=target_class_index)
            self.es = ErrorMessageSensor(roi=roi_error, threshold=error_match_thresh)
            print(f"RewardSensor: ThreshSelect={match_thresh_select}, ThreshDead={match_thresh_dead}")
            print(f"ErrorMessageSensor: ThreshError={error_match_thresh}")
        except Exception as e: print(f"ERROR init utilities: {e}"); traceback.print_exc(); raise e

        self.action_space = spaces.Discrete(len(KEY_FUNCTIONS))
        self.action_names = { 0: 'Tab', 1: 'Shift_Tab', 2: 'G_Key', 3: 'F_Attack', 4: 'No_Op' }
        self.error_keys = list(ErrorMessageSensor.DEFAULT_TEMPLATES.keys())
        self.observation_space = spaces.Dict({
            "frame":   spaces.Box(low=0, high=255, shape=(96, 54, 3), dtype=np.uint8),
            "sel_flag":spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            "is_dead": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            "yolo":    spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            "errors":  spaces.MultiBinary(len(self.error_keys))
        })

        self.max_steps = max_steps; self.current_step = 0
        self.render_mode = render_mode; self.window_name = "WowKeyEnv Feed (ReduceSwitchReward)"
        if self.render_mode == "human":
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            try: cv2.startWindowThread(); print("Started OpenCV window thread.")
            except Exception as e: print(f"Warning: cv2.startWindowThread() failed: {e}")

        self.last_grabbed_frame = None
        self.current_analysis = self._get_empty_analysis()
        self.prev_analysis_for_step = self._get_empty_analysis()
        self.agent_should_consider_g_key = False
        print("WowKeyEnv (v_ReduceSwitchTargetReward) initialized successfully.")

    def _get_empty_analysis(self):
        return {"sel_flag": 0, "is_dead": 0, "yolo_prob_target": 0.0, "errors_dict": {key: 0 for key in self.error_keys},'need_face': False, 'need_range': False, 'no_target_error': False}

    def _get_obs(self): # Renamed from _get_obs(self, current_analysis_results) for simplicity
        analysis_to_use = self.current_analysis 
        frame_data = self.last_grabbed_frame 
        if frame_data is None or frame_data.size == 0: frame_resized = np.zeros(self.observation_space["frame"].shape, dtype=np.uint8)
        else:
            try: frame_resized = cv2.resize(frame_data, (54, 96), interpolation=cv2.INTER_AREA)
            except Exception: frame_resized = np.zeros(self.observation_space["frame"].shape, dtype=np.uint8)
        return {"frame": frame_resized,
                "sel_flag": np.array([analysis_to_use['sel_flag']], dtype=np.uint8),
                "is_dead": np.array([analysis_to_use['is_dead']], dtype=np.uint8),
                "yolo": np.array([analysis_to_use['yolo_prob_target'], 0.0], dtype=np.float32),
                "errors": np.array([analysis_to_use['errors_dict'].get(key,0) for key in self.error_keys], dtype=np.int8)}

    def analyze_state_from_frame(self, frame_to_analyze):
        if frame_to_analyze is None or frame_to_analyze.size == 0: return self._get_empty_analysis()
        current_analysis_data = self._get_empty_analysis()
        try:
            sel_flag_raw, is_dead_raw, yolo_logits = self.rs.analyze(frame_to_analyze)
            current_analysis_data["sel_flag"] = int(sel_flag_raw)
            current_analysis_data["is_dead"] = int(is_dead_raw)
            if isinstance(yolo_logits, np.ndarray) and yolo_logits.size > 0: current_analysis_data["yolo_prob_target"] = float(yolo_logits[0])
            error_flags_dict = self.es.detect(frame_to_analyze)
            current_analysis_data["errors_dict"] = error_flags_dict
            current_analysis_data['need_face'] = error_flags_dict.get('face', 0) == 1
            current_analysis_data['need_range'] = error_flags_dict.get('range', 0) == 1
            current_analysis_data['no_target_error'] = error_flags_dict.get('no_target', 0) == 1
            return current_analysis_data
        except Exception as e: print(f"ERROR in analyze_state: {e}"); traceback.print_exc(); return current_analysis_data

    def step(self, action: int):
        self.prev_analysis_for_step = self.current_analysis.copy() 
        
        try: send_key(action, wait=0.05)
        except Exception as e: print(f"ERROR sending key for action {action}: {e}")
        
        # No fixed sleep after attack anymore, rely on natural game/sensor timing
            
        try: self.last_grabbed_frame = self.grab.grab()
        except Exception as e: print(f"ERROR grabbing screen: {e}"); self.last_grabbed_frame = None
        
        self.current_analysis = self.analyze_state_from_frame(self.last_grabbed_frame)

        reward = -0.05; terminated = False
        sel_now = self.current_analysis.get("sel_flag", 0); dead_now = self.current_analysis.get("is_dead", 0)
        no_target_err_now = self.current_analysis.get("no_target_error", False)
        prev_sel = self.prev_analysis_for_step.get("sel_flag", 0); prev_dead = self.prev_analysis_for_step.get("is_dead", 0)

        if prev_sel == 1 and prev_dead == 0 and dead_now == 1: 
            print("INFO (step): Target KILLED (direct detection)! Big Reward!")
            reward = 10.0; terminated = True; sel_now = 0; self.agent_should_consider_g_key = False 
        elif self.agent_should_consider_g_key and action == 2: 
            if sel_now == 1 and dead_now == 1: print("INFO (step): Target CONFIRMED DEAD via G-Key! Big Reward!"); reward = 15.0; terminated = True; sel_now = 0 
            elif sel_now == 1 and dead_now == 0: print("INFO (step): G-Key re-selected/found a LIVE target."); reward = 0.5 
            else: print("INFO (step): G-Key check was not conclusive."); reward = -0.2 
            self.agent_should_consider_g_key = False 
        elif sel_now == 1 and dead_now == 0: 
            self.agent_should_consider_g_key = False 
            # !!!!! Logic to reduce reward if switching from a live target !!!!!
            if (action == 0 or action == 1) and prev_sel == 1 and prev_dead == 0:
                print("INFO (step): Switched from a live target. No/Low selection reward.")
                reward = -0.05 # Or 0.0, or a very small positive like 0.01
            else:
                reward = 0.20 # Standard reward for selecting/having a live target
            # !!!!! End logic for switching target !!!!!
            
            if action == 3: # Attack
                need_face = self.current_analysis.get("need_face", False); need_range = self.current_analysis.get("need_range", False)
                if no_target_err_now: reward = -0.8; self.agent_should_consider_g_key = True;
                elif need_face: reward = -0.5
                elif need_range: reward = -0.3
                else: 
                    # If base reward was 0.20 (not a switch), add 0.8 for successful attack to make it 1.0
                    # If base reward was -0.05 (was a switch), add 1.05 to make it 1.0
                    # Simpler: just set reward to 1.0 if no error, and add bonus
                    reward = 1.0 # Good attack on live target
                    reward += 0.1 # Bonus for continuing valid attack
        elif sel_now == 0 and not terminated : 
            if prev_sel == 1 and prev_dead == 0 and action != 2 : print("INFO (step): Target lost. Prompting G-Key."); self.agent_should_consider_g_key = True; reward = -0.1 
            elif action == 3 and no_target_err_now: print("INFO (step): Attack with 'no target' error. Prompting G-Key."); self.agent_should_consider_g_key = True; reward = -0.15 
            elif action == 3: reward = -0.10; self.agent_should_consider_g_key = False
            else: self.agent_should_consider_g_key = False
        
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        if truncated and not terminated : terminated = True 
        
        info = self.current_analysis.copy()
        info.update({'step': self.current_step, 'reward_this_step': reward, 'action_taken': action, 'action_name': self.action_names.get(action, "Unknown"), 'agent_should_consider_g_key': self.agent_should_consider_g_key})
        
        agent_obs = self._get_obs() 

        if self.render_mode == "human": self.render(info_for_display=info)
        return agent_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.current_step = 0; print("Environment reset."); time.sleep(0.2)
        try: self.last_grabbed_frame = self.grab.grab()
        except Exception as e: print(f"ERROR grabbing screen in reset: {e}"); self.last_grabbed_frame = None
        
        self.current_analysis = self.analyze_state_from_frame(self.last_grabbed_frame)
        self.prev_analysis_for_step = self.current_analysis.copy() if self.current_analysis else self._get_empty_analysis()
        
        observation = self._get_obs()
        
        info = self.current_analysis.copy() if self.current_analysis else self._get_empty_analysis()
        info.update({'step': self.current_step, 'reward_this_step': 0.0, 'action_taken': -1, 'action_name': "N/A", 'agent_should_consider_g_key': False})
        
        self.agent_should_consider_g_key = False 
        if self.render_mode == "human": self.render(info_for_display=info)
        return observation, info

    def render(self, info_for_display=None):
         if self.render_mode == "human":
            display_frame_render = None
            if self.last_grabbed_frame is not None and self.last_grabbed_frame.size > 0:
                try:
                    vis_frame = self.last_grabbed_frame.copy()
                    display_frame_render = cv2.resize(vis_frame, (640, 360), interpolation=cv2.INTER_AREA)
                    rx,ry,rw,rh = self.rs.roi_select_x, self.rs.roi_select_y, self.rs.roi_select_w, self.rs.roi_select_h
                    cv2.rectangle(display_frame_render, (rx,ry), (rx+rw, ry+rh), (0,255,255), 1); cv2.putText(display_frame_render, "TargetROI", (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255),1)
                    ex,ey,ew,eh = self.es.x, self.es.y, self.es.w, self.es.h
                    cv2.rectangle(display_frame_render, (ex,ey), (ex+ew, ey+eh), (255,0,255), 1); cv2.putText(display_frame_render, "ErrorROI", (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255),1)

                    if info_for_display:
                        sel = info_for_display.get('sel_flag', 0); dead = info_for_display.get('is_dead', 0)
                        face = info_for_display.get('need_face', False); range_err = info_for_display.get('need_range', False)
                        no_tgt = info_for_display.get('no_target_error', False); rew = info_for_display.get('reward_this_step', 0.0)
                        act_name = info_for_display.get('action_name', "N/A"); g_prompt = info_for_display.get('agent_should_consider_g_key', False)
                        cv2.putText(display_frame_render, f"Sel:{sel} Dead:{dead} Rew:{rew:.2f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
                        cv2.putText(display_frame_render, f"FaceE:{face} RangeE:{range_err} NoTgtE:{no_tgt}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
                        cv2.putText(display_frame_render, f"LastAct: {act_name} SuggestG:{g_prompt}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
                except Exception as e: print(f"Error preparing frame for rendering: {e}"); display_frame_render = None
            if display_frame_render is None: 
                display_frame_render = np.zeros((360, 640, 3), dtype=np.uint8); cv2.putText(display_frame_render, "Frame Error or Waiting...", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            try: cv2.imshow(self.window_name, display_frame_render); cv2.waitKey(30)
            except Exception as e:
                 print(f"Error during cv2.imshow or cv2.waitKey: {e}")
                 try: cv2.destroyWindow(self.window_name); print("Force closed render window due to error.")
                 except Exception: pass
                 
    def close(self):
        if self.render_mode == "human":
            print("Closing OpenCV window.")
            try: cv2.destroyWindow(self.window_name); cv2.waitKey(10)
            except Exception as e: print(f"Error destroying window in close: {e}")
        print("WowKeyEnv closed.")