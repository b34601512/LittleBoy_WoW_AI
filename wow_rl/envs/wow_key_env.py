# D:\wow_ai\wow_rl\envs\wow_key_env.py
# (Version: Fix ValueError in RewardSensor init print, Updated LOOT_CHAT_ROI)
import gymnasium as gym
import numpy as np
import cv2
import time
from gymnasium import spaces
import traceback
import os

try:
    from wow_rl.utils.screen_grabber import ScreenGrabber
    # KEY_FUNCTIONS will be mapped via self.ACTION_TO_KEY_MAP
    from wow_rl.utils.interactor_keys import send_key 
    from wow_rl.utils.reward_sensor import RewardSensor
    from wow_rl.utils.error_sensor import ErrorMessageSensor
except ImportError as e: print(f"CRITICAL ERROR importing utils: {e}"); raise e

# --- Constants for WowKeyEnv ---
# Using your provided coordinates for LOOT_CHAT_ROI
LOOT_CHAT_ROI = (35, 669, 520, 151) # X, Y, W, H 
project_root_for_paths_static = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMPLATE_LOOT_PICKED_UP = os.path.join(project_root_for_paths_static, "data", "template_loot_picked_up.png")
TEMPLATE_LOOT_OBTAINED_ITEM = os.path.join(project_root_for_paths_static, "data", "template_loot_obtained_item.png")
MATCH_THRESHOLD_LOOT_SUCCESS = 0.75 

FRAME_WIDTH_BC, FRAME_HEIGHT_BC = 54, 96 # For behavior cloning data

class WowKeyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    ACTION_TABLE = { # O3's recommended 8-action space
        0: 'F_AttackSelect', 1: 'G_SelectLastCorpse', 2: 'Bracket_LootOrInteract',
        3: 'Turn_Left_Discrete', 4: 'Turn_Right_Discrete', 5: 'W_Forward_Tap',
        6: 'No_Op', 7: 'Tab_Switch_Fallback',
        8: 'ESC_CancelTarget' # 新增：ESC取消目标
    }
    ACTION_TO_KEY_MAP = { # Maps action ID to pydirectinput key string or None
        0: 'f', 1: 'g', 2: ']',
        3: 'left', 4: 'right', 5: 'w',
        6: None, 7: 'tab',
        8: 'esc' # 新增：ESC键
    }

    def __init__(self,
                 detector_w=r'D:\wow_ai\runs\detect\gwyx_detect_run_s_ep200_neg\weights\best.pt',
                 render_mode=None, max_steps=300,
                 roi_reward=(250, 35, 205, 64), match_thresh_select=0.5, match_thresh_dead=0.5,
                 roi_error=(800, 110, 330, 90), error_match_thresh=0.5,
                 target_class_index=0):
        super().__init__()
        print(f"Initializing WowKeyEnv (v_LootChatROI_Fix)...") # New version
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        try:
            self.grab = ScreenGrabber()
            # --- RewardSensor Initialization ---
            # We need to ensure RewardSensor's __init__ uses "is not None" for its template checks
            # Assuming RewardSensor.py has been fixed as per previous discussion.
            self.rs = RewardSensor(
                roi_select=roi_reward, 
                template_path_hostile=os.path.join(self.project_root, "data", "target_frame_template.png"),
                template_path_neutral=os.path.join(self.project_root, "data", "template_target_neutral_selected.png"),
                match_thresh_select=match_thresh_select, 
                template_path_dead=os.path.join(self.project_root, "data", "template_target_dead.png"),
                match_thresh_dead=match_thresh_dead, 
                yolo_weight=detector_w, target_idx=target_class_index
            )
            self.es = ErrorMessageSensor(roi=roi_error, threshold=error_match_thresh)
            self.template_loot_picked = self._load_loot_template(TEMPLATE_LOOT_PICKED_UP, "loot_picked")
            self.template_loot_obtained = self._load_loot_template(TEMPLATE_LOOT_OBTAINED_ITEM, "loot_obtained")
            print(f"RewardSensor: ThreshSelect={match_thresh_select}, ThreshDead={match_thresh_dead}")
            print(f"ErrorMessageSensor: ThreshError={error_match_thresh}")
            print(f"LootChatROI set to: {LOOT_CHAT_ROI}, LootMatchThresh: {MATCH_THRESHOLD_LOOT_SUCCESS}")
        except Exception as e: print(f"ERROR init utilities: {e}"); traceback.print_exc(); raise e

        self.action_space = spaces.Discrete(len(self.ACTION_TABLE))
        self.action_names = {k: v for k, v in self.ACTION_TABLE.items()}
        print(f"Action space: Discrete({len(self.ACTION_TABLE)}) -> {self.action_names}")

        self.error_keys = list(self.es.DEFAULT_TEMPLATES.keys()) 
        self.observation_space = spaces.Dict({
            "frame":   spaces.Box(low=0, high=255, shape=(FRAME_HEIGHT_BC, FRAME_WIDTH_BC, 3), dtype=np.uint8),
            "sel_flag":spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            "is_dead": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            "can_loot_target": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            "just_tried_loot": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8), 
            "errors":  spaces.MultiBinary(len(self.error_keys))
        })
        print(f"Observation space structure: {self.observation_space}")

        self.max_steps = max_steps; self.current_step = 0
        self.render_mode = render_mode; self.window_name = "WowKeyEnv (LootChatROI_Fix)"
        if self.render_mode == "human":
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            try: cv2.startWindowThread(); print("Started OpenCV window thread.")
            except Exception as e: print(f"Warning: cv2.startWindowThread() failed: {e}")

        self.last_grabbed_frame = None
        self.current_analysis = self._get_empty_analysis_extended()
        self.prev_analysis_for_step = self._get_empty_analysis_extended()
        self.just_tried_loot_flag = False
        self.prompt_esc_cancellation = False # 新增：是否提示AI按ESC
        self.just_pressed_esc = False      # 新增：上一步是否按了ESC
        print(f"WowKeyEnv (LootChatROI_Fix_ESC_Logic) initialized.")

        # 完整流程追踪变量（新增）
        self.kill_step_record = None      # 记录击杀发生的步骤编号
        self.loot_step_record = None      # 记录拾取发生的步骤编号
        self.complete_cycle_bonus_given = False  # 是否已给予完整循环奖励

    def _load_loot_template(self, template_path, name):
        if not os.path.exists(template_path): print(f"WARNING (Loot Tpl): '{name}' NOT FOUND: {template_path}"); return None
        template = cv2.imread(template_path, cv2.IMREAD_COLOR);
        if template is None: print(f"WARNING (Loot Tpl): Failed to LOAD '{name}' from {template_path}"); return None
        print(f"  Loot Template '{name}' loaded, shape: {template.shape}"); return template

    def _get_empty_analysis_extended(self):
        # error_keys 会在 __init__ 中基于 ErrorMessageSensor 更新
        # self.error_keys 在这里可能还未完全初始化，但没关系，因为 analyze_state_from_frame 会用最新的
        current_error_keys = getattr(self, 'error_keys', []) # 安全获取
        base_errors = {key: 0 for key in current_error_keys}
        
        analysis = {"sel_flag": 0, "is_dead": 0, "yolo_prob_target": 0.0,
                    "errors_dict": base_errors,
                    "can_loot_target": 0, "loot_success_this_frame": 0}
        
        # 确保所有已知的 'need_X' 键都有默认值
        for key in ['need_face', 'need_range', 'need_no_target', 'need_cant_attack_target']: # 添加 need_cant_attack_target
             if key not in analysis: # 以防万一 error_keys 初始化延迟
                analysis[key] = False
        return analysis

    def _check_loot_success_in_frame(self, frame_full):
        if frame_full is None or frame_full.size == 0: return 0
        try:
            roi_x, roi_y, roi_w, roi_h = LOOT_CHAT_ROI
            fh, fw = frame_full.shape[:2]
            y_start, y_end = max(0, roi_y), min(fh, roi_y + roi_h)
            x_start, x_end = max(0, roi_x), min(fw, roi_x + roi_w)
            if y_start >= y_end or x_start >= x_end: return 0
            crop = frame_full[y_start:y_end, x_start:x_end]
            if crop.size == 0: return 0
            # Debug: Show loot ROI
            # cv2.imshow("Loot Chat ROI For Debug", crop) 
            # cv2.waitKey(1)

            if self.template_loot_obtained is not None: # Prioritize "obtained"
                res_obtained = cv2.matchTemplate(crop, self.template_loot_obtained, cv2.TM_CCOEFF_NORMED)
                _, max_val_obtained, _, _ = cv2.minMaxLoc(res_obtained)
                # print(f"DEBUG Loot Obtained MatchVal: {max_val_obtained:.2f}")
                if max_val_obtained >= MATCH_THRESHOLD_LOOT_SUCCESS: return 1
            if self.template_loot_picked is not None:
                res_picked = cv2.matchTemplate(crop, self.template_loot_picked, cv2.TM_CCOEFF_NORMED)
                _, max_val_picked, _, _ = cv2.minMaxLoc(res_picked)
                # print(f"DEBUG Loot Picked MatchVal: {max_val_picked:.2f}")
                if max_val_picked >= MATCH_THRESHOLD_LOOT_SUCCESS: return 1
        except Exception as e: print(f"Error in _check_loot_success: {e}");
        return 0
        
    def _get_obs(self): 
        # ... (remains the same as your last working version, uses self.current_analysis) ...
        analysis = self.current_analysis
        frame_data = self.last_grabbed_frame 
        if frame_data is None or frame_data.size == 0: frame_resized = np.zeros((FRAME_HEIGHT_BC, FRAME_WIDTH_BC, 3), dtype=np.uint8)
        else:
            try: frame_resized = cv2.resize(frame_data, (FRAME_WIDTH_BC, FRAME_HEIGHT_BC), interpolation=cv2.INTER_AREA)
            except Exception: frame_resized = np.zeros((FRAME_HEIGHT_BC, FRAME_WIDTH_BC, 3), dtype=np.uint8)
        return {"frame": frame_resized,"sel_flag": np.array([analysis['sel_flag']], dtype=np.uint8),"is_dead": np.array([analysis['is_dead']], dtype=np.uint8),"can_loot_target": np.array([analysis['can_loot_target']], dtype=np.uint8),"just_tried_loot": np.array([1 if self.just_tried_loot_flag else 0], dtype=np.uint8),"errors": np.array([analysis['errors_dict'].get(key,0) for key in self.error_keys], dtype=np.int8)}


    def analyze_state_from_frame(self, frame_to_analyze):
        # ... (remains the same as your last working version) ...
        if frame_to_analyze is None or frame_to_analyze.size == 0: return self._get_empty_analysis_extended()
        analysis = self._get_empty_analysis_extended() # 这会使用最新的 error_keys
        try:
            sel_flag_raw, is_dead_raw, _ = self.rs.analyze(frame_to_analyze)
            analysis["sel_flag"] = int(sel_flag_raw); analysis["is_dead"] = int(is_dead_raw)
            
            error_flags_dict = self.es.detect(frame_to_analyze) 
            analysis["errors_dict"] = error_flags_dict
            
            # 动态生成 need_X 标志
            for err_key in self.error_keys: # self.error_keys 此时已包含 'cant_attack_target'
                analysis[f'need_{err_key}'] = error_flags_dict.get(err_key, 0) == 1
                
            # 更新 no_target_error 逻辑，确保它不与 cant_attack_target 冲突或混淆
            is_no_target = analysis.get('need_no_target', False)
            is_no_attackable_target = analysis.get('need_no_attackable_target', False)
            is_cant_attack_target = analysis.get('need_cant_attack_target', False)
            
            analysis['no_target_error'] = (is_no_target or is_no_attackable_target) and not is_cant_attack_target

            analysis["can_loot_target"] = 1 if analysis["sel_flag"] == 1 and analysis["is_dead"] == 1 else 0
            if self.just_tried_loot_flag:
                if self._check_loot_success_in_frame(frame_to_analyze): analysis["loot_success_this_frame"] = 1; print("INFO (analyze): Loot success detected in chat!")
                self.just_tried_loot_flag = False 
            return analysis
        except Exception as e: print(f"ERROR in analyze_state: {e}"); traceback.print_exc(); return analysis

    def step(self, action: int):
        self.prev_analysis_for_step = self.current_analysis.copy()
        self.just_pressed_esc = False # 重置ESC标记

        action_key_to_press = self.ACTION_TO_KEY_MAP.get(action)
        action_name = self.action_names.get(action, "Unknown")

        if action_name == 'ESC_CancelTarget':
            self.just_pressed_esc = True

        if action_key_to_press is not None:
            try: send_key(action_key_to_press, wait=0.02) 
            except Exception as e: print(f"ERROR sending key: {e}")
        
        self.just_tried_loot_flag = False 
        if action_name == 'Bracket_LootOrInteract' and \
           self.prev_analysis_for_step.get("can_loot_target",0) == 1:
            self.just_tried_loot_flag = True
            time.sleep(0.4)

        try: self.last_grabbed_frame = self.grab.grab()
        except Exception as e: print(f"ERROR grabbing screen: {e}"); self.last_grabbed_frame = None
        
        self.current_analysis = self.analyze_state_from_frame(self.last_grabbed_frame)

        reward = -0.05; terminated = False
        sel_now = self.current_analysis.get("sel_flag", 0); dead_now = self.current_analysis.get("is_dead", 0)
        no_target_err_now = self.current_analysis.get("no_target_error", False) # 会被 cant_attack_target 影响
        cant_attack_target_err_now = self.current_analysis.get("need_cant_attack_target", False)
        loot_succeeded_this_step = self.current_analysis.get("loot_success_this_frame", 0) == 1
        prev_sel = self.prev_analysis_for_step.get("sel_flag", 0); prev_dead = self.prev_analysis_for_step.get("is_dead", 0)
        
        # 优先处理ESC动作的奖励
        if self.just_pressed_esc:
            if self.prompt_esc_cancellation:
                if prev_sel == 1 and sel_now == 0 : # 成功取消了目标
                    print("INFO (step): ESC successfully cancelled prompted target. Reward +1.0")
                    reward += 1.0
                    self.prompt_esc_cancellation = False # 重置提示
                else: # 提示了但ESC未取消目标 (可能目标已自然消失或ESC无效)
                    print("INFO (step): ESC pressed after prompt, but target not cancelled or was already gone. Penalty -0.3")
                    reward -= 0.3
                    self.prompt_esc_cancellation = False # 无论如何重置提示
            else: # 无故按ESC
                print("INFO (step): ESC pressed without prompt. Penalty -0.5")
                reward -= 0.5
        elif loot_succeeded_this_step:
            print(f"INFO (step): Loot SUCCESS! Reward +5.0 (Action: {action_name})")
            reward += 5.0  # 🎯 大幅提高拾取奖励，让AI上瘾！
            # 记录拾取步骤
            self.loot_step_record = self.current_step
            
            # 检查是否完成完整击杀+拾取流程
            if (self.kill_step_record is not None and 
                not self.complete_cycle_bonus_given and
                self.current_step - self.kill_step_record <= 10):  # 击杀后10步内拾取
                print(f"INFO (step): COMPLETE KILL+LOOT CYCLE! Extra bonus +0.5")
                reward += 0.5  # 保持额外奖励不变
                self.complete_cycle_bonus_given = True
            
            terminated = True
        elif prev_sel == 1 and prev_dead == 0 and dead_now == 1: 
            print(f"INFO (step): Target KILLED! Reward +2.0 (Action: {action_name})")
            reward += 2.0  # 🎯 降低击杀奖励，击杀只是获得拾取机会的手段
            # 记录击杀步骤，重置循环状态
            self.kill_step_record = self.current_step
            self.loot_step_record = None
            self.complete_cycle_bonus_given = False
            self.prompt_esc_cancellation = False # 目标死了，不需要ESC了
        # 处理 'cant_attack_target' 错误
        elif prev_sel == 1 and (action_name == 'F_AttackSelect' or (action_name == 'Bracket_LootOrInteract' and prev_dead == 0)) and cant_attack_target_err_now:
            print(f"INFO (step): Attacked selected target but \'cant_attack_target\' error. Penalty -2.0. Action: {action_name}")
            reward -= 2.0
            self.prompt_esc_cancellation = True # 提示按ESC
        # G键逻辑 (确保在 cant_attack_target 不触发时才考虑)
        elif self.agent_should_consider_g_key and action_name == 'G_SelectLastCorpse' and not self.prompt_esc_cancellation: 
            if sel_now == 1 and dead_now == 1: 
                print("INFO (step): G-Key selected corpse. Reward +1.5")
                reward += 1.5
            else: 
                print("INFO (step): G-Key no (dead) last target. Penalty -0.3")
                reward -= 0.3
            self.agent_should_consider_g_key = False 
        elif action_name == 'G_SelectLastCorpse' and not self.agent_should_consider_g_key and not self.prompt_esc_cancellation:
            print("INFO (step): G-Key used when not recommended. Penalty -1.5")
            reward -= 1.5
        # F键简化奖惩机制（用户建议版本）
        elif action_name == 'F_AttackSelect' and not self.prompt_esc_cancellation:
            # 检查是否有任何报错信息（包含所有新增的错误类型）
            has_any_error = (
                # 原有错误类型
                self.current_analysis.get("need_face", False) or
                self.current_analysis.get("need_range", False) or  
                no_target_err_now or
                cant_attack_target_err_now or
                
                # 新增错误类型
                self.current_analysis.get("need_facing_wrong_way", False) or      # 面朝错误方向
                self.current_analysis.get("need_spell_not_ready", False) or       # 法术没准备好
                self.current_analysis.get("need_too_far_away", False) or          # 你距离太远
                self.current_analysis.get("need_player_dead", False) or           # 玩家死亡
                self.current_analysis.get("need_cannot_attack_while_dead", False) or  # 死亡状态不能攻击
                self.current_analysis.get("need_no_attackable_target", False)     # 没有可攻击目标
            )
            
            if has_any_error:
                # 收集所有错误类型用于详细报告
                error_types = []
                if self.current_analysis.get("need_face", False): error_types.append("face")
                if self.current_analysis.get("need_range", False): error_types.append("range") 
                if no_target_err_now: error_types.append("no_target")
                if cant_attack_target_err_now: error_types.append("cant_attack")
                if self.current_analysis.get("need_facing_wrong_way", False): error_types.append("facing_wrong_way")
                if self.current_analysis.get("need_spell_not_ready", False): error_types.append("spell_not_ready")
                if self.current_analysis.get("need_too_far_away", False): error_types.append("too_far_away")
                if self.current_analysis.get("need_player_dead", False): error_types.append("player_dead")
                if self.current_analysis.get("need_cannot_attack_while_dead", False): error_types.append("cannot_attack_while_dead")
                if self.current_analysis.get("need_no_attackable_target", False): error_types.append("no_attackable_target")
                
                # 有任何报错就轻微惩罚，让AI学习避免错误操作
                print(f"INFO (step): F_AttackSelect with errors: {error_types}. Penalty -0.2")
                reward -= 0.2
                
                # 如果是no_target相关错误，提示使用G键
                if no_target_err_now or self.current_analysis.get("need_no_attackable_target", False):
                    self.agent_should_consider_g_key = True
            else:
                # 无报错就给予奖励，鼓励正确使用F键
                print(f"INFO (step): F_AttackSelect successful (no errors). Reward +1.0")
                reward += 1.0
        # Bracket拾取攻击逻辑（保持原有复杂性，因为涉及击杀检测）
        elif action_name == 'Bracket_LootOrInteract' and prev_sel == 1 and prev_dead == 0 and not self.prompt_esc_cancellation:
            # 这是作为攻击使用的]键
            if sel_now == 1 and dead_now == 0:
                if self.current_analysis.get("need_face", False): reward -= 0.2; print("INFO (step): Attack with face_error.")
                elif self.current_analysis.get("need_range", False): reward -= 0.2; print("INFO (step): Attack with range_error.")
                elif no_target_err_now : reward -= 0.3; print("INFO (step): Attack on selected live target, but \'no_target_error\' or similar is active. Penalty -0.3")
                else: reward += 1.5;
            elif no_target_err_now: 
                print(f"INFO (step): {action_name} (as attack) attempt resulted in \'no target\' error. Penalty -0.3")
                reward -= 0.3
                self.agent_should_consider_g_key = True
        # 无效拾取
        elif action_name == 'Bracket_LootOrInteract' and not self.prev_analysis_for_step.get("can_loot_target",0) and not self.prompt_esc_cancellation:
             print("INFO (step): ']' pressed with no valid lootable target. Penalty -6.0")
             reward -= 6.0  # 🎯 严厉惩罚无效拾取，杜绝乱按（大于拾取奖励5.0）
        # Tab键惩罚
        elif action_name == 'Tab_Switch_Fallback' and not self.prompt_esc_cancellation:
            if prev_sel == 1: 
                print("INFO (step): Tab pressed with existing target. Penalty -2.0")
                reward -= 2.0
            else:
                print("INFO (step): Tab pressed (no existing target). Penalty -1.0")
                reward -= 1.0

        # Update G-Key prompt - 只有在不提示ESC且符合原条件时才提示G
        if not self.prompt_esc_cancellation:
            if sel_now == 0 and prev_sel == 1 and prev_dead == 0 and not dead_now and not loot_succeeded_this_step: 
                print("INFO (step): Target lost (was live), not dead. Prompt G-Key.")
                self.agent_should_consider_g_key = True
            # 如果是F或Bracket攻击后出现no_target错误，也提示G
            elif (action_name == 'F_AttackSelect' or action_name == 'Bracket_LootOrInteract') and no_target_err_now:
                 print("INFO (step): Attack with \'no target\' error. Prompt G-Key.")
                 self.agent_should_consider_g_key = True
            else: # 其他情况默认不提示G
                self.agent_should_consider_g_key = False
        else: # 如果提示ESC，则不提示G
            self.agent_should_consider_g_key = False
        
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        if truncated and not terminated : terminated = True 
        
        info = self.current_analysis.copy()
        # 添加 prompt_esc_cancellation 到 info
        info.update({'step': self.current_step, 'reward_this_step': reward, 'action_taken': action, 
                     'action_name': action_name, 
                     'agent_should_consider_g_key': self.agent_should_consider_g_key,
                     'prompt_esc_cancellation': self.prompt_esc_cancellation}) # 新增
        agent_obs = self._get_obs() 
        if self.render_mode == "human": self.render(info_for_display=info)

        # --- 动作掩码逻辑 ---
        sel_flag = self.current_analysis.get("sel_flag", 0)
        is_dead = self.current_analysis.get("is_dead", 0)
        # 使用 self.prompt_esc_cancellation 和 cant_attack_target_err_now
        
        action_mask = np.ones(self.action_space.n, dtype=np.int8) 
        
        action_idx_loot = -1; action_idx_attack = -1; action_idx_tab = -1; action_idx_g_key = -1; action_idx_esc = -1

        for idx, name_map in self.ACTION_TABLE.items():
            if name_map == 'Bracket_LootOrInteract': action_idx_loot = idx
            elif name_map == 'F_AttackSelect': action_idx_attack = idx
            elif name_map == 'Tab_Switch_Fallback': action_idx_tab = idx
            elif name_map == 'G_SelectLastCorpse': action_idx_g_key = idx
            elif name_map == 'ESC_CancelTarget': action_idx_esc = idx # 获取ESC动作索引
        
        # 主要逻辑：如果提示按ESC (因为cant_attack_target错误)
        if self.prompt_esc_cancellation and sel_flag == 1: # 必须有目标才需要ESC
            for i in range(self.action_space.n): # 禁用所有其他动作
                if i != action_idx_esc and i != self.ACTION_TABLE.get('No_Op'): # 允许ESC和NoOp
                    action_mask[i] = 0
            if action_idx_esc != -1: action_mask[action_idx_esc] = 1 # 确保ESC可用
            print(f"DEBUG MASK: Prompting ESC. Mask: {action_mask}")
        else: # 正常情况下的掩码逻辑
            # 最严格的拾取掩码：只有选中且死亡才允许拾取（按ChatGPT建议）
            if action_idx_loot != -1:
                if not (sel_flag == 1 and is_dead == 1):  # 必须选中且死亡
                    action_mask[action_idx_loot] = 0
            
            # 尸体阶段禁止攻击和TAB
            if action_idx_attack != -1:
                if is_dead == 1: # 尸体阶段禁止攻击
                    action_mask[action_idx_attack] = 0
            
            # Tab键完全禁用（已证实会导致错误行为）
            if action_idx_tab != -1:
                action_mask[action_idx_tab] = 0

            # G键的掩码：只有在 agent_should_consider_g_key 为True时才启用
            if action_idx_g_key != -1 and not self.agent_should_consider_g_key:
                 action_mask[action_idx_g_key] = 0
            
            # ESC键在非prompt状态下禁用
            if action_idx_esc != -1 and not self.prompt_esc_cancellation:
                 action_mask[action_idx_esc] = 0

        info['action_mask'] = action_mask
        # --- 动作掩码逻辑结束 ---

        return agent_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None): # reset logic mostly same
        super().reset(seed=seed); self.current_step = 0; print("Environment reset."); time.sleep(0.2)
        try: self.last_grabbed_frame = self.grab.grab()
        except Exception as e: print(f"ERROR grabbing screen in reset: {e}"); self.last_grabbed_frame = None
        self.current_analysis = self.analyze_state_from_frame(self.last_grabbed_frame)
        self.prev_analysis_for_step = self.current_analysis.copy() if self.current_analysis else self._get_empty_analysis_extended()
        observation = self._get_obs()
        info = self.current_analysis.copy() if self.current_analysis else self._get_empty_analysis_extended()
        info.update({'step': self.current_step, 'reward_this_step': 0.0, 'action_taken': -1, 'action_name': "N/A", 'agent_should_consider_g_key': False})
        self.agent_should_consider_g_key = False
        self.prompt_esc_cancellation = False # 重置ESC提示
        self.just_pressed_esc = False      # 重置ESC标记
        # 重置完整流程追踪变量
        self.kill_step_record = None
        self.loot_step_record = None
        self.complete_cycle_bonus_given = False
        if self.render_mode == "human": self.render(info_for_display=info)
        return observation, info

    def render(self, info_for_display=None): # render logic mostly same
         if self.render_mode == "human":
            display_frame_render = None
            if self.last_grabbed_frame is not None and self.last_grabbed_frame.size > 0:
                try:
                    vis_frame = self.last_grabbed_frame.copy()
                    display_frame_render = cv2.resize(vis_frame, (640, 360), interpolation=cv2.INTER_AREA)
                    # Draw ROIs (Target, Error, Loot Chat)
                    rx,ry,rw,rh = self.rs.roi_select_x, self.rs.roi_select_y, self.rs.roi_select_w, self.rs.roi_select_h; cv2.rectangle(display_frame_render, (rx,ry), (rx+rw, ry+rh), (0,255,255), 1); cv2.putText(display_frame_render, "TargetROI", (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255),1)
                    ex,ey,ew,eh = self.es.x, self.es.y, self.es.w, self.es.h; cv2.rectangle(display_frame_render, (ex,ey), (ex+ew, ey+eh), (255,0,255), 1); cv2.putText(display_frame_render, "ErrorROI", (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255),1)
                    lcrx,lcry,lcrw,lcrh = LOOT_CHAT_ROI; cv2.rectangle(display_frame_render, (lcrx,lcry), (lcrx+lcrw,lcry+lcrh), (255,255,0), 1); cv2.putText(display_frame_render, "LootChatROI", (lcrx, lcry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0),1)
                    if info_for_display: # Display dynamic info
                        sel=info_for_display.get('sel_flag',0); dead=info_for_display.get('is_dead',0); can_loot=info_for_display.get('can_loot_target',0); looted=info_for_display.get('loot_success_this_frame',0); rew=info_for_display.get('reward_this_step',0.0)
                        
                        # 错误信息显示
                        face=info_for_display.get('need_face',False); range_e=info_for_display.get('need_range',False); no_tgt=info_for_display.get('no_target_error',False)
                        cant_atk=info_for_display.get('need_cant_attack_target', False)
                        facing_wrong=info_for_display.get('need_facing_wrong_way', False)
                        spell_not_ready=info_for_display.get('need_spell_not_ready', False)
                        too_far=info_for_display.get('need_too_far_away', False)
                        player_dead=info_for_display.get('need_player_dead', False)
                        cannot_attack_dead=info_for_display.get('need_cannot_attack_while_dead', False)
                        no_attackable=info_for_display.get('need_no_attackable_target', False)
                        
                        prompt_esc=info_for_display.get('prompt_esc_cancellation', False)
                        act_n=info_for_display.get('action_name',"N/A"); g_p=info_for_display.get('agent_should_consider_g_key',False); just_tried_l = 1 if self.just_tried_loot_flag else 0
                        
                        # 第一行：基本状态
                        cv2.putText(display_frame_render, f"Sel:{sel} Dead:{dead} CanLoot:{can_loot} LootOK:{looted} Rew:{rew:.2f}", (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1)
                        
                        # 第二行：原有错误
                        cv2.putText(display_frame_render, f"FaceE:{face} RangeE:{range_e} NoTgtE:{no_tgt} CantAtkE:{cant_atk}", (5,35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1)
                        
                        # 第三行：新增错误1
                        cv2.putText(display_frame_render, f"FacingWrongE:{facing_wrong} SpellNotReadyE:{spell_not_ready} TooFarE:{too_far}", (5,50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0),1)
                        
                        # 第四行：新增错误2
                        cv2.putText(display_frame_render, f"PlayerDeadE:{player_dead} CannotAtkDeadE:{cannot_attack_dead} NoAtkTargetE:{no_attackable}", (5,65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0),1)
                        
                        # 第五行：动作信息
                        cv2.putText(display_frame_render, f"Act:{act_n} SuggestG:{g_p} SuggestESC:{prompt_esc} TriedLoot:{just_tried_l}", (5,80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1)
                except Exception as e: print(f"Error rendering: {e}"); display_frame_render = None
            if display_frame_render is None: display_frame_render = np.zeros((360, 640, 3), dtype=np.uint8); cv2.putText(display_frame_render, "FrameErr", (50,180), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            try: cv2.imshow(self.window_name, display_frame_render); cv2.waitKey(1)
            except Exception as e: print(f"Error imshow/waitKey: {e}")
                 
    def close(self): # close method remains the same
        if self.render_mode == "human": print("Closing OpenCV window."); 
        try: cv2.destroyAllWindows(); cv2.waitKey(10)
        except Exception as e: print(f"Error destroying window: {e}")
        print("WowKeyEnv closed.")