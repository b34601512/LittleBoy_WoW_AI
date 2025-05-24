# 小男孩项目实时在线学习脚本
# F8开始录制 → F9暂停/恢复 → F10保存并更新模型
import os
import sys
import time
import pickle
import numpy as np
import torch
import keyboard
import threading
from datetime import datetime
from collections import deque
import cv2

# 动态路径设置
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入项目模块
from wow_rl.envs.wow_key_env import WowKeyEnv
from wow_rl.utils.grabber import ScreenGrabber

# 配置参数
ONLINE_DATA_DIR = os.path.join(project_root, "data", "online_learning")
BC_MODEL_PATH = os.path.join(project_root, "runs", "bc_models", "bc_policy_simple.pth")
BUFFER_SIZE = 1000  # 最大缓存样本数
SAVE_INTERVAL = 300  # 5分钟自动保存一次

# 键盘到动作的映射
KEY_TO_ACTION = {
    'f': 0,         # F_AttackSelect
    'g': 1,         # G_SelectLastCorpse  
    ']': 2,         # Bracket_LootOrInteract
    'left': 3,      # Turn_Left_Discrete
    'right': 4,     # Turn_Right_Discrete
    'w': 5,         # W_Forward_Tap
    'tab': 7,       # Tab_Switch_Fallback (虽然被禁用，但记录用户意图)
    'esc': 8,       # ESC_CancelTarget
}

class OnlineLearner:
    def __init__(self):
        self.recording = False
        self.buffer = deque(maxlen=BUFFER_SIZE)
        self.env_utils = None
        self.last_save_time = time.time()
        
        # 创建必要目录
        os.makedirs(ONLINE_DATA_DIR, exist_ok=True)
        
        # 初始化环境工具
        self.init_env_utils()
        
    def init_env_utils(self):
        """初始化环境分析工具"""
        try:
            # 创建一个WowKeyEnv实例用于分析状态
            self.env_utils = WowKeyEnv(render_mode=None, max_steps=1000)
            print("✅ 环境分析工具初始化成功")
        except Exception as e:
            print(f"❌ 环境工具初始化失败: {e}")
            
    def start_recording(self):
        """开始录制用户操作"""
        if self.recording:
            print("⚠️ 已经在录制中...")
            return
        
        self.recording = True
        print("🎬 开始录制用户操作...")
        print("📝 操作指南:")
        print("  F8: 开始/恢复录制")  
        print("  F9: 暂停录制")
        print("  F10: 停止录制并保存数据")
        print("💡 现在请正常游玩，您的操作将被记录用于训练!")
        
        # 启动录制线程
        self.record_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.record_thread.start()
        
    def _recording_loop(self):
        """录制主循环"""
        last_capture_time = time.time()
        
        while self.recording:
            try:
                current_time = time.time()
                
                # 捕获屏幕和状态
                if current_time - last_capture_time >= 0.1:  # 每100ms采样一次
                    frame = self.env_utils.grab.grab()
                    if frame is not None and frame.size > 0:
                        # 分析当前状态
                        state_dict = self.env_utils.analyze_state_from_frame(frame)
                        
                        # 检查是否有按键操作
                        action_id = self._detect_current_action()
                        
                        if action_id is not None and state_dict:
                            # 记录操作
                            frame_resized = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
                            self.buffer.append((frame_resized, state_dict, action_id))
                            
                            action_name = list(KEY_TO_ACTION.keys())[list(KEY_TO_ACTION.values()).index(action_id)]
                            print(f"📦 记录操作 {len(self.buffer)}: {action_name} (动作ID: {action_id})")
                    
                    last_capture_time = current_time
                
                # 定期自动保存
                if current_time - self.last_save_time >= SAVE_INTERVAL:
                    self._auto_save()
                    self.last_save_time = current_time
                
                time.sleep(0.05)  # 避免过度占用CPU
                
            except Exception as e:
                print(f"❌ 录制过程出错: {e}")
                time.sleep(0.5)
    
    def _detect_current_action(self):
        """检测当前的用户操作"""
        # 这是一个简化版本，实际应该监听键盘事件
        # 为了演示，这里返回None，实际需要键盘监听
        return None
    
    def _auto_save(self):
        """自动保存当前缓存"""
        if len(self.buffer) == 0:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(ONLINE_DATA_DIR, f"online_demo_{timestamp}.pkl")
        
        try:
            # 转换缓存为标准格式
            demo_data = list(self.buffer)
            
            with open(save_path, 'wb') as f:
                pickle.dump(demo_data, f)
            
            print(f"💾 自动保存 {len(demo_data)} 个样本到: {save_path}")
            
        except Exception as e:
            print(f"❌ 自动保存失败: {e}")
    
    def stop_recording(self):
        """停止录制并保存"""
        if not self.recording:
            print("⚠️ 没有正在进行的录制")
            return
        
        self.recording = False
        print("⏹️ 停止录制...")
        
        # 最终保存
        if len(self.buffer) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_save_path = os.path.join(ONLINE_DATA_DIR, f"final_demo_{timestamp}.pkl")
            
            try:
                demo_data = list(self.buffer)
                with open(final_save_path, 'wb') as f:
                    pickle.dump(demo_data, f)
                
                print(f"✅ 最终保存完成: {final_save_path}")
                print(f"📊 总共录制了 {len(demo_data)} 个操作样本")
                
                # 可以在这里触发模型更新
                self._update_model_if_needed()
                
            except Exception as e:
                print(f"❌ 最终保存失败: {e}")
        
        # 清理
        if self.env_utils:
            self.env_utils.close()
    
    def _update_model_if_needed(self):
        """如果有足够数据，更新BC模型"""
        if len(self.buffer) >= 100:  # 至少100个样本才更新
            print("🔄 样本数量足够，可以考虑更新BC模型...")
            print("💡 提示: 运行 python scripts/bc_train_simple.py 来更新模型")
        else:
            print("📈 样本数量还不够，继续收集更多数据...")

def main():
    """主函数"""
    print("🚀 小男孩项目在线学习系统启动!")
    print("="*50)
    
    learner = OnlineLearner()
    
    print("⌨️ 控制键:")
    print("  F8: 开始录制用户操作")
    print("  F9: 暂停/恢复录制") 
    print("  F10: 停止录制并保存数据")
    print("  Ctrl+C: 退出程序")
    print("="*50)
    
    # 设置键盘监听
    def on_f8():
        if not learner.recording:
            learner.start_recording()
        else:
            print("⚠️ 已经在录制中...")
    
    def on_f9():
        if learner.recording:
            learner.recording = False
            print("⏸️ 录制已暂停，按F8恢复")
        else:
            print("⚠️ 当前没有录制任务")
    
    def on_f10():
        learner.stop_recording()
        print("👋 程序即将退出...")
        exit(0)
    
    # 注册热键
    keyboard.add_hotkey('f8', on_f8)
    keyboard.add_hotkey('f9', on_f9) 
    keyboard.add_hotkey('f10', on_f10)
    
    try:
        print("✅ 在线学习系统就绪! 按F8开始录制您的游戏操作...")
        keyboard.wait()  # 等待键盘事件
    except KeyboardInterrupt:
        print("\n👋 用户中断，退出程序...")
        learner.stop_recording()

if __name__ == "__main__":
    main() 