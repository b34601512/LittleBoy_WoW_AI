# D:\wow_ai\scripts\online_bc_update.py
# 在线行为克隆更新脚本：边玩边学习
import os
import sys
import time
import torch
import numpy as np
import keyboard
import threading
from datetime import datetime
from collections import deque
import cv2  # 用于图像处理

# --- 动态路径设置 ---
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir) # wow_ai
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    project_root = None
# --- 路径设置结束 ---

# 导入必要的库
try:
    from torch.utils.data import DataLoader, TensorDataset
    from wow_rl.envs.wow_key_env import WowKeyEnv
    from wow_rl.utils.grabber import ScreenGrabber
    from wow_rl.models.bc_simple import SimpleBCNetwork
    import win32api  # 用于获取键盘状态
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所有必要的库: torch, keyboard, win32api, opencv-python")
    sys.exit(1)

# --- 配置 ---
# BC模型路径
BC_MODEL_PATH = os.path.join(project_root, "runs", "bc_models", "bc_policy_simple.pth")
# 新数据保存路径
NEW_DATA_PATH = os.path.join(project_root, "data", "online_demos")
# 键盘映射
ACTION_TABLE = { # O3's recommended 8-action space
    0: 'F_AttackSelect', 1: 'G_SelectLastCorpse', 2: 'Bracket_LootOrInteract',
    3: 'Turn_Left_Discrete', 4: 'Turn_Right_Discrete', 5: 'W_Forward_Tap',
    6: 'No_Op', 7: 'Tab_Switch_Fallback'
}
KEY_TO_ACTION_MAP = {
    'f': 0, 'g': 1, ']': 2, 'left': 3, 'right': 4, 'w': 5, '': 6, 'tab': 7
}
# 创建反向映射
ACTION_TO_KEY_MAP = {v: k for k, v in KEY_TO_ACTION_MAP.items()}
# --- 配置结束 ---

class OnlineBCUpdater:
    def __init__(self):
        """初始化在线BC更新器"""
        self.buffer = deque(maxlen=1000)  # 存储最近1000帧
        self.recording = False
        self.model = None
        self.grabber = ScreenGrabber(region=(0, 0, 1920, 1080))
        self.create_dirs()
        self.load_model()
        
        # 记录当前按下的键
        self.current_key = ''
        self.last_update_time = time.time()
        self.update_interval = 300  # 5分钟更新一次
        
    def create_dirs(self):
        """创建必要的目录"""
        os.makedirs(NEW_DATA_PATH, exist_ok=True)
        
    def load_model(self):
        """加载BC模型"""
        if not os.path.exists(BC_MODEL_PATH):
            print(f"错误: 未找到BC模型文件: {BC_MODEL_PATH}")
            return False
            
        try:
            # 创建模型实例
            self.model = SimpleBCNetwork(input_dims=(3, 96, 54), n_actions=8)
            # 加载权重
            self.model.load_state_dict(torch.load(BC_MODEL_PATH))
            self.model.eval()  # 设置为评估模式
            print(f"已加载BC模型: {BC_MODEL_PATH}")
            return True
        except Exception as e:
            print(f"加载模型时出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start_recording(self):
        """开始记录玩家操作"""
        if self.recording:
            print("已经在记录中...")
            return
            
        self.recording = True
        print("开始记录玩家操作。按F9暂停，按F10保存并更新模型。")
        
        # 启动记录线程
        self.record_thread = threading.Thread(target=self._record_loop)
        self.record_thread.daemon = True
        self.record_thread.start()
    
    def stop_recording(self):
        """停止记录"""
        if not self.recording:
            print("没有正在进行的记录...")
            return
            
        self.recording = False
        print("停止记录。")
    
    def _record_loop(self):
        """记录循环，捕获屏幕和按键"""
        try:
            while self.recording:
                # 捕获屏幕
                frame = self.grabber.grab()
                if frame is None:
                    time.sleep(0.05)
                    continue
                
                # 调整大小以匹配模型输入
                frame_resized = cv2.resize(frame, (54, 96), interpolation=cv2.INTER_AREA)
                
                # 检测按下的键
                action = self._get_current_action()
                
                # 保存到缓冲区
                if action is not None:
                    self.buffer.append((frame_resized, action))
                
                # 检查是否需要更新模型
                current_time = time.time()
                if current_time - self.last_update_time >= self.update_interval:
                    print(f"自动更新时间到 ({self.update_interval}秒)，更新模型...")
                    self.update_model()
                    self.last_update_time = current_time
                
                time.sleep(0.05)  # 20 FPS
        except Exception as e:
            print(f"记录过程中出错: {e}")
            import traceback
            traceback.print_exc()
            self.recording = False
    
    def _get_current_action(self):
        """获取当前按下的键对应的动作ID"""
        # 检查常用按键
        for key, action_id in KEY_TO_ACTION_MAP.items():
            if key and keyboard.is_pressed(key):
                return action_id
        
        # 如果没有按键，返回No_Op
        return 6  # No_Op
    
    def update_model(self):
        """使用收集的数据更新模型"""
        if len(self.buffer) == 0:
            print("缓冲区为空，没有数据用于更新...")
            return
            
        print(f"使用 {len(self.buffer)} 个样本更新模型...")
        
        try:
            # 准备数据
            frames = []
            actions = []
            
            for frame, action in self.buffer:
                frames.append(frame)
                actions.append(action)
            
            # 转换为张量
            X = torch.tensor(np.array(frames), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
            y = torch.tensor(actions, dtype=torch.long)
            
            # 创建数据集和加载器
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # 设置模型为训练模式
            self.model.train()
            
            # 创建优化器
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
            criterion = torch.nn.CrossEntropyLoss()
            
            # 训练一个epoch
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # 保存更新后的模型
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(NEW_DATA_PATH, f"bc_model_update_{timestamp}.pth")
            torch.save(self.model.state_dict(), save_path)
            
            # 更新原始模型文件
            torch.save(self.model.state_dict(), BC_MODEL_PATH)
            
            print(f"模型更新完成并保存到: {BC_MODEL_PATH}")
            print(f"备份保存到: {save_path}")
            
            # 重置为评估模式
            self.model.eval()
            
        except Exception as e:
            print(f"更新模型时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def save_buffer_to_file(self):
        """将缓冲区数据保存到文件"""
        if len(self.buffer) == 0:
            print("缓冲区为空，没有数据可保存...")
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(NEW_DATA_PATH, f"online_demo_{timestamp}.npz")
            
            # 准备数据
            frames = []
            actions = []
            
            for frame, action in self.buffer:
                frames.append(frame)
                actions.append(action)
            
            # 保存为npz文件
            np.savez_compressed(
                save_path,
                frames=np.array(frames),
                actions=np.array(actions)
            )
            
            print(f"已保存 {len(self.buffer)} 个样本到: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"保存数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数"""
    print("=" * 60)
    print(" 魔兽世界AI代理 - 在线BC学习 ")
    print("=" * 60)
    print("本脚本在游戏中记录玩家的操作，并实时更新BC模型。")
    print("\n主要功能键:")
    print("  F8:  开始记录玩家操作")
    print("  F9:  暂停记录")
    print("  F10: 保存数据并更新模型")
    print("  Esc: 退出程序")
    print("-" * 60)
    
    updater = OnlineBCUpdater()
    
    # 设置键盘钩子
    def on_key_press(e):
        key_name = e.name.lower()
        if key_name == 'f8':
            updater.start_recording()
        elif key_name == 'f9':
            updater.stop_recording()
        elif key_name == 'f10':
            updater.save_buffer_to_file()
            updater.update_model()
        elif key_name == 'esc':
            print("退出程序...")
            sys.exit(0)
    
    keyboard.on_press(on_key_press)
    
    print("在线BC学习器已启动。按F8开始记录，按Esc退出。")
    
    # 保持程序运行
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("程序被中断")
    finally:
        # 清理资源
        keyboard.unhook_all()
        if updater.recording:
            updater.stop_recording()
        print("程序已退出")

if __name__ == "__main__":
    main() 