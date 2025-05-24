# D:\wow_ai\scripts\plot_training_curves.py
"""
小男孩项目训练曲线可视化脚本
用于实时监控PPO训练进展和学习状态
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
import glob

# 动态路径设置
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"项目根目录: {project_root}")
except NameError:
    project_root = None

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    print("TensorBoard数据处理库导入成功")
except ImportError:
    print("错误: 无法导入TensorBoard库，请安装: pip install tensorboard")
    sys.exit(1)

# 配置参数
LOG_DIR = os.path.join(project_root, "runs", "ppo_logs_m1_stage")
PLOT_SAVE_DIR = os.path.join(project_root, "runs", "training_curves")
REFRESH_INTERVAL = 30  # 每30秒刷新一次

class TrainingCurveMonitor:
    def __init__(self, log_dir, save_dir, auto_refresh=True):
        self.log_dir = Path(log_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.auto_refresh = auto_refresh
        
        # 创建图形
        plt.style.use('seaborn-v0_8')  # 使用更现代的样式
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('小男孩项目 - PPO训练监控', fontsize=16, fontweight='bold')
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def find_latest_run(self):
        """查找最新的训练运行目录"""
        if not self.log_dir.exists():
            print(f"日志目录不存在: {self.log_dir}")
            return None
            
        # 查找所有PPO训练目录
        run_dirs = list(self.log_dir.glob("PPO_*"))
        if not run_dirs:
            print(f"未找到PPO训练日志: {self.log_dir}")
            return None
            
        # 返回最新的目录
        latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
        print(f"找到最新训练运行: {latest_run}")
        return latest_run
        
    def load_tensorboard_data(self, run_dir):
        """从TensorBoard日志加载数据"""
        try:
            ea = EventAccumulator(str(run_dir))
            ea.Reload()
            
            # 获取可用的标量数据
            scalar_tags = ea.Tags()['scalars']
            print(f"可用数据标签: {scalar_tags}")
            
            data = {}
            
            # 加载平均奖励数据
            if 'rollout/ep_rew_mean' in scalar_tags:
                reward_data = ea.Scalars('rollout/ep_rew_mean')
                data['reward_mean'] = {
                    'steps': [x.step for x in reward_data],
                    'values': [x.value for x in reward_data],
                    'wall_time': [x.wall_time for x in reward_data]
                }
                
            # 加载回合长度数据
            if 'rollout/ep_len_mean' in scalar_tags:
                length_data = ea.Scalars('rollout/ep_len_mean')
                data['length_mean'] = {
                    'steps': [x.step for x in length_data],
                    'values': [x.value for x in length_data]
                }
                
            # 加载损失数据
            if 'train/policy_loss' in scalar_tags:
                policy_loss = ea.Scalars('train/policy_loss')
                data['policy_loss'] = {
                    'steps': [x.step for x in policy_loss],
                    'values': [x.value for x in policy_loss]
                }
                
            if 'train/value_loss' in scalar_tags:
                value_loss = ea.Scalars('train/value_loss')
                data['value_loss'] = {
                    'steps': [x.step for x in value_loss],
                    'values': [x.value for x in value_loss]
                }
                
            # 加载学习率数据
            if 'train/learning_rate' in scalar_tags:
                lr_data = ea.Scalars('train/learning_rate')
                data['learning_rate'] = {
                    'steps': [x.step for x in lr_data],
                    'values': [x.value for x in lr_data]
                }
                
            return data
            
        except Exception as e:
            print(f"加载TensorBoard数据时出错: {e}")
            return None
            
    def plot_curves(self, data):
        """绘制训练曲线"""
        # 清除之前的图形
        for ax in self.axes.flat:
            ax.clear()
            
        # 1. 平均奖励曲线 (最重要)
        ax1 = self.axes[0, 0]
        if 'reward_mean' in data:
            steps = data['reward_mean']['steps']
            values = data['reward_mean']['values']
            ax1.plot(steps, values, 'b-', linewidth=2, label='平均奖励')
            ax1.set_title('平均奖励变化', fontweight='bold')
            ax1.set_xlabel('训练步数')
            ax1.set_ylabel('平均奖励')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 添加平滑趋势线
            if len(values) > 10:
                window = min(len(values) // 10, 50)
                smooth_values = np.convolve(values, np.ones(window)/window, mode='valid')
                smooth_steps = steps[window-1:]
                ax1.plot(smooth_steps, smooth_values, 'r--', linewidth=2, label=f'趋势线(窗口={window})')
                ax1.legend()
                
            # 显示最新数值
            if values:
                latest_reward = values[-1]
                ax1.text(0.02, 0.98, f'最新奖励: {latest_reward:.3f}', 
                        transform=ax1.transAxes, fontsize=12, fontweight='bold',
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        else:
            ax1.text(0.5, 0.5, '暂无奖励数据', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('平均奖励变化')
            
        # 2. 回合长度
        ax2 = self.axes[0, 1]
        if 'length_mean' in data:
            ax2.plot(data['length_mean']['steps'], data['length_mean']['values'], 'g-', linewidth=2)
            ax2.set_title('平均回合长度')
            ax2.set_xlabel('训练步数')
            ax2.set_ylabel('平均步数')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, '暂无回合长度数据', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('平均回合长度')
            
        # 3. 策略损失
        ax3 = self.axes[1, 0]
        if 'policy_loss' in data:
            ax3.plot(data['policy_loss']['steps'], data['policy_loss']['values'], 'r-', linewidth=2)
            ax3.set_title('策略损失')
            ax3.set_xlabel('训练步数')
            ax3.set_ylabel('损失值')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '暂无策略损失数据', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('策略损失')
            
        # 4. 价值损失
        ax4 = self.axes[1, 1]
        if 'value_loss' in data:
            ax4.plot(data['value_loss']['steps'], data['value_loss']['values'], 'm-', linewidth=2)
            ax4.set_title('价值损失')
            ax4.set_xlabel('训练步数')
            ax4.set_ylabel('损失值')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, '暂无价值损失数据', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('价值损失')
            
        plt.tight_layout()
        
        # 保存图片
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = self.save_dir / f"training_curves_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"曲线图已保存: {save_path}")
        
    def run_monitoring(self):
        """运行监控循环"""
        print(f"开始监控训练进度...")
        print(f"日志目录: {self.log_dir}")
        print(f"图片保存目录: {self.save_dir}")
        print(f"刷新间隔: {REFRESH_INTERVAL}秒")
        print("按 Ctrl+C 停止监控")
        
        try:
            while True:
                # 查找最新运行
                latest_run = self.find_latest_run()
                if latest_run:
                    # 加载数据
                    data = self.load_tensorboard_data(latest_run)
                    if data:
                        # 绘制曲线
                        self.plot_curves(data)
                        
                        # 显示图形
                        if self.auto_refresh:
                            plt.pause(0.1)  # 非阻塞显示
                        else:
                            plt.show()
                            break
                    else:
                        print("无法加载训练数据，等待下次刷新...")
                else:
                    print("未找到训练日志，等待训练开始...")
                    
                if not self.auto_refresh:
                    break
                    
                # 等待下次刷新
                time.sleep(REFRESH_INTERVAL)
                
        except KeyboardInterrupt:
            print("\n监控已停止")
        finally:
            plt.close('all')

def main():
    """主函数"""
    print("=" * 60)
    print("         小男孩项目 - 训练曲线监控工具")
    print("=" * 60)
    
    # 检查日志目录
    if not os.path.exists(LOG_DIR):
        print(f"错误: 日志目录不存在: {LOG_DIR}")
        print("请先开始PPO训练以生成日志文件")
        return
        
    # 创建监控器
    monitor = TrainingCurveMonitor(LOG_DIR, PLOT_SAVE_DIR, auto_refresh=True)
    
    # 运行监控
    monitor.run_monitoring()

if __name__ == "__main__":
    main() 