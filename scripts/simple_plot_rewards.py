# 小男孩项目简化训练曲线查看脚本
# 避免显示窗口，直接保存图片
import os
import sys
import matplotlib
matplotlib.use('Agg')  # 避免显示窗口
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 动态路径设置
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.dirname(script_dir)

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    print("✅ TensorBoard库导入成功")
except ImportError:
    print("❌ 错误: 无法导入TensorBoard库，请安装: pip install tensorboard")
    sys.exit(1)

# 配置参数
LOG_DIR = os.path.join(project_root, "runs", "ppo_logs_m1_stage")
SAVE_DIR = os.path.join(project_root, "runs", "training_curves")

def extract_latest_training_data():
    """提取最新训练数据"""
    if not os.path.exists(LOG_DIR):
        print(f"❌ 日志目录不存在: {LOG_DIR}")
        return None
    
    # 找到最新的训练目录
    subdirs = [d for d in os.listdir(LOG_DIR) if os.path.isdir(os.path.join(LOG_DIR, d))]
    if not subdirs:
        print("❌ 没有找到训练目录")
        return None
    
    latest_dir = max(subdirs, key=lambda x: os.path.getmtime(os.path.join(LOG_DIR, x)))
    latest_path = os.path.join(LOG_DIR, latest_dir)
    print(f"📊 分析最新训练: {latest_dir}")
    
    # 加载TensorBoard数据
    ea = EventAccumulator(latest_path)
    ea.Reload()
    
    # 提取标量数据
    scalars = ea.Tags()['scalars']
    print(f"📈 可用指标: {scalars}")
    
    data = {}
    for tag in scalars:
        if 'ep_rew_mean' in tag or 'ep_len_mean' in tag or 'loss' in tag:
            scalar_events = ea.Scalars(tag)
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            data[tag] = {'steps': steps, 'values': values}
    
    return data, latest_dir

def plot_training_curves(data, run_name):
    """生成训练曲线图"""
    if not data:
        print("❌ 没有数据可绘制")
        return
    
    # 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'小男孩项目训练进展 - {run_name}', fontsize=16, fontweight='bold')
    
    # 1. 平均奖励曲线
    if any('ep_rew_mean' in tag for tag in data.keys()):
        reward_tag = next(tag for tag in data.keys() if 'ep_rew_mean' in tag)
        steps = data[reward_tag]['steps']
        values = data[reward_tag]['values']
        
        axes[0,0].plot(steps, values, 'b-', linewidth=2, alpha=0.7)
        if len(values) > 10:
            # 添加趋势线
            z = np.polyfit(steps, values, 1)
            trend_line = np.poly1d(z)
            axes[0,0].plot(steps, trend_line(steps), 'r--', alpha=0.8, label='趋势线')
        
        axes[0,0].set_title('🏆 平均奖励变化', fontsize=12, fontweight='bold')
        axes[0,0].set_xlabel('训练步数')
        axes[0,0].set_ylabel('平均奖励')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        # 显示最新数值
        if values:
            latest_reward = values[-1]
            axes[0,0].text(0.02, 0.98, f'最新奖励: {latest_reward:.2f}', 
                          transform=axes[0,0].transAxes, fontsize=10,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                          verticalalignment='top')
    
    # 2. 平均回合长度
    if any('ep_len_mean' in tag for tag in data.keys()):
        length_tag = next(tag for tag in data.keys() if 'ep_len_mean' in tag)
        steps = data[length_tag]['steps']
        values = data[length_tag]['values']
        
        axes[0,1].plot(steps, values, 'g-', linewidth=2, alpha=0.7)
        axes[0,1].set_title('📏 平均回合长度', fontsize=12, fontweight='bold')
        axes[0,1].set_xlabel('训练步数')
        axes[0,1].set_ylabel('平均步数')
        axes[0,1].grid(True, alpha=0.3)
        
        if values:
            latest_length = values[-1]
            axes[0,1].text(0.02, 0.98, f'最新长度: {latest_length:.0f}', 
                          transform=axes[0,1].transAxes, fontsize=10,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                          verticalalignment='top')
    
    # 3. 损失函数
    loss_tags = [tag for tag in data.keys() if 'loss' in tag.lower()]
    if loss_tags:
        for i, tag in enumerate(loss_tags[:2]):  # 最多显示2个损失
            steps = data[tag]['steps']
            values = data[tag]['values']
            color = ['orange', 'purple'][i]
            axes[1,i].plot(steps, values, color=color, linewidth=2, alpha=0.7)
            axes[1,i].set_title(f'📉 {tag}', fontsize=12, fontweight='bold')
            axes[1,i].set_xlabel('训练步数')
            axes[1,i].set_ylabel('损失值')
            axes[1,i].grid(True, alpha=0.3)
            
            if values:
                latest_loss = values[-1]
                axes[1,i].text(0.02, 0.98, f'最新: {latest_loss:.4f}', 
                              transform=axes[1,i].transAxes, fontsize=10,
                              bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                              verticalalignment='top')
    
    # 如果没有足够的数据，隐藏空白子图
    for i in range(2):
        for j in range(2):
            if not axes[i,j].has_data():
                axes[i,j].set_visible(False)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(SAVE_DIR, f'training_progress_{run_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 训练曲线已保存到: {save_path}")
    
    return save_path

def main():
    print("🚀 开始分析小男孩训练进展...")
    
    # 提取数据
    result = extract_latest_training_data()
    if result is None:
        return
    
    data, run_name = result
    
    # 生成图表
    save_path = plot_training_curves(data, run_name)
    
    # 输出摘要
    print("\n📊 训练摘要:")
    for tag, info in data.items():
        if info['values']:
            latest = info['values'][-1]
            print(f"  {tag}: {latest:.4f}")
    
    print(f"\n🎯 查看完整曲线图: {save_path}")
    print("💡 提示: 图片已保存，可以用图片查看器打开")

if __name__ == "__main__":
    main() 