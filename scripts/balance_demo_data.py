# D:\wow_ai\scripts\balance_demo_data.py
# 均衡演示数据，增加稀有动作（G和]键）的样本数量
import pickle
import os
import sys
import random
import numpy as np
from collections import Counter

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

# --- 配置 ---
# 输入文件
INPUT_FILENAME = "wow_paladin_demo_v1.pkl"
INPUT_FILE_PATH = os.path.join(project_root, "data", INPUT_FILENAME)

# 输出文件
OUTPUT_FILENAME = "wow_paladin_demo_balanced.pkl"
OUTPUT_FILE_PATH = os.path.join(project_root, "data", OUTPUT_FILENAME)

# 需要增强的动作ID
# 在wow_key_env.py中，G键的索引是1，]键的索引是2
# ACTION_TABLE = {0: 'F_AttackSelect', 1: 'G_SelectLastCorpse', 2: 'Bracket_LootOrInteract', ...}
ACTION_TO_BOOST = [1, 2]  # G和]键
BOOST_FACTOR = 5  # 将这些动作的样本复制5倍

def balance_demo_data():
    # 读取原始数据
    print(f"正在从 {INPUT_FILE_PATH} 读取演示数据...")
    try:
        with open(INPUT_FILE_PATH, "rb") as f:
            demo_data = pickle.load(f)
        print(f"成功加载了 {len(demo_data)} 个数据点。")
    except Exception as e:
        print(f"读取数据出错: {e}")
        return

    # 检查数据格式
    if not demo_data or not isinstance(demo_data, list):
        print("数据为空或格式不正确")
        return

    # 检查第一个样本的格式
    sample = demo_data[0]
    is_new_format = isinstance(sample, tuple) and len(sample) == 3
    
    if is_new_format:
        print("检测到新格式: (frame, state_dict, action_id)")
    else:
        print("数据格式不符合预期，需要 (frame, state_dict, action_id) 格式")
        return

    # 统计原始动作分布
    action_counts = Counter()
    for item in demo_data:
        if is_new_format:
            _, _, action_id = item
        else:
            _, action_id = item
        action_counts[action_id] += 1

    print("\n原始动作分布:")
    for action_id, count in sorted(action_counts.items()):
        print(f"动作 {action_id}: {count} 次")

    # 分离需要增强的样本
    boost_samples = []
    
    for item in demo_data:
        if is_new_format:
            _, _, action_id = item
        else:
            _, action_id = item
        
        if action_id in ACTION_TO_BOOST:
            boost_samples.append(item)

    if not boost_samples:
        print(f"警告: 未找到需要增强的动作 {ACTION_TO_BOOST}")
        return

    print(f"\n找到 {len(boost_samples)} 个需要增强的样本")
    
    # 创建增强后的数据集
    enhanced_data = demo_data.copy()
    
    for _ in range(BOOST_FACTOR - 1):  # 已经有一份在原始数据中
        enhanced_data.extend(boost_samples)
    
    # 打乱数据顺序
    random.shuffle(enhanced_data)
    
    # 统计增强后的动作分布
    enhanced_action_counts = Counter()
    for item in enhanced_data:
        if is_new_format:
            _, _, action_id = item
        else:
            _, action_id = item
        enhanced_action_counts[action_id] += 1

    print("\n增强后的动作分布:")
    for action_id, count in sorted(enhanced_action_counts.items()):
        print(f"动作 {action_id}: {count} 次")
        
    # 保存增强后的数据
    print(f"\n正在将增强后的数据（共 {len(enhanced_data)} 个样本）保存到 {OUTPUT_FILE_PATH}...")
    try:
        with open(OUTPUT_FILE_PATH, "wb") as f:
            pickle.dump(enhanced_data, f)
        print(f"增强后的数据已成功保存！")
    except Exception as e:
        print(f"保存数据出错: {e}")

if __name__ == "__main__":
    balance_demo_data() 