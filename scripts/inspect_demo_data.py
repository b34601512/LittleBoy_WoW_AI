# D:\wow_ai\scripts\inspect_demo_data.py
import pickle
import os
import sys
import numpy as np

# --- 动态路径设置 (与 record_demo.py 类似，确保能找到项目根目录) ---
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir) # wow_ai
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        # print(f"Added project root '{project_root}' to sys.path")
except NameError:
    project_root = None
# --- 路径设置结束 ---

# --- Configuration ---
# 与 record_demo.py 中的 OUTPUT_FILENAME 一致，或者直接写完整路径
# DATA_FILENAME = "wow_demo_data.pkl" # 如果检查旧数据
DATA_FILENAME = "wow_demo_data_with_heals.pkl" # 检查我们最新的数据
DATA_FILE_PATH = os.path.join(project_root, "data", DATA_FILENAME)

# --- ACTION_MAP for reference (与 record_demo.py 中的一致) ---
ACTION_MAP_REFERENCE = {
    "no_op": 0, "tab": 1, "f": 2, "g": 3, "r": 4,
    "w": 5, "s": 6, "a": 7, "d": 8, "space": 9,
    "shift+tab": 10, "v": 11, "t": 12,
}
ACTION_ID_TO_NAME_REFERENCE = {v: k for k, v in ACTION_MAP_REFERENCE.items()}
# --- End ACTION_MAP ---

def inspect_data(file_path):
    print(f"Attempting to load data from: {file_path}")
    if not os.path.exists(file_path):
        print(f"ERROR: Data file not found at '{file_path}'")
        return

    try:
        with open(file_path, "rb") as f:
            recorded_data = pickle.load(f)
        print(f"Successfully loaded {len(recorded_data)} data points.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if not recorded_data:
        print("The data file is empty.")
        return

    print("\n--- Sample Data Inspection ---")
    num_samples_to_show = min(5, len(recorded_data)) # Show up to 5 samples
    
    print(f"\nShowing first {num_samples_to_show} samples:")
    for i in range(num_samples_to_show):
        # 首先检查数据结构
        data_item = recorded_data[i]
        print(f"\n--- Sample {i+1} ---")
        
        # 适应不同的数据格式
        if isinstance(data_item, tuple) and len(data_item) == 2:
            # 旧格式: (state_dict, action_id)
            state_dict, action_id = data_item
            print(f"  Format: Old (state_dict, action_id)")
        elif isinstance(data_item, tuple) and len(data_item) == 3:
            # 新格式可能是: (frame, state_dict, action_id)
            frame, state_dict, action_id = data_item
            print(f"  Format: New (frame, state_dict, action_id)")
            print(f"  Frame shape: {frame.shape if hasattr(frame, 'shape') else type(frame)}")
        else:
            print(f"  Unknown data format: {type(data_item)}")
            print(f"  Data preview: {str(data_item)[:100]}")
            continue
        
        # 显示动作ID和名称
        action_name = ACTION_ID_TO_NAME_REFERENCE.get(action_id, f"Unknown ID {action_id}")
        print(f"  Action ID: {action_id} (Means: '{action_name}')")
        
        # 显示状态字典
        print(f"  State Dictionary (Keys and types/shapes):")
        if isinstance(state_dict, dict):
            for key, value in state_dict.items():
                value_type = type(value)
                value_shape = getattr(value, 'shape', 'N/A (not an array)')
                print(f"    '{key}': type={value_type}, shape/value_preview={value_shape if value_shape != 'N/A (not an array)' else str(value)[:50]}")
        else:
            print(f"    Unexpected state format: {type(state_dict)}")

    if len(recorded_data) > num_samples_to_show:
        print(f"\nShowing last sample (Sample {len(recorded_data)}):")
        
        # 同样适应不同的数据格式
        data_item = recorded_data[-1]
        print(f"\n--- Sample {len(recorded_data)} ---")
        
        if isinstance(data_item, tuple) and len(data_item) == 2:
            state_dict, action_id = data_item
            print(f"  Format: Old (state_dict, action_id)")
        elif isinstance(data_item, tuple) and len(data_item) == 3:
            frame, state_dict, action_id = data_item
            print(f"  Format: New (frame, state_dict, action_id)")
            print(f"  Frame shape: {frame.shape if hasattr(frame, 'shape') else type(frame)}")
        else:
            print(f"  Unknown data format: {type(data_item)}")
            print(f"  Data preview: {str(data_item)[:100]}")
            return
        
        action_name = ACTION_ID_TO_NAME_REFERENCE.get(action_id, f"Unknown ID {action_id}")
        print(f"  Action ID: {action_id} (Means: '{action_name}')")
        
        print(f"  State Dictionary (Keys and types/shapes):")
        if isinstance(state_dict, dict):
            for key, value in state_dict.items():
                value_type = type(value)
                value_shape = getattr(value, 'shape', 'N/A (not an array)')
                print(f"    '{key}': type={value_type}, shape/value_preview={value_shape if value_shape != 'N/A (not an array)' else str(value)[:50]}")
        else:
            print(f"    Unexpected state format: {type(state_dict)}")
            
    # Count action occurrences
    action_counts = {}
    for item in recorded_data:
        if isinstance(item, tuple):
            if len(item) == 2:
                _, action_id = item
            elif len(item) == 3:
                _, _, action_id = item
            else:
                continue
            action_counts[action_id] = action_counts.get(action_id, 0) + 1
    
    print("\n--- Action Distribution ---")
    for action_id, count in sorted(action_counts.items()):
        action_name = ACTION_ID_TO_NAME_REFERENCE.get(action_id, f"Unknown ID {action_id}")
        print(f"  Action {action_id} ('{action_name}'): {count} occurrences")

if __name__ == "__main__":
    # 处理命令行参数
    if len(sys.argv) > 1:
        # 如果提供了完整路径
        if os.path.exists(sys.argv[1]):
            custom_file_path = sys.argv[1]
        # 如果只提供了文件名
        elif os.path.exists(os.path.join(project_root, "data", sys.argv[1])):
            custom_file_path = os.path.join(project_root, "data", sys.argv[1]) 
        else:
            print(f"WARNING: File not found at {sys.argv[1]}")
            print(f"Falling back to default: {DATA_FILE_PATH}")
            custom_file_path = DATA_FILE_PATH
        inspect_data(custom_file_path)
    else:
        inspect_data(DATA_FILE_PATH)