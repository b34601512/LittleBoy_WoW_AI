# D:\wow_ai\scripts\test_bc_model.py
import os
import sys
import pickle
import numpy as np
import torch
import time

# --- 动态路径设置 ---
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    project_root = None
# --- 路径设置结束 ---

# --- 引入训练脚本中的模型定义 ---
from scripts.bc_train_simple import BCMLPPolicy, preprocess_state, INPUT_FEATURE_DIM, ACTION_MAP_SIZE

# --- 配置路径 ---
MODEL_PATH = os.path.join(project_root, "runs", "bc_models", "bc_policy_simple.pth")
SCALER_PATH = os.path.join(project_root, "runs", "bc_models", "bc_state_scaler.pkl")
TEST_DATA_PATH = os.path.join(project_root, "data", "wow_paladin_demo_v1.pkl")

# --- ACTION_MAP for reference ---
ACTION_MAP_REFERENCE = {
    "no_op": 0, "tab": 1, "f": 2, "g": 3, "r": 4,
    "w": 5, "s": 6, "a": 7, "d": 8, "space": 9,
    "shift+tab": 10, "v": 11, "t": 12,
}
ACTION_ID_TO_NAME = {v: k for k, v in ACTION_MAP_REFERENCE.items()}

def load_model():
    """加载训练好的模型和缩放器"""
    print(f"Loading model from {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCMLPPolicy(INPUT_FEATURE_DIM, [128, 64], ACTION_MAP_SIZE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()  # 设置为评估模式
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler, device

def predict_action(model, scaler, state_dict, device):
    """使用模型预测给定状态应该采取的动作"""
    # 预处理状态
    state_features = preprocess_state(state_dict)
    
    # 使用缩放器归一化特征
    state_features_scaled = scaler.transform(state_features.reshape(1, -1))
    
    # 转换为张量并移动到设备
    state_tensor = torch.tensor(state_features_scaled, dtype=torch.float32).to(device)
    
    # 预测动作
    with torch.no_grad():
        logits = model(state_tensor)
        action_probs = torch.softmax(logits, dim=1)
        predicted_action = torch.argmax(action_probs, dim=1).item()
    
    # 获取前3个最可能的动作
    top3_probs, top3_actions = torch.topk(action_probs, 3, dim=1)
    top3_actions = top3_actions.cpu().numpy()[0]
    top3_probs = top3_probs.cpu().numpy()[0]
    
    return predicted_action, action_probs.cpu().numpy()[0], top3_actions, top3_probs

def test_on_validation_data():
    """在验证数据上测试模型"""
    model, scaler, device = load_model()
    
    # 加载测试数据
    print(f"Loading test data from {TEST_DATA_PATH}")
    with open(TEST_DATA_PATH, 'rb') as f:
        test_data = pickle.load(f)
    
    correct = 0
    total = 0
    confusion_matrix = np.zeros((ACTION_MAP_SIZE, ACTION_MAP_SIZE), dtype=int)
    action_counts = {}
    
    # 分析前20个样本的预测
    print("\n--- Detailed Analysis of First 20 Samples ---")
    for i in range(min(20, len(test_data))):
        # 假设数据格式是 (frame, state_dict, action_id)
        frame, state_dict, true_action = test_data[i]
        
        # 预测动作
        predicted_action, all_probs, top3_actions, top3_probs = predict_action(model, scaler, state_dict, device)
        
        # 记录结果
        is_correct = (predicted_action == true_action)
        action_name_true = ACTION_ID_TO_NAME.get(true_action, f"Unknown_{true_action}")
        action_name_pred = ACTION_ID_TO_NAME.get(predicted_action, f"Unknown_{predicted_action}")
        
        print(f"Sample {i+1}:")
        print(f"  True action: {true_action} ({action_name_true})")
        print(f"  Predicted: {predicted_action} ({action_name_pred}), Correct: {is_correct}")
        print(f"  Top 3 predictions:")
        for j in range(3):
            action_id = top3_actions[j]
            action_name = ACTION_ID_TO_NAME.get(action_id, f"Unknown_{action_id}")
            print(f"    {j+1}. Action {action_id} ({action_name}): {top3_probs[j]:.4f}")
        print()
    
    # 在整个数据集上评估
    print("\n--- Evaluating on Full Dataset ---")
    for i, data_point in enumerate(test_data):
        if isinstance(data_point, tuple) and len(data_point) == 3:
            frame, state_dict, true_action = data_point
        elif isinstance(data_point, tuple) and len(data_point) == 2:
            state_dict, true_action = data_point
        else:
            print(f"Unknown data format at index {i}")
            continue
        
        # 预测动作
        predicted_action, _, _, _ = predict_action(model, scaler, state_dict, device)
        
        # 更新统计
        confusion_matrix[true_action, predicted_action] += 1
        action_counts[true_action] = action_counts.get(true_action, 0) + 1
        
        if predicted_action == true_action:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Model accuracy on test data: {accuracy:.4f} ({correct}/{total})")
    
    # 按动作类型分析准确率
    print("\n--- Accuracy by Action Type ---")
    for action_id, count in sorted(action_counts.items()):
        action_correct = confusion_matrix[action_id, action_id]
        action_accuracy = action_correct / count if count > 0 else 0
        action_name = ACTION_ID_TO_NAME.get(action_id, f"Unknown_{action_id}")
        print(f"Action {action_id} ({action_name}): {action_accuracy:.4f} ({action_correct}/{count})")
    
    return accuracy

if __name__ == "__main__":
    test_on_validation_data() 