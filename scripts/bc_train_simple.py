# D:\wow_ai\scripts\bc_train_simple.py
# (Behavioral Cloning training script using structured states only)
import pickle
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler # For normalizing continuous features

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

# --- Configuration ---
DATA_FILENAME = "wow_demo_data_with_heals.pkl"
DATA_FILE_PATH = os.path.join(project_root, "data", DATA_FILENAME)
MODEL_SAVE_PATH = os.path.join(project_root, "runs", "bc_models", "bc_policy_simple.pth")
SCALER_SAVE_PATH = os.path.join(project_root, "runs", "bc_models", "bc_state_scaler.pkl") # For saving the scaler

# --- ACTION_MAP for reference (from record_demo.py) ---
# This defines the number of output neurons for the policy network
ACTION_MAP_SIZE = 13 # Actions 0 through 12 (no_op to t_healingtouch)

# --- Hyperparameters ---
INPUT_FEATURE_DIM = 9 # We will construct a 9-dimensional feature vector
HIDDEN_DIMS = [128, 64] # MLP hidden layers
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 50 # Start with a moderate number of epochs
TEST_SPLIT_RATIO = 0.2 # 20% of data for validation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Define the Policy Network (MLP) ---
class BCMLPPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU()) # Or nn.Tanh()
            # layers.append(nn.Dropout(0.2)) # Optional dropout for regularization
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        # No Softmax here, as nn.CrossEntropyLoss will apply it internally
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# --- Function to preprocess state_dict into a feature vector ---
def preprocess_state(state_dict):
    # Order of features matters and must be consistent!
    # [sel_flag, is_dead, yolo_prob_target, 
    #  err_face, err_range, err_no_target,
    #  need_face, need_range, no_target_error]
    
    s = state_dict
    feature_vector = [
        float(s.get('sel_flag', 0)),
        float(s.get('is_dead', 0)),
        float(s.get('yolo_prob_target', 0.0)),
        float(s.get('errors_dict', {}).get('face', 0)),
        float(s.get('errors_dict', {}).get('range', 0)),
        float(s.get('errors_dict', {}).get('no_target', 0)),
        float(s.get('need_face', False)), # bools converted to float (0.0 or 1.0)
        float(s.get('need_range', False)),
        float(s.get('no_target_error', False))
    ]
    if len(feature_vector) != INPUT_FEATURE_DIM:
        raise ValueError(f"Feature vector length mismatch! Expected {INPUT_FEATURE_DIM}, got {len(feature_vector)}")
    return np.array(feature_vector, dtype=np.float32)


# --- Main Training Logic ---
def train_bc_policy():
    print(f"Attempting to load data from: {DATA_FILE_PATH}")
    if not os.path.exists(DATA_FILE_PATH):
        print(f"ERROR: Data file not found at '{DATA_FILE_PATH}'"); return

    try:
        with open(DATA_FILE_PATH, "rb") as f:
            recorded_data = pickle.load(f) # List of (state_dict, action_id)
        print(f"Successfully loaded {len(recorded_data)} data points.")
    except Exception as e:
        print(f"Error loading data: {e}"); return

    if not recorded_data:
        print("The data file is empty. No training can be done."); return

    # 1. Preprocess data into features (X) and labels (y)
    print("Preprocessing data...")
    all_features = []
    all_labels = []
    for state_dict, action_id in recorded_data:
        try:
            features = preprocess_state(state_dict)
            all_features.append(features)
            all_labels.append(action_id)
        except Exception as e:
            print(f"Skipping data point due to preprocessing error: {e}. State: {state_dict}")
            continue
    
    if not all_features:
        print("No valid data points after preprocessing. Exiting."); return

    X = np.array(all_features)
    y = np.array(all_labels, dtype=np.int64) # CrossEntropyLoss expects LongTensor for labels

    print(f"Data preprocessed. Feature shape: {X.shape}, Label shape: {y.shape}")

    # 2. Normalize continuous features (Optional but recommended for yolo_prob_target if its range varies)
    # For simplicity, we'll normalize all features here.
    # Important: Save the scaler to use it during inference!
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler
    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    with open(SCALER_SAVE_PATH, 'wb') as f_scaler:
        pickle.dump(scaler, f_scaler)
    print(f"State feature scaler saved to {SCALER_SAVE_PATH}")


    # 3. Create PyTorch datasets and dataloaders
    dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), 
                            torch.tensor(y, dtype=torch.long))
    
    dataset_size = len(dataset)
    val_size = int(TEST_SPLIT_RATIO * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Created DataLoader. Train size: {train_size}, Validation size: {val_size}")

    # 4. Initialize model, criterion, optimizer
    model = BCMLPPolicy(INPUT_FEATURE_DIM, HIDDEN_DIMS, ACTION_MAP_SIZE).to(device)
    criterion = nn.CrossEntropyLoss() # Suitable for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Model initialized:\n{model}")
    print(f"Training for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE}...")

    # 5. Training loop
    best_val_accuracy = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train() # Set model to training mode
        train_loss_sum = 0
        train_correct_sum = 0
        train_total_samples = 0

        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features) # Forward pass
            loss = criterion(outputs, labels) # Calculate loss
            loss.backward() # Backward pass
            optimizer.step() # Update weights
            
            train_loss_sum += loss.item() * features.size(0)
            _, predicted_actions = torch.max(outputs, 1)
            train_correct_sum += (predicted_actions == labels).sum().item()
            train_total_samples += labels.size(0)

        avg_train_loss = train_loss_sum / train_total_samples
        train_accuracy = train_correct_sum / train_total_samples

        # Validation loop
        model.eval() # Set model to evaluation mode
        val_loss_sum = 0
        val_correct_sum = 0
        val_total_samples = 0
        with torch.no_grad(): # No need to calculate gradients during validation
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss_sum += loss.item() * features.size(0)
                _, predicted_actions = torch.max(outputs, 1)
                val_correct_sum += (predicted_actions == labels).sum().item()
                val_total_samples += labels.size(0)

        avg_val_loss = val_loss_sum / val_total_samples
        val_accuracy = val_correct_sum / val_total_samples
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Save model if it has the best validation accuracy so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  Best model saved to {MODEL_SAVE_PATH} (Val Acc: {best_val_accuracy:.4f})")

    print("\nTraining finished.")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Final model (last epoch or best) saved at: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_bc_policy()