import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# --- CONFIGURATION ---
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/zhuoying/projects/def-xilinliu/data/extracted_data_2ch")
    parser.add_argument("--scratch_dir", type=str, default=os.getenv("SCRATCH", "./"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--confidence_threshold", type=float, default=0.9, help="Threshold for pseudo-labeling")
    return parser.parse_args()

# --- MODEL ---
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=2, n_classes=5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 1920, 128) 
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# --- DATA UTILITIES ---
def list_to_dataset(data_list):
    if not data_list: return None
    n_samples = len(data_list)
    x_tensor = torch.zeros((n_samples, 2, 7680))
    y_tensor = torch.zeros((n_samples), dtype=torch.long)
    
    for i, item in enumerate(data_list):
        x_tensor[i, 0, :] = item[0]
        x_tensor[i, 1, :] = item[1]
        label = item[2]
        if isinstance(label, torch.Tensor): label = label.item()
        y_tensor[i] = int(label)
    return TensorDataset(x_tensor, y_tensor)

def load_folds(folds, base_path, folder_type="train_set.pt"):
    data = []
    print(f"Loading {folder_type} from folds: {folds}")
    for fold in folds:
        path = os.path.join(base_path, str(fold), folder_type)
        if os.path.exists(path):
            # weights_only=False for compatibility
            fold_data = torch.load(path, weights_only=False)
            data.extend(fold_data)
    return data

def filter_leakage(data_list, banned_pids):
    """Removes any sample belonging to banned patient IDs."""
    clean_data = []
    removed = 0
    for item in data_list:
        pid = item[3]
        if isinstance(pid, torch.Tensor): pid = int(pid.item())
        
        if pid in banned_pids:
            removed += 1
        else:
            clean_data.append(item)
    return clean_data, removed

# --- SEMI-SUPERVISED LOGIC ---

def generate_pseudo_labels(model, unlabeled_data, device, threshold=0.9):
    """Generates pseudo-labels for unlabeled data using the trained model."""
    model.eval()
    pseudo_labeled_data = []
    
    # Process in batches for speed
    batch_size = 64
    n_samples = len(unlabeled_data)
    
    print(f"Generating pseudo-labels for {n_samples} unlabeled samples...")
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = unlabeled_data[i:i+batch_size]
            
            # Prepare batch tensor
            inputs = torch.stack([item[0] for item in batch]) # item[0] is ch1? No, wait.
            # Correct structure: Unlabeled data is likely (ch1, ch2, label, pid)
            # We need to stack (ch1, ch2)
            x_batch = torch.zeros((len(batch), 2, 7680))
            for j, item in enumerate(batch):
                x_batch[j, 0, :] = item[0]
                x_batch[j, 1, :] = item[1]
            
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            probs = F.softmax(outputs, dim=1)
            confidence, preds = torch.max(probs, dim=1)
            
            # Filter by confidence
            for j in range(len(batch)):
                if confidence[j].item() >= threshold:
                    # Create new tuple: (ch1, ch2, PSEUDO_LABEL, pid)
                    original_item = batch[j]
                    new_item = (original_item[0], original_item[1], preds[j].item(), original_item[3])
                    pseudo_labeled_data.append(new_item)

    print(f"-> Generated {len(pseudo_labeled_data)} pseudo-labels (High Confidence).")
    return pseudo_labeled_data

def train_loop(model, train_data, device, epochs, lr, phase_name="Supervised"):
    # Safe Split by Patient ID
    patient_ids = sorted(list(set([item[3] for item in train_data])))
    split_idx = int(len(patient_ids) * 0.8)
    train_pids = set(patient_ids[:split_idx])
    val_pids = set(patient_ids[split_idx:])
    
    train_subset = [x for x in train_data if x[3] in train_pids]
    val_subset = [x for x in train_data if x[3] in val_pids]
    
    print(f"\n--- {phase_name} Training ---")
    print(f"Train Samples: {len(train_subset)} | Val Samples: {len(val_subset)}")

    train_loader = DataLoader(list_to_dataset(train_subset), batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(list_to_dataset(val_subset), batch_size=64, shuffle=False, num_workers=4)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                pred = torch.argmax(model(vx), dim=1)
                correct += (pred == vy).sum().item()
                total += vy.size(0)
        
        val_acc = correct / total if total > 0 else 0
        avg_loss = total_loss / len(train_loader)
        
        # Print every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
             print(f"[{phase_name}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

# --- MAIN ---
if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. IDENTIFY LEAKAGE PATIENTS (From Test Fold 10)
    test_path = os.path.join(args.data_dir, "10", "val_set.pt")
    print(f"Scanning Test Fold 10 for Patients...")
    test_data_raw = torch.load(test_path, weights_only=False)
    
    banned_pids = set()
    for item in test_data_raw:
        pid = item[3]
        if isinstance(pid, torch.Tensor): pid = int(pid.item())
        banned_pids.add(pid)
    print(f"BANNED PATIENTS (In Test Set): {banned_pids}")

    # 2. LOAD LABELED DATA
    labeled_folds = [0, 2, 4, 6, 8]
    labeled_data_raw = load_folds(labeled_folds, args.data_dir, "train_set.pt")
    
    # Filter Labeled Data
    labeled_data, rem1 = filter_leakage(labeled_data_raw, banned_pids)
    print(f"Labeled Data: Loaded {len(labeled_data_raw)}, Removed {rem1} leaking samples.")

    # 3. STEP 1: TRAIN TEACHER (Initial Supervised)
    model = SimpleCNN().to(device)
    # Train for fewer epochs initially (e.g., 50% of total)
    train_loop(model, labeled_data, device, epochs=int(args.epochs/2), lr=args.lr, phase_name="Teacher")

    # 4. LOAD UNLABELED DATA
    # Folds not in labeled, and not test fold 10
    all_folds = range(0, 20) # Adjust range based on your dataset size (e.g., 0-20 or 0-40)
    # Assuming structure is folder '0', '2', '4'... 
    # Note: You had unlabeled_folds = [i for i in range(0, 62, 2)...] in your original code
    # Let's try to load a few unlabeled folds. 
    unlabeled_folds = [12, 14, 16] # Example folds for SSL
    unlabeled_data_raw = load_folds(unlabeled_folds, args.data_dir, "train_set.pt") # Using train_set as source of unlabeled data
    
    # Filter Unlabeled Data (CRITICAL: Test patients might be here too!)
    unlabeled_data, rem2 = filter_leakage(unlabeled_data_raw, banned_pids)
    print(f"Unlabeled Data: Loaded {len(unlabeled_data_raw)}, Removed {rem2} leaking samples.")

    # 5. STEP 2: GENERATE PSEUDO LABELS
    pseudo_data = generate_pseudo_labels(model, unlabeled_data, device, threshold=args.confidence_threshold)
    
    # 6. STEP 3: RETRAIN STUDENT (Semi-Supervised)
    # Combine datasets
    full_training_data = labeled_data + pseudo_data
    print(f"Combined Dataset: {len(labeled_data)} labeled + {len(pseudo_data)} pseudo = {len(full_training_data)} total.")
    
    # Reset model or fine-tune? Usually fine-tune or train fresh. Let's fine-tune.
    train_loop(model, full_training_data, device, epochs=args.epochs, lr=args.lr/2, phase_name="Student")
    
    # 7. FINAL EVALUATION
    print("\n--- FINAL EVALUATION ON FOLD 10 ---")
    test_ds = list_to_dataset(test_data_raw)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            all_preds.extend(torch.argmax(model(x), dim=1).cpu().tolist())
            all_labels.extend(y.tolist())
            
    print(classification_report(all_labels, all_preds, digits=4))
    print(confusion_matrix(all_labels, all_preds))
    
    # Save
    torch.save(model.state_dict(), os.path.join(args.scratch_dir, "semi_supervised_model.pt"))
    print("Saved semi_supervised_model.pt")
