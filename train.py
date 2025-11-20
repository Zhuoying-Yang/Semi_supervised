import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# --- CONFIGURATION ---
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/zhuoying/projects/def-xilinliu/data/extracted_data_2ch")
    parser.add_argument("--scratch_dir", type=str, default=os.getenv("SCRATCH", "./"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--confidence_threshold", type=float, default=0.85, help="Lowered threshold for pseudo-labels")
    return parser.parse_args()

# --- IMPROVED MODEL ARCHITECTURE ---
class ImprovedSleepCNN(nn.Module):
    def __init__(self, in_channels=2, n_classes=5):
        super(ImprovedSleepCNN, self).__init__()
        
        # Block 1: Large Kernel to capture low-frequency (Delta/Theta) waves
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=65, stride=2, padding=32)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(4)
        
        # Block 2: Medium Kernel
        self.conv2 = nn.Conv1d(64, 128, kernel_size=17, stride=1, padding=8)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        
        # Block 3: Small Kernel for high-frequency features (Spindles)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)

        # Block 4: Deep features
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(2)

        # Global Average Pooling (Reduces parameters, prevents overfitting)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # Block 4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Global Pooling & FC
        x = self.global_pool(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.dropout(x)
        return self.fc(x)

# --- UTILS ---
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
            data.extend(torch.load(path, weights_only=False))
    return data

def filter_leakage(data_list, banned_pids):
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

def get_class_weights(train_data, device):
    """Calculates weights to balance N1, N2, N3, REM, Wake."""
    labels = [int(x[2].item()) if isinstance(x[2], torch.Tensor) else int(x[2]) for x in train_data]
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float).to(device)

# --- TRAINING LOGIC ---
def train_loop(model, train_data, device, epochs, lr, phase_name="Supervised"):
    # Split by Patient
    patient_ids = sorted(list(set([item[3] for item in train_data])))
    split_idx = int(len(patient_ids) * 0.85) # 85/15 split for better training
    train_pids = set(patient_ids[:split_idx])
    val_pids = set(patient_ids[split_idx:])
    
    train_subset = [x for x in train_data if x[3] in train_pids]
    val_subset = [x for x in train_data if x[3] in val_pids]
    
    print(f"Train Samples: {len(train_subset)} | Val Samples: {len(val_subset)}")

    train_loader = DataLoader(list_to_dataset(train_subset), batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(list_to_dataset(val_subset), batch_size=128, shuffle=False, num_workers=4)
    
    # Calculate Class Weights (Crucial for N1 accuracy)
    class_weights = get_class_weights(train_subset, device)
    print(f"Class Weights: {class_weights.cpu().numpy()}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) # Added weight_decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
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
        
        # Step the scheduler
        scheduler.step(val_acc)
        
        if (epoch + 1) % 1 == 0:
             print(f"[{phase_name}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

def generate_pseudo_labels(model, unlabeled_data, device, threshold=0.85):
    model.eval()
    pseudo_data = []
    batch_size = 128
    print(f"Generating pseudo-labels for {len(unlabeled_data)} samples...")
    
    with torch.no_grad():
        for i in range(0, len(unlabeled_data), batch_size):
            batch = unlabeled_data[i:i+batch_size]
            x_batch = torch.zeros((len(batch), 2, 7680))
            for j, item in enumerate(batch):
                x_batch[j, 0, :] = item[0]
                x_batch[j, 1, :] = item[1]
            
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            probs = F.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, dim=1)
            
            for j in range(len(batch)):
                if conf[j].item() >= threshold:
                    orig = batch[j]
                    pseudo_data.append((orig[0], orig[1], preds[j].item(), orig[3]))

    print(f"-> Generated {len(pseudo_data)} pseudo-labels.")
    return pseudo_data

# --- MAIN ---
if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. TEST PATIENTS CHECK
    test_path = os.path.join(args.data_dir, "10", "val_set.pt")
    test_data_raw = torch.load(test_path, weights_only=False)
    banned_pids = set([int(item[3].item()) if isinstance(item[3], torch.Tensor) else int(item[3]) for item in test_data_raw])
    print(f"BANNED PATIENTS: {banned_pids}")

    # 2. LOAD LABELED
    labeled_folds = [0, 2, 4, 6, 8]
    labeled_data_raw = load_folds(labeled_folds, args.data_dir, "train_set.pt")
    labeled_data, _ = filter_leakage(labeled_data_raw, banned_pids)
    print(f"Labeled Samples: {len(labeled_data)}")

    # 3. TRAIN TEACHER
    model = ImprovedSleepCNN().to(device)
    train_loop(model, labeled_data, device, epochs=30, lr=args.lr, phase_name="Teacher")

    # 4. LOAD UNLABELED & PSEUDO LABEL
    unlabeled_folds = [12, 14, 16, 18] # Added more folds
    unlabeled_data_raw = load_folds(unlabeled_folds, args.data_dir, "train_set.pt")
    unlabeled_data, _ = filter_leakage(unlabeled_data_raw, banned_pids)
    
    pseudo_data = generate_pseudo_labels(model, unlabeled_data, device, threshold=args.confidence_threshold)
    
    # 5. TRAIN STUDENT
    full_data = labeled_data + pseudo_data
    # Reset optimizer state by re-calling train_loop, but we keep the model weights
    print(f"Retraining Student on {len(full_data)} samples...")
    train_loop(model, full_data, device, epochs=50, lr=args.lr/2, phase_name="Student")
    
    # 6. FINAL TEST
    print("\n--- FINAL TEST FOLD 10 ---")
    test_ds = list_to_dataset(test_data_raw)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            all_preds.extend(torch.argmax(model(x), dim=1).cpu().tolist())
            all_labels.extend(y.tolist())
            
    print(classification_report(all_labels, all_preds, digits=4))
    print(confusion_matrix(all_labels, all_preds))
    
    torch.save(model.state_dict(), os.path.join(args.scratch_dir, "improved_sleep_model.pt"))
