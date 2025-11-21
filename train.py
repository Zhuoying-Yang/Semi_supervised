import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import copy
import gc

# --- CONFIGURATION ---
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/zhuoying/projects/def-xilinliu/data/extracted_data_2ch")
    parser.add_argument("--scratch_dir", type=str, default=os.getenv("SCRATCH", "./"))
    parser.add_argument("--target_fold", type=int, default=10, help="The fold to use for Testing")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--confidence_threshold", type=float, default=0.90)
    parser.add_argument("--patience", type=int, default=15)
    return parser.parse_args()

# --- IMPROVED DEEP CNN ---
class ImprovedSleepCNN(nn.Module):
    def __init__(self, in_channels=2, n_classes=5):
        super(ImprovedSleepCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=65, stride=2, padding=32)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=17, stride=1, padding=8)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

# --- DATASET CLASS ---
class SleepDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # item: (ch1, ch2, label, pid)
        item = self.data[idx]
        x = torch.stack([item[0], item[1]], dim=0)
        label = item[2]
        if isinstance(label, torch.Tensor): label = label.item()
        return x, int(label)

# --- HELPER FUNCTIONS ---
def load_single_fold(base_path, fold_idx, filename):
    path = os.path.join(base_path, str(fold_idx), filename)
    print(f"   Loading file: {path}...")
    data = torch.load(path, weights_only=False)
    return data

def split_labeled_unlabeled(full_train_data, labeled_pids):
    """Splits the training data into Labeled and Unlabeled based on Patient IDs."""
    labeled = []
    unlabeled = []
    
    for item in full_train_data:
        pid = item[3]
        if isinstance(pid, torch.Tensor): pid = int(pid.item())
        
        if pid in labeled_pids:
            labeled.append(item)
        else:
            unlabeled.append(item)
            
    return labeled, unlabeled

def get_class_weights(train_data, device):
    labels = [int(x[2].item()) if isinstance(x[2], torch.Tensor) else int(x[2]) for x in train_data]
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float).to(device)

# --- TRAINING LOOP ---
def train_loop(model, train_data, device, epochs, lr, patience, phase_name="Supervised"):
    # Internal Val Split
    patient_ids = sorted(list(set([int(item[3].item()) if isinstance(item[3], torch.Tensor) else int(item[3]) for item in train_data])))
    split_idx = int(len(patient_ids) * 0.85)
    train_pids = set(patient_ids[:split_idx])
    
    train_subset = [x for x in train_data if (int(x[3].item()) if isinstance(x[3], torch.Tensor) else int(x[3])) in train_pids]
    val_subset = [x for x in train_data if (int(x[3].item()) if isinstance(x[3], torch.Tensor) else int(x[3])) not in train_pids]
    
    print(f"[{phase_name}] Train: {len(train_subset)} | Val: {len(val_subset)}")

    train_loader = DataLoader(SleepDataset(train_subset), batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(SleepDataset(val_subset), batch_size=128, shuffle=False, num_workers=4)
    
    class_weights = get_class_weights(train_subset, device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    best_acc = 0.0
    wait = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
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
        
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        if (epoch + 1) % 1 == 0:
             print(f"[{phase_name}] Ep {epoch+1} | Loss: {avg_loss:.4f} | Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early Stopping! Best Acc: {best_acc:.4f}")
                break
    
    model.load_state_dict(best_model_wts)

def generate_pseudo_labels(model, unlabeled_data, device, threshold=0.90):
    model.eval()
    pseudo_data = []
    batch_size = 128
    
    # Chunking to save memory during inference
    print(f"Generating pseudo-labels for {len(unlabeled_data)} samples...")
    
    with torch.no_grad():
        for i in range(0, len(unlabeled_data), batch_size):
            batch = unlabeled_data[i:i+batch_size]
            # Manually stack batch
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
                    # Structure: (ch1, ch2, PSEUDO_LABEL, pid)
                    pseudo_data.append((orig[0], orig[1], preds[j].item(), orig[3]))
    
    print(f"-> Generated {len(pseudo_data)} pseudo-labels.")
    return pseudo_data

# --- MAIN ---
if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. LOAD FOLD 10 (Contains ALL training data except Patients 10 & 11)
    # This file is ~100GB. We load it ONCE.
    print(f"\n--- LOADING FOLD {args.target_fold} MASTER DATA ---")
    full_train_data = load_single_fold(args.data_dir, args.target_fold, "train_set.pt")
    
    # 2. DEFINE LABELED PATIENTS
    # We treat Patients 0-9 as "Labeled". Everyone else (12-61) is "Unlabeled".
    # Note: Patients 10 & 11 are NOT in this file (they are in val_set.pt), so no leakage risk.
    labeled_pids = set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(f"Labeled Patients: {labeled_pids}")
    
    # 3. SPLIT DATA IN MEMORY
    print("Splitting data into Labeled and Unlabeled...")
    labeled_data, unlabeled_data = split_labeled_unlabeled(full_train_data, labeled_pids)
    
    # Free up the massive full_train_data list to save RAM
    del full_train_data
    gc.collect()
    
    print(f"Labeled Samples: {len(labeled_data)}")
    print(f"Unlabeled Samples: {len(unlabeled_data)}")
    
    # 4. TRAIN TEACHER
    print("\n--- Training Teacher ---")
    model = ImprovedSleepCNN().to(device)
    train_loop(model, labeled_data, device, epochs=40, lr=args.lr, patience=args.patience, phase_name="Teacher")
    
    # 5. PSEUDO LABELING
    print("\n--- Generating Pseudo Labels ---")
    pseudo_data = generate_pseudo_labels(model, unlabeled_data, device, threshold=args.confidence_threshold)
    
    # Free Unlabeled Data to save RAM (we only need pseudo_data now)
    del unlabeled_data
    gc.collect()
    
    # 6. TRAIN STUDENT
    full_data = labeled_data + pseudo_data
    print(f"\n--- Retraining Student on {len(full_data)} samples ---")
    train_loop(model, full_data, device, epochs=80, lr=args.lr, patience=args.patience, phase_name="Student")
    
    # 7. FINAL TEST (Load Test Set Last to save memory)
    print("\n--- FINAL TEST (Fold {args.target_fold} Validation Set) ---")
    test_data = load_single_fold(args.data_dir, args.target_fold, "val_set.pt")
    test_loader = DataLoader(SleepDataset(test_data), batch_size=128, shuffle=False)
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            all_preds.extend(torch.argmax(model(x), dim=1).cpu().tolist())
            all_labels.extend(y.tolist())
            
    print(classification_report(all_labels, all_preds, digits=4))
    print(confusion_matrix(all_labels, all_preds))
    
    torch.save(model.state_dict(), os.path.join(args.scratch_dir, "final_sleep_model.pt"))
