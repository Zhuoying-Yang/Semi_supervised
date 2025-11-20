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
import copy
import gc  # Added for memory management

# --- CONFIGURATION ---
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/zhuoying/projects/def-xilinliu/data/extracted_data_2ch")
    parser.add_argument("--scratch_dir", type=str, default=os.getenv("SCRATCH", "./"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--confidence_threshold", type=float, default=0.90)
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()

# --- SIMPLE CNN MODEL ---
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
    print(f"   Loading {len(folds)} folds: {folds}...")
    for fold in folds:
        path = os.path.join(base_path, str(fold), folder_type)
        if os.path.exists(path):
            data.extend(torch.load(path, weights_only=False))
    return data

def filter_leakage(data_list, banned_pids):
    clean_data = []
    for item in data_list:
        pid = item[3]
        if isinstance(pid, torch.Tensor): pid = int(pid.item())
        if pid not in banned_pids:
            clean_data.append(item)
    return clean_data

def get_class_weights(train_data, device):
    labels = [int(x[2].item()) if isinstance(x[2], torch.Tensor) else int(x[2]) for x in train_data]
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float).to(device)

# --- TRAINING LOOP ---
def train_loop(model, train_data, device, epochs, lr, patience, phase_name="Supervised"):
    patient_ids = sorted(list(set([item[3] for item in train_data])))
    split_idx = int(len(patient_ids) * 0.85)
    train_pids = set(patient_ids[:split_idx])
    val_pids = set(patient_ids[split_idx:])
    
    train_subset = [x for x in train_data if x[3] in train_pids]
    val_subset = [x for x in train_data if x[3] in val_pids]
    
    print(f"[{phase_name}] Train: {len(train_subset)} | Val: {len(val_subset)}")

    train_loader = DataLoader(list_to_dataset(train_subset), batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(list_to_dataset(val_subset), batch_size=64, shuffle=False, num_workers=4)
    class_weights = get_class_weights(train_subset, device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
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
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
             print(f"[{phase_name}] Ep {epoch+1} | Loss: {avg_loss:.4f} | Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"🛑 Early Stopping! Best Acc: {best_acc:.4f}")
                break
    
    model.load_state_dict(best_model_wts)

# --- STREAMING PSEUDO LABELS (MEMORY FIX) ---
def stream_pseudo_labels(model, fold_list, device, banned_pids, data_dir, threshold=0.90):
    model.eval()
    all_pseudo_data = []
    chunk_size = 5 # Load 5 folds at a time
    
    print(f"\n🌊 Starting Streaming Pseudo-Labeling on {len(fold_list)} folds...")
    
    # Split folds into chunks
    chunks = [fold_list[i:i + chunk_size] for i in range(0, len(fold_list), chunk_size)]
    
    for i, chunk_folds in enumerate(chunks):
        print(f"   Processing Chunk {i+1}/{len(chunks)}: Folds {chunk_folds}")
        
        # 1. Load Chunk
        chunk_raw = load_folds(chunk_folds, data_dir, "train_set.pt")
        
        # 2. Filter Leakage
        chunk_clean = filter_leakage(chunk_raw, banned_pids)
        
        # 3. Generate Labels
        batch_size = 64
        chunk_pseudo = []
        with torch.no_grad():
            for k in range(0, len(chunk_clean), batch_size):
                batch = chunk_clean[k:k+batch_size]
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
                        # Append tuple: (ch1, ch2, PSEUDO_LABEL, pid)
                        chunk_pseudo.append((orig[0], orig[1], preds[j].item(), orig[3]))
        
        print(f"   -> Chunk produced {len(chunk_pseudo)} pseudo-labels.")
        all_pseudo_data.extend(chunk_pseudo)
        
        # 4. FREE MEMORY (Crucial)
        del chunk_raw
        del chunk_clean
        del chunk_pseudo
        gc.collect() # Force garbage collection
        
    print(f"Total Pseudo-Labels Generated: {len(all_pseudo_data)}")
    return all_pseudo_data

# --- MAIN ---
if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. BANNED PATIENTS
    test_path = os.path.join(args.data_dir, "10", "val_set.pt")
    test_data_raw = torch.load(test_path, weights_only=False)
    banned_pids = set([int(item[3].item()) if isinstance(item[3], torch.Tensor) else int(item[3]) for item in test_data_raw])
    print(f"BANNED PATIENTS: {banned_pids}")

    # 2. LABELED DATA
    labeled_folds = [0, 2, 4, 6, 8]
    labeled_data = load_folds(labeled_folds, args.data_dir)
    labeled_data = filter_leakage(labeled_data, banned_pids)
    
    # 3. TEACHER
    model = SimpleCNN().to(device)
    train_loop(model, labeled_data, device, epochs=50, lr=args.lr, patience=args.patience, phase_name="Teacher")

    # 4. STREAMING UNLABELED DATA
    all_folds = [i for i in range(0, 62, 2)]
    exclude_folds = set(labeled_folds + [10])
    unlabeled_folds = [f for f in all_folds if f not in exclude_folds]
    
    # This new function handles memory efficiently
    pseudo_data = stream_pseudo_labels(model, unlabeled_folds, device, banned_pids, args.data_dir, args.confidence_threshold)
    
    # 5. STUDENT
    full_data = labeled_data + pseudo_data
    print(f"Retraining Student on {len(full_data)} samples...")
    train_loop(model, full_data, device, epochs=100, lr=args.lr/2, patience=args.patience, phase_name="Student")
    
    # 6. TEST
    print("\n--- FINAL TEST FOLD 10 ---")
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
    
    torch.save(model.state_dict(), os.path.join(args.scratch_dir, "simple_sleep_student_final.pt"))
