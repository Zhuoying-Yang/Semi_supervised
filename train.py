import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import copy
import gc
import random

# --- CONFIGURATION ---
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/zhuoying/projects/def-xilinliu/data/extracted_data_2ch")
    parser.add_argument("--scratch_dir", type=str, default=os.getenv("SCRATCH", "./"))
    parser.add_argument("--target_fold", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--confidence_threshold", type=float, default=0.92) 
    parser.add_argument("--patience", type=int, default=12)
    return parser.parse_args()

# --- MODEL: DEEP CONTEXTUAL SLEEPNET ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class DeepContextualSleepNet(nn.Module):
    def __init__(self, in_channels=6, n_classes=5): 
        super(DeepContextualSleepNet, self).__init__()
        self.norm = nn.InstanceNorm1d(in_channels, affine=True)
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResidualBlock(64, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.rnn = nn.GRU(512, 256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        x = torch.clamp(self.norm(x), -10, 10) # Added clipping to stabilize
        x = self.features(x).squeeze(-1).unsqueeze(1) 
        x, _ = self.rnn(x)
        x = self.dropout(x.squeeze(1))
        return self.fc(x)

# --- DATASET & SAMPLING HELPERS ---
def create_contextual_data(data_list):
    print("Creating Contextual Windows...")
    context_data = []
    for i in range(1, len(data_list) - 1):
        prev, curr, nxt = data_list[i-1], data_list[i], data_list[i+1]
        get_pid = lambda d: int(d[3].item() if isinstance(d[3], torch.Tensor) else d[3])
        if get_pid(prev) == get_pid(curr) == get_pid(nxt):
            x = torch.cat([prev[0], prev[1], curr[0], curr[1], nxt[0], nxt[1]], dim=0)
            context_data.append((x, curr[2], get_pid(curr)))
    return context_data

class ContextDataset(Dataset):
    def __init__(self, data_list): self.data = data_list
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        x, label, pid = self.data[idx]
        return x.view(6, -1), int(label.item() if isinstance(label, torch.Tensor) else label)

def get_sampler(data_list):
    """Calculates weights for WeightedRandomSampler to balance batches."""
    labels = [int(x[1]) for x in data_list]
    class_count = np.array([labels.count(i) for i in range(5)])
    class_count = np.where(class_count == 0, 1, class_count) # Avoid div by zero
    weight = 1. / class_count
    samples_weight = np.array([weight[t] for t in labels])
    samples_weight = torch.from_numpy(samples_weight)
    return WeightedRandomSampler(samples_weight, len(samples_weight))

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + ( lam if random.random() > 0.5 else (1 - lam) ) * x[index, :] # Tweak for stability
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# --- TRAINING LOOP ---
def train_phase(model, train_data, val_data, device, epochs, lr, patience, name, use_mixup=False):
    # Use Sampler to balance the batches
    sampler = get_sampler(train_data)
    train_loader = DataLoader(ContextDataset(train_data), batch_size=128, sampler=sampler, num_workers=4)
    val_loader = DataLoader(ContextDataset(val_data), batch_size=128, shuffle=False, num_workers=4)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Removed fixed weight as Sampler handles it
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc, wait = 0.0, 0
    best_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if use_mixup:
                inputs, y_a, y_b, lam = mixup_data(x, y)
                outputs = model(inputs)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                loss = criterion(model(x), y)
            loss.backward(); optimizer.step()
        scheduler.step()
        
        model.eval(); correct, total = 0, 0
        with torch.no_grad():
            for vx, vy in val_loader:
                correct += (model(vx.to(device)).argmax(1) == vy.to(device)).sum().item()
                total += vy.size(0)
        
        acc = correct / total
        print(f"[{name}] Ep {epoch+1} | Val Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc; wait = 0; best_wts = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_wts)

# --- MAIN ---
if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_train_obj = torch.load(os.path.join(args.data_dir, str(args.target_fold), "train_set.pt"), weights_only=False)
    raw_train = [raw_train_obj[i] for i in range(len(raw_train_obj))]; raw_train.sort(key=lambda x: int(x[3]))
    context_train = create_contextual_data(raw_train)
    del raw_train; gc.collect()

    labeled = [x for x in context_train if int(x[2]) in range(10)]
    unlabeled = [x for x in context_train if int(x[2]) not in range(10)]

    # 1. Teacher
    model = DeepContextualSleepNet().to(device)
    random.shuffle(labeled); split = int(0.85 * len(labeled))
    train_phase(model, labeled[:split], labeled[split:], device, 40, args.lr, args.patience, "Teacher")

    # 2. Pseudo Labeling with Very Lenient Minority Thresholds
    model.eval(); bins = {i: [] for i in range(5)}
    print(f"\n--- Phase 2: Generating Pseudo Labels ---")
    with torch.no_grad():
        for i in range(0, len(unlabeled), 128):
            batch = unlabeled[i:i+128]
            x_b = torch.stack([b[0].view(6, -1) for b in batch]).to(device)
            probs = F.softmax(model(x_b), dim=1)
            conf, preds = torch.max(probs, dim=1)
            for j in range(len(batch)):
                label = preds[j].item()
                # Lenient Thresholds for N1 (1) and N3 (3) to get more data
                thresh = args.confidence_threshold
                if label == 1: thresh -= 0.20
                if label == 3: thresh -= 0.10
                if label == 2: thresh -= 0.05
                
                if conf[j].item() >= thresh:
                    bins[label].append((batch[j][0], label, batch[j][2]))

    final_pseudo = []
    for c in range(5):
        available = len(bins[c])
        # If we have very few pseudo-labels for a class, oversample them
        if 0 < available < 500:
            bins[c] = bins[c] * (500 // available)
        
        sample_size = min(len(bins[c]), 4000) 
        if sample_size > 0: final_pseudo += random.sample(bins[c], sample_size)
        print(f"  Class {c}: harvested {available}, using {len(final_pseudo[-sample_size:])} (w/ oversampling)")
    
    # 3. Student
    print("\n--- Phase 3: Student with Balanced Sampler ---")
    combined = labeled + final_pseudo
    random.shuffle(combined); split = int(0.85 * len(combined))
    train_phase(model, combined[:split], combined[split:], device, args.epochs, args.lr * 0.5, args.patience, "Student", use_mixup=True)

    # 4. Final Evaluation
    raw_test_obj = torch.load(os.path.join(args.data_dir, str(args.target_fold), "val_set.pt"), weights_only=False)
    raw_test = [raw_test_obj[i] for i in range(len(raw_test_obj))]; raw_test.sort(key=lambda x: int(x[3])); context_test = create_contextual_data(raw_test)
    test_loader = DataLoader(ContextDataset(context_test), batch_size=128, shuffle=False)
    model.eval(); all_p, all_y = [], []
    with torch.no_grad():
        for x, y in test_loader:
            all_p.extend(model(x.to(device)).argmax(1).cpu().tolist()); all_y.extend(y.tolist())
    print(classification_report(all_y, all_p, digits=4)); print(confusion_matrix(all_y, all_p))
