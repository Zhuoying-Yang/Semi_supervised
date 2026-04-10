import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import gc
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
import copy

# For FLOPs calculation
try:
    from thop import profile
except ImportError:
    profile = None

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/zhuoying/projects/def-xilinliu/data/extracted_data_2ch")
    parser.add_argument("--save_dir", type=str, default="/scratch/zhuoying/sleep_results/baseline")
    parser.add_argument("--seed", type=int, default=40) 
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seq_len", type=int, default=15) 
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()

# --- MODEL ARCHITECTURE --- 
class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.stream_l = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=64, stride=8, padding=32),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(4, 4),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.stream_s = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(4, 4),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
    def forward(self, x):
        return torch.cat([self.stream_l(x).flatten(1), self.stream_s(x).flatten(1)], dim=1)

class CNNOnlySleepNet(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.cnn_backbone = MultiScaleCNN()
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512), nn.ReLU()
        )
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        batch, seq, ch, time = x.size()
        x = x.view(batch * seq, ch, time)
        feats = self.cnn_backbone(x)
        feats = feats.view(batch, seq, 256).transpose(1, 2)
        out = self.temporal_conv(feats)
        return self.fc(out.transpose(1, 2))

# --- DATA UTILS ---
class SeqDataset(Dataset):
    def __init__(self, seqs): self.data = seqs
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx][0].float(), self.data[idx][1].long(), torch.tensor(self.data[idx][2], dtype=torch.long)

def create_sequences(data_list, seq_len=15):
    if not data_list: return []
    sequences = []
    data_list.sort(key=lambda x: int(x[3])) 
    indexed_data = [list(item) + [i] for i, item in enumerate(data_list)]
    cur_x, cur_y, cur_idx, last_pid = [], [], [], -1
    for item in indexed_data:
        pid = int(item[3])
        if pid != last_pid and last_pid != -1: cur_x, cur_y, cur_idx = [], [], []
        cur_x.append(torch.stack([item[0], item[1]])); cur_y.append(item[2]); cur_idx.append(item[4]); last_pid = pid
        if len(cur_x) == seq_len:
            sequences.append((torch.stack(cur_x), torch.tensor(cur_y, dtype=torch.long), np.array(cur_idx)))
            cur_x.pop(0); cur_y.pop(0); cur_idx.pop(0)
    return sequences

# --- EVALUATION ---
def evaluate(model, loader, device, seq_len):
    model.eval()
    segment_agg, segment_gt = defaultdict(lambda: torch.zeros(5).to(device)), {}
    with torch.no_grad():
        for vx, vy, v_idx in loader:
            vx = vx.to(device)
            probs = F.softmax(model(vx), dim=-1)
            for b in range(vx.size(0)):
                for s in range(seq_len):
                    g_idx = v_idx[b, s].item()
                    segment_agg[g_idx] += probs[b, s]
                    segment_gt[g_idx] = vy[b, s].item()
    y_true = [segment_gt[i] for i in sorted(segment_gt.keys())]
    y_pred = [torch.argmax(segment_agg[i]).item() for i in sorted(segment_gt.keys())]
    return np.mean(np.array(y_true) == np.array(y_pred)), y_true, y_pred

# --- RUNNER ---
def run_fold(target_fold, args, device):
    print(f"\n--- BASELINE (10% SUPERVISED ONLY) FOLD: {target_fold} ---")
    set_seed(args.seed)

    # 1. Load and Split (80% Train Patients, 20% Val Patients)
    train_obj = torch.load(os.path.join(args.data_dir, str(target_fold), "train_set.pt"), weights_only=False)
    train_list = [train_obj[i] for i in range(len(train_obj))]
    
    all_train_pids = sorted(list(set([int(x[3]) for x in train_list])))
    random.Random(args.seed).shuffle(all_train_pids)
    val_pids = all_train_pids[:int(len(all_train_pids) * 0.2)]
    train_pids = all_train_pids[int(len(all_train_pids) * 0.2):]

    # 2. Extract ONLY 10% Labeled Patients for training
    labeled_pids = train_pids[:max(1, int(len(train_pids)*0.1))]
    
    l_seqs = create_sequences([x for x in train_list if int(x[3]) in labeled_pids], args.seq_len)
    v_seqs = create_sequences([x for x in train_list if int(x[3]) in val_pids], args.seq_len)
    
    test_obj = torch.load(os.path.join(args.data_dir, str(target_fold), "val_set.pt"), weights_only=False)
    t_seqs = create_sequences([test_obj[i] for i in range(len(test_obj))], args.seq_len)
    del train_obj, train_list, test_obj; gc.collect()

    l_loader = DataLoader(SeqDataset(l_seqs), batch_size=16, shuffle=True)
    val_loader = DataLoader(SeqDataset(v_seqs), batch_size=32)
    test_loader = DataLoader(SeqDataset(t_seqs), batch_size=32)

    model = CNNOnlySleepNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=3e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs, T_mult=1)

    best_val_acc, best_path = 0.0, os.path.join(args.save_dir, f"best_baseline_fold_{target_fold}.pth")

    # 3. Training Loop (Supervised Only)
    for epoch in range(args.epochs):
        model.train()
        for x, y, _ in l_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            # Simple Cross Entropy (No Mixup, No SSL)
            loss = F.cross_entropy(model(x).view(-1, 5), y.view(-1), label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        val_acc, _, _ = evaluate(model, val_loader, device, args.seq_len)
        print(f"Epoch {epoch+1:02d} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
        torch.cuda.empty_cache()

    # 4. Final Evaluation
    model.load_state_dict(torch.load(best_path))
    test_acc, y_true, y_pred = evaluate(model, test_loader, device, args.seq_len)
    print(f"\n--- FINAL BASELINE TEST FOLD {target_fold} ---")
    print(f"Accuracy: {test_acc:.4f}\nConfusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    print(classification_report(y_true, y_pred, target_names=['W', 'N1', 'N2', 'N3', 'REM']))

if __name__ == "__main__":
    args = get_args()
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_folds = sorted([int(f) for f in os.listdir(args.data_dir) if f.isdigit()])
    for f in all_folds: run_fold(f, args, device)
