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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser()
    # Point this to your single .pt file
    parser.add_argument("--data_file", type=str, default="/home/zhuoying/projects/def-xilinliu/data/sleepedf_pretrain.pt")
    parser.add_argument("--save_dir", type=str, default="/scratch/zhuoying/sleep_results")
    parser.add_argument("--seed", type=int, default=42) 
    parser.add_argument("--epochs", type=int, default=20) # Bumped epochs for better pre-training
    parser.add_argument("--seq_len", type=int, default=15) 
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()

# --- MODEL ARCHITECTURE (Same as your MASS pipeline) ---
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
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        batch, seq, ch, time = x.size()
        x = x.view(batch * seq, ch, time)
        feats = self.cnn_backbone(x)
        feats = feats.view(batch, seq, 256).transpose(1, 2)
        out = self.temporal_conv(feats)
        out = out.transpose(1, 2)
        return self.fc(out)

# --- DATA UTILS ---
class SeqDataset(Dataset):
    def __init__(self, seqs): 
        self.data = seqs
    def __len__(self): 
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx][0].float(), self.data[idx][1].long(), torch.tensor(self.data[idx][2], dtype=torch.long)

def create_sequences(data_list, seq_len=15):
    if not data_list: return []
    sequences = []
    data_list.sort(key=lambda x: int(x[3])) 
    indexed_data = [list(item) + [i] for i, item in enumerate(data_list)]
    cur_x, cur_y, cur_idx, last_pid = [], [], [], -1

    for item in indexed_data:
        pid = int(item[3])
        if pid != last_pid and last_pid != -1: 
            cur_x, cur_y, cur_idx = [], [], []
        cur_x.append(torch.stack([item[0], item[1]]))
        cur_y.append(item[2])
        cur_idx.append(item[4])
        last_pid = pid
        if len(cur_x) == seq_len:
            sequences.append((torch.stack(cur_x), torch.tensor(cur_y, dtype=torch.long), np.array(cur_idx)))
            cur_x.pop(0); cur_y.pop(0); cur_idx.pop(0)
    return sequences

def mixup_seq(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam

# --- EVALUATION FUNCTION ---
def evaluate(model, loader, device, seq_len):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for vx, vy, _ in loader:
            vx, vy = vx.to(device), vy.to(device)
            logits = model(vx)
            y_true.extend(vy.view(-1).cpu().numpy())
            y_pred.extend(torch.argmax(logits, dim=-1).view(-1).cpu().numpy())
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    return acc, y_true, y_pred

def run_pretraining(args, device):
    print(f"\n{'='*40}\nSTARTING SLEEP-EDF PRE-TRAINING\n{'='*40}")
    set_seed(args.seed)

    # 1. Load the single large file
    print(f"Loading data from {args.data_file}...")
    full_data = torch.load(args.data_file, weights_only=False)
    
    # 2. Patient-Level Split (90% Train, 10% Val)
    all_pids = sorted(list(set([int(x[3]) for x in full_data])))
    random.Random(args.seed).shuffle(all_pids)
    split_point = int(len(all_pids) * 0.9)
    train_pids = set(all_pids[:split_point])

    train_list = [x for x in full_data if int(x[3]) in train_pids]
    val_list = [x for x in full_data if int(x[3]) not in train_pids]
    del full_data; gc.collect()

    print(f"Train samples: {len(train_list)} | Val samples: {len(val_list)}")

    # 3. Create Sequences
    train_seqs = create_sequences(train_list, args.seq_len)
    val_seqs = create_sequences(val_list, args.seq_len)
    del train_list, val_list; gc.collect()

    train_loader = DataLoader(SeqDataset(train_seqs), batch_size=32, shuffle=True)
    val_loader = DataLoader(SeqDataset(val_seqs), batch_size=64)

    model = CNNOnlySleepNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=3e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs, T_mult=1)

    best_val_acc = 0.0
    best_model_path = os.path.join(args.save_dir, "sleepedf_pretrained_weights.pth")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            if random.random() > 0.5:
                mixed_x, ya, yb, lam = mixup_seq(x, y)
                logits = model(mixed_x).view(-1, 5)
                loss = lam * F.cross_entropy(logits, ya.view(-1)) + (1-lam) * F.cross_entropy(logits, yb.view(-1))
            else:
                loss = F.cross_entropy(model(x).view(-1, 5), y.view(-1), label_smoothing=0.1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        val_acc, y_true, y_pred = evaluate(model, val_loader, device, args.seq_len)
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  >> Saved New Best Weights (Acc: {best_val_acc:.4f})")

if __name__ == "__main__":
    args = get_args()
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_pretraining(args, device)
