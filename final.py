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
    print("Please install thop (pip install thop) for FLOPs calculation.")
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
    parser.add_argument("--save_dir", type=str, default="/scratch/zhuoying/sleep_results")
    parser.add_argument("--seed", type=int, default=40) 
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seq_len", type=int, default=15) 
    parser.add_argument("--burn_in", type=int, default=8)
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

# --- UTILS FOR MODEL STATS ---
def report_model_stats(model, device, seq_len):
    # Dummy input: [Batch=1, Seq=seq_len, Ch=2, Time=3000 (standard sleep epoch)]
    dummy_input = torch.randn(1, seq_len, 2, 3000).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    
    print(f"\n--- MODEL STATS ---")
    print(f"Total Parameters: {params:.2f}M")
    
    if profile:
        flops, _ = profile(copy.deepcopy(model), inputs=(dummy_input,), verbose=False)
        print(f"Total FLOPs: {flops / 1e9:.2f} GFLOPS")
    print("-------------------\n")

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
    segment_agg = defaultdict(lambda: torch.zeros(5).to(device))
    segment_gt = {}
    
    with torch.no_grad():
        for vx, vy, v_idx in loader:
            vx = vx.to(device)
            logits = model(vx)
            probs = F.softmax(logits, dim=-1)
            for b in range(vx.size(0)):
                for s in range(seq_len):
                    g_idx = v_idx[b, s].item()
                    segment_agg[g_idx] += probs[b, s]
                    segment_gt[g_idx] = vy[b, s].item()
    
    y_true, y_pred = [], []
    for g_idx in sorted(segment_gt.keys()):
        y_true.append(segment_gt[g_idx])
        y_pred.append(torch.argmax(segment_agg[g_idx]).item())
    
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    return acc, y_true, y_pred

# --- RUNNER ---
def run_fold(target_fold, args, device):
    print(f"\n{'='*40}\nSTARTING FOLD: {target_fold}\n{'='*40}")
    set_seed(args.seed)

    # Load Training Data
    train_path = os.path.join(args.data_dir, str(target_fold), "train_set.pt")
    train_obj = torch.load(train_path, weights_only=False)
    train_list = [train_obj[i] for i in range(len(train_obj))]
    
    # 1. Patient-Level Split (80% Train, 20% Val)
    all_train_pids = sorted(list(set([int(x[3]) for x in train_list])))
    random.Random(args.seed).shuffle(all_train_pids)
    
    split_point = int(len(all_train_pids) * 0.2)
    val_pids = all_train_pids[:split_point]
    actual_train_pids = all_train_pids[split_point:]

    # 2. SSL Logic within the Training PIDs (10% Labeled)
    labeled_pids = actual_train_pids[:max(1, int(len(actual_train_pids)*0.1))]
    
    # Create Sequences
    l_seqs = create_sequences([x for x in train_list if int(x[3]) in labeled_pids], args.seq_len)
    u_seqs = create_sequences([x for x in train_list if int(x[3]) in actual_train_pids and int(x[3]) not in labeled_pids], args.seq_len)
    v_seqs = create_sequences([x for x in train_list if int(x[3]) in val_pids], args.seq_len)

    # Load Advisor's Test Data (Original val_set.pt)
    test_obj = torch.load(os.path.join(args.data_dir, str(target_fold), "val_set.pt"), weights_only=False)
    test_list = [test_obj[i] for i in range(len(test_obj))]
    t_seqs = create_sequences(test_list, args.seq_len)

    # Free Memory Immediately
    del train_obj, train_list, test_obj, test_list; gc.collect()

    l_loader = DataLoader(SeqDataset(l_seqs), batch_size=16, shuffle=True)
    u_loader = DataLoader(SeqDataset(u_seqs), batch_size=16, shuffle=True)
    val_loader = DataLoader(SeqDataset(v_seqs), batch_size=32)
    test_loader = DataLoader(SeqDataset(t_seqs), batch_size=32)

    # Model Setup
    model = CNNOnlySleepNet().to(device)
    if target_fold == 0: report_model_stats(model, device, args.seq_len)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=3e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs, T_mult=1)

    # Checkpointing Logic
    best_val_acc = 0.0
    best_model_path = os.path.join(args.save_dir, f"best_model_fold_{target_fold}.pth")

    for epoch in range(args.epochs):
        model.train()
        u_iter = iter(u_loader)
        
        for x_l, y_l, _ in l_loader:
            try: x_u, _, _ = next(u_iter)
            except StopIteration: u_iter = iter(u_loader); x_u, _, _ = next(u_iter)
            
            x_l, y_l, x_u = x_l.to(device), y_l.to(device), x_u.to(device)
            optimizer.zero_grad()

            # Supervised Loss
            if random.random() > 0.5:
                mixed_x, ya, yb, lam = mixup_seq(x_l, y_l)
                logits = model(mixed_x).view(-1, 5)
                loss_sup = lam * F.cross_entropy(logits, ya.view(-1)) + (1-lam) * F.cross_entropy(logits, yb.view(-1))
            else:
                loss_sup = F.cross_entropy(model(x_l).view(-1, 5), y_l.view(-1), label_smoothing=0.1)

            # Unsupervised Loss (SSL)
            loss_unsup = torch.tensor(0.0).to(device)
            if epoch >= args.burn_in:
                ramp = min(0.2, 0.2 * (epoch - args.burn_in + 1) / (args.epochs - args.burn_in))
                with torch.no_grad():
                    t_probs = F.softmax(model(x_u + torch.randn_like(x_u)*0.005) / 0.5, dim=-1)
                loss_unsup = ramp * F.kl_div(F.log_softmax(model(x_u), dim=-1), t_probs, reduction='batchmean')

            (loss_sup + loss_unsup).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Validation at Epoch end
        val_acc, _, _ = evaluate(model, val_loader, device, args.seq_len)
        print(f"Fold {target_fold} | Epoch {epoch+1:02d} | Val Acc: {val_acc:.4f}")

        # Save Checkpoint (Overwrites existing file to save space)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  >> Saved New Best Model (Acc: {best_val_acc:.4f})")
        
        # Clear CUDA Cache to avoid OOM
        torch.cuda.empty_cache()

    # --- FINAL TEST ON BEST CHECKPOINT ---
    print(f"\nFinal Testing Fold {target_fold} with Best Checkpoint...")
    model.load_state_dict(torch.load(best_model_path))
    test_acc, y_true, y_pred = evaluate(model, test_loader, device, args.seq_len)
    
    print(f"\n--- FOLD {target_fold} FINAL TEST RESULTS ---")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['W', 'N1', 'N2', 'N3', 'REM']))
    
    # Optional: Delete model to free RAM for next fold
    del model, l_loader, u_loader, val_loader, test_loader; gc.collect()

if __name__ == "__main__":
    args = get_args()
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_folds = sorted([int(f) for f in os.listdir(args.data_dir) if f.isdigit()])
    for f in all_folds:
        run_fold(f, args, device)
