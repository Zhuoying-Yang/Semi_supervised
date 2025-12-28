import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import random
import gc

# --- 1. THE REPRODUCIBILITY ANCHOR ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensures deterministic algorithms on Narval's NVIDIA GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/zhuoying/projects/def-xilinliu/data/extracted_data_2ch")
    parser.add_argument("--save_path", type=str, default="reproducible_sota.pth")
    parser.add_argument("--target_fold", type=int, default=10)
    parser.add_argument("--seed", type=int, default=40) # Global Seed
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seq_len", type=int, default=21) 
    parser.add_argument("--burn_in", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()

# --- AUGMENTATION: SEQUENCE MIXUP ---
def mixup_seq(x, y, alpha=0.2):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# --- SOTA MODEL ARCHITECTURE ---
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

class SOTASleepNet(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.cnn = MultiScaleCNN()
        self.rnn = nn.LSTM(256, 256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(512, n_classes)
    def forward(self, x):
        batch, seq, ch, time = x.size()
        x = x.view(batch * seq, ch, time)
        feats = self.cnn(x).view(batch, seq, -1)
        out, _ = self.rnn(feats)
        return self.fc(out)

class SeqDataset(Dataset):
    def __init__(self, seqs): self.data = seqs
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx][0].float(), self.data[idx][1].long()

def create_sequences(data_list, seq_len=21):
    sequences = []
    data_list.sort(key=lambda x: int(x[3])) 
    cur_x, cur_y, last_pid = [], [], -1
    for item in data_list:
        pid = int(item[3])
        if pid != last_pid and last_pid != -1: cur_x, cur_y = [], []
        cur_x.append(torch.stack([item[0], item[1]]))
        cur_y.append(item[2])
        last_pid = pid
        if len(cur_x) == seq_len:
            sequences.append((torch.stack(cur_x), torch.tensor(cur_y, dtype=torch.long)))
            cur_x.pop(0); cur_y.pop(0)
    return sequences

# --- MAIN ---
if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed) # Fixed Seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fixed Data Splitting
    train_obj = torch.load(os.path.join(args.data_dir, str(args.target_fold), "train_set.pt"), weights_only=False)
    train_list = [train_obj[i] for i in range(len(train_obj))]
    
    # 2. FIXED PATIENT SPLIT
    pids = sorted(list(set([int(x[3]) for x in train_list])))
    # Deterministic shuffle using a private Random instance
    rng = random.Random(args.seed)
    rng.shuffle(pids)
    labeled_pids = pids[:max(1, int(len(pids)*0.1))]
    print(f"Fixed Labeled PIDs: {labeled_pids}")
    
    l_seqs = create_sequences([x for x in train_list if int(x[3]) in labeled_pids], args.seq_len)
    u_seqs = create_sequences([x for x in train_list if int(x[3]) not in labeled_pids], args.seq_len)
    del train_obj, train_list; gc.collect()

    # Generator for reproducible shuffling in DataLoader
    g = torch.Generator()
    g.manual_seed(args.seed)

    l_loader = DataLoader(SeqDataset(l_seqs), batch_size=16, shuffle=True, generator=g)
    u_loader = DataLoader(SeqDataset(u_seqs), batch_size=16, shuffle=True, generator=g)

    model = SOTASleepNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=3e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1)

    best_acc = 0
    patience = 0
    print(f"--- RUNNING REPRODUCIBLE SOTA (SEED: {args.seed}) ---")
    
    for epoch in range(args.epochs):
        model.train()
        u_iter = iter(u_loader)
        for x_l, y_l in l_loader:
            try: x_u, _ = next(u_iter)
            except: u_iter = iter(u_loader); x_u, _ = next(u_iter)
            x_l, y_l, x_u = x_l.to(device), y_l.to(device), x_u.to(device)

            if random.random() > 0.5:
                mixed_x, ya, yb, lam = mixup_seq(x_l, y_l)
                loss_sup = lam * F.cross_entropy(model(mixed_x).view(-1, 5), ya.view(-1)) + \
                           (1-lam) * F.cross_entropy(model(mixed_x).view(-1, 5), yb.view(-1))
            else:
                loss_sup = F.cross_entropy(model(x_l).view(-1, 5), y_l.view(-1), label_smoothing=0.1)

            loss_unsup = torch.tensor(0.0).to(device)
            if epoch >= args.burn_in:
                with torch.no_grad():
                    t_probs = F.softmax(model(x_u + torch.randn_like(x_u)*0.005) / 1.2, dim=-1)
                loss_unsup = 0.4 * F.kl_div(F.log_softmax(model(x_u), dim=-1), t_probs, reduction='batchmean')

            (loss_sup + loss_unsup).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad()
        
        scheduler.step()

        # Validation
        val_obj = torch.load(os.path.join(args.data_dir, str(args.target_fold), "val_set.pt"), weights_only=False)
        val_loader = DataLoader(SeqDataset(create_sequences([val_obj[i] for i in range(len(val_obj))], args.seq_len)), batch_size=32)
        model.eval(); all_p, all_y = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                all_p.extend(model(vx.to(device)).argmax(-1).view(-1).cpu().tolist())
                all_y.extend(vy.view(-1).tolist())
        
        acc = np.mean(np.array(all_p) == np.array(all_y))
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.save_path)
            print(f"** BEST ACC: {best_acc:.4f} **")
            patience = 0
        else:
            patience += 1
        
        if patience > 20: break
            
        print(f"Epoch {epoch+1} | Acc: {acc:.4f}")

    # Final Report using best model
    model.load_state_dict(torch.load(args.save_path))
    model.eval()
    fp, fy = [], []
    with torch.no_grad():
        for vx, vy in val_loader:
            fp.extend(model(vx.to(device)).argmax(-1).view(-1).cpu().tolist())
            fy.extend(vy.view(-1).tolist())
    print("\nFinal Classification Report (Reproducible):")
    print(classification_report(fy, fp, digits=4, target_names=['Wake', 'N1', 'N2', 'N3', 'REM']))
