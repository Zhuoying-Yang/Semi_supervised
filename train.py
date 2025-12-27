import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
import numpy as np
import random
import gc

# --- CONFIGURATION ---
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/zhuoying/projects/def-xilinliu/data/extracted_data_2ch")
    parser.add_argument("--save_path", type=str, default="sota_final_model.pth")
    parser.add_argument("--target_fold", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seq_len", type=int, default=21) 
    parser.add_argument("--burn_in", type=int, default=10) # Reduced to start semi-sup earlier
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()

# --- SOTA TWO-STREAM CNN ---
class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        # Stream 1: Large kernels for Delta/Slow waves (N3)
        self.stream_large = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=50, stride=6, padding=24),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(8, 8),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        # Stream 2: Small kernels for Spindles/Beta waves (REM/Wake)
        self.stream_small = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=6, stride=1, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(4, 4),
            nn.Conv1d(64, 128, kernel_size=6, stride=1, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        out_l = self.stream_large(x).flatten(1)
        out_s = self.stream_small(x).flatten(1)
        return torch.cat([out_l, out_s], dim=1) # 256 total features

class Seq2SeqSleepNet(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.cnn = MultiScaleCNN()
        self.rnn = nn.GRU(256, 256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        batch, seq, ch, time = x.size()
        x = x.view(batch * seq, ch, time)
        feats = self.cnn(x).view(batch, seq, -1)
        out, _ = self.rnn(feats)
        return self.fc(out)

# --- DATA HELPERS ---
class SeqDataset(Dataset):
    def __init__(self, seqs): self.data = seqs
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x.float(), y.long()

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_obj = torch.load(os.path.join(args.data_dir, str(args.target_fold), "train_set.pt"), weights_only=False)
    train_list = [train_obj[i] for i in range(len(train_obj))]
    
    pids = sorted(list(set([int(x[3]) for x in train_list])))
    random.shuffle(pids)
    labeled_pids = pids[:max(1, int(len(pids)*0.1))]
    
    l_seqs = create_sequences([x for x in train_list if int(x[3]) in labeled_pids], args.seq_len)
    u_seqs = create_sequences([x for x in train_list if int(x[3]) not in labeled_pids], args.seq_len)
    del train_obj, train_list; gc.collect()

    l_loader = DataLoader(SeqDataset(l_seqs), batch_size=16, shuffle=True)
    u_loader = DataLoader(SeqDataset(u_seqs), batch_size=16, shuffle=True)

    model = Seq2SeqSleepNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    # Added Cosine Scheduler to prevent the accuracy drop at later epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc, patience_counter = 0, 0
    for epoch in range(args.epochs):
        model.train()
        u_iter = iter(u_loader)
        for x_l, y_l in l_loader:
            try: x_u, _ = next(u_iter)
            except: u_iter = iter(u_loader); x_u, _ = next(u_iter)
            x_l, y_l, x_u = x_l.to(device), y_l.to(device), x_u.to(device)

            logits_l = model(x_l)
            loss_sup = F.cross_entropy(logits_l.view(-1, 5), y_l.view(-1), label_smoothing=0.1)

            loss_unsup = torch.tensor(0.0).to(device)
            if epoch >= args.burn_in:
                # Dynamic Unsupervised weight that slowly ramps up
                unsup_w = min(1.0, (epoch - args.burn_in) / 10)
                with torch.no_grad():
                    t_probs = F.softmax(model(x_u + torch.randn_like(x_u)*0.005) / 2.0, dim=-1)
                loss_unsup = unsup_w * F.kl_div(F.log_softmax(model(x_u), dim=-1), t_probs, reduction='batchmean')

            (loss_sup + loss_unsup).backward()
            optimizer.step(); optimizer.zero_grad()
        
        scheduler.step()
        
        # Validation Logic
        if (epoch + 1) % 2 == 0:
            val_obj = torch.load(os.path.join(args.data_dir, str(args.target_fold), "val_set.pt"), weights_only=False)
            val_loader = DataLoader(SeqDataset(create_sequences([val_obj[i] for i in range(len(val_obj))], args.seq_len)), batch_size=32)
            model.eval(); all_p, all_y = [], []
            with torch.no_grad():
                for vx, vy in val_loader:
                    all_p.extend(model(vx.to(device)).argmax(-1).view(-1).cpu().tolist())
                    all_y.extend(vy.view(-1).tolist())
            acc = np.mean(np.array(all_p) == np.array(all_y))
            if acc > best_acc:
                best_acc = acc; torch.save(model.state_dict(), args.save_path)
                print(f"** NEW BEST: {best_acc:.4f} **")
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter > 12: break # Early stopping
        print(f"Epoch {epoch+1} | LR: {optimizer.param_groups[0]['lr']:.6f}")
