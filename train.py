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

# --- CONFIGURATION ---
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/zhuoying/projects/def-xilinliu/data/extracted_data_2ch")
    parser.add_argument("--save_path", type=str, default="sota_seq2seq_model.pth")
    parser.add_argument("--target_fold", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seq_len", type=int, default=21) 
    parser.add_argument("--burn_in", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()

# --- AUGMENTATIONS ---
def weak_aug(x): 
    return x + torch.randn_like(x) * 0.005

def strong_aug(x):
    x = x.clone()
    if random.random() > 0.3:
        idx = random.randint(0, x.shape[1]-1)
        x[:, idx, :, :] = 0 # Epoch dropout
    return x * random.uniform(0.8, 1.2)

# --- SOTA MODEL: SEQ2SEQ SLEEPNET ---
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.large_filter = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=50, stride=6, padding=24),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8)
        )
        self.conv_block = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.large_filter(x)
        x = self.conv_block(x)
        return self.avgpool(x).flatten(1)

class Seq2SeqSleepNet(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.rnn = nn.GRU(128, 256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        batch_size, seq_len, chans, time_pts = x.size()
        x = x.view(batch_size * seq_len, chans, time_pts)
        features = self.cnn(x)
        features = features.view(batch_size, seq_len, -1)
        out, _ = self.rnn(features)
        return self.fc(out) 

# --- DATASET WRAPPERS ---
class SeqDataset(Dataset):
    def __init__(self, sequences): self.data = sequences
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x.float(), y.long()

def create_sequences(data_list, seq_len=21):
    sequences = []
    data_list.sort(key=lambda x: int(x[3])) 
    current_x, current_y = [], []
    last_pid = -1
    for item in data_list:
        pid = int(item[3])
        if pid != last_pid and last_pid != -1:
            current_x, current_y = [], []
        current_x.append(torch.stack([item[0], item[1]]))
        current_y.append(item[2])
        last_pid = pid
        if len(current_x) == seq_len:
            sequences.append((torch.stack(current_x), torch.tensor(current_y, dtype=torch.long)))
            current_x.pop(0)
            current_y.pop(0)
    return sequences

# --- MAIN ---
if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading raw data...")
    train_obj = torch.load(os.path.join(args.data_dir, str(args.target_fold), "train_set.pt"), weights_only=False)
    train_list = [train_obj[i] for i in range(len(train_obj))]
    
    # SPLIT BY PATIENT ID (Avoids Tensor comparison error)
    pids = sorted(list(set([int(x[3]) for x in train_list])))
    random.shuffle(pids)
    labeled_pids = pids[:max(1, int(len(pids)*0.1))] # 10% Labeled Patients
    
    labeled_list = [x for x in train_list if int(x[3]) in labeled_pids]
    unlabeled_list = [x for x in train_list if int(x[3]) not in labeled_pids]
    
    print(f"Building sequences (Labeled: {len(labeled_pids)} patients)...")
    labeled_seqs = create_sequences(labeled_list, args.seq_len)
    unlabeled_seqs = create_sequences(unlabeled_list, args.seq_len)
    
    del train_obj, train_list, labeled_list, unlabeled_list; gc.collect()

    l_loader = DataLoader(SeqDataset(labeled_seqs), batch_size=16, shuffle=True)
    u_loader = DataLoader(SeqDataset(unlabeled_seqs), batch_size=16, shuffle=True)

    model = Seq2SeqSleepNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        u_iter = iter(u_loader)

        for x_l, y_l in l_loader:
            try: x_u, _ = next(u_iter)
            except StopIteration: u_iter = iter(u_loader); x_u, _ = next(u_iter)
            x_l, y_l, x_u = x_l.to(device), y_l.to(device), x_u.to(device)

            logits_l = model(x_l)
            loss_sup = F.cross_entropy(logits_l.view(-1, 5), y_l.view(-1), label_smoothing=0.1)

            loss_unsup = torch.tensor(0.0).to(device)
            if epoch >= args.burn_in:
                with torch.no_grad():
                    t_probs = F.softmax(model(weak_aug(x_u)) / 2.0, dim=-1)
                s_logits = model(strong_aug(x_u))
                loss_unsup = F.kl_div(F.log_softmax(s_logits, dim=-1), t_probs, reduction='batchmean')

            loss = loss_sup + (0.5 * loss_unsup)
            optimizer.zero_grad(); loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation logic for Sequences
        if (epoch + 1) % 5 == 0:
            val_obj = torch.load(os.path.join(args.data_dir, str(args.target_fold), "val_set.pt"), weights_only=False)
            val_seqs = create_sequences([val_obj[i] for i in range(len(val_obj))], args.seq_len)
            val_loader = DataLoader(SeqDataset(val_seqs), batch_size=32)
            model.eval()
            all_p, all_y = [], []
            with torch.no_grad():
                for vx, vy in val_loader:
                    preds = model(vx.to(device)).argmax(dim=-1).view(-1)
                    all_p.extend(preds.cpu().tolist()); all_y.extend(vy.view(-1).tolist())
            
            acc = np.mean(np.array(all_p) == np.array(all_y))
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), args.save_path)
                print(f"** NEW BEST: {best_acc:.4f} **")
            
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(l_loader):.4f}")
