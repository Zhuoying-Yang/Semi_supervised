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
    parser.add_argument("--save_path", type=str, default="final_sota_model.pth")
    parser.add_argument("--target_fold", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--burn_in", type=int, default=20) # Longer burn-in to fully saturate supervised learning
    parser.add_argument("--lr", type=float, default=5e-5)
    return parser.parse_args()

# --- AUGMENTATIONS ---
def weak_aug(x): return x + torch.randn_like(x) * 0.005
def strong_aug(x):
    x = x.clone()
    # Random masking: Set small segments to zero to force feature recovery
    if random.random() > 0.5:
        mask_len = random.randint(50, 200)
        start = random.randint(0, x.shape[-1] - mask_len)
        x[:, start:start+mask_len] = 0
    return x * random.uniform(0.8, 1.2)

# --- MODEL ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride), nn.BatchNorm1d(out_channels))
    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

class SemiSleepNet(nn.Module):
    def __init__(self, in_channels=6, n_classes=5):
        super().__init__()
        self.norm = nn.InstanceNorm1d(in_channels, affine=True)
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            ResidualBlock(64, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.rnn = nn.GRU(256, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, n_classes)
    def forward(self, x):
        x = torch.clamp(self.norm(x), -10, 10)
        x = self.features(x).squeeze(-1).unsqueeze(1)
        x, _ = self.rnn(x)
        return self.fc(x.squeeze(1))

# --- DATA ---
class ContextDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        x, label, _ = self.data[idx]
        return x.view(6, -1), int(label)

def create_context(data_list):
    res = []
    data_list.sort(key=lambda x: int(x[3]))
    for i in range(1, len(data_list)-1):
        p, c, n = data_list[i-1], data_list[i], data_list[i+1]
        if int(p[3]) == int(c[3]) == int(n[3]):
            x = torch.cat([p[0], p[1], c[0], c[1], n[0], n[1]], 0)
            res.append((x, c[2], c[3]))
    return res

if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_obj = torch.load(os.path.join(args.data_dir, str(args.target_fold), "train_set.pt"), weights_only=False)
    ctx_train = create_context([train_obj[i] for i in range(len(train_obj))])
    del train_obj; gc.collect()
    
    labeled = [x for x in ctx_train if int(x[2]) < 10]
    unlabeled = [x for x in ctx_train if int(x[2]) >= 10]
    l_loader = DataLoader(ContextDataset(labeled), batch_size=32, shuffle=True)
    u_loader = DataLoader(ContextDataset(unlabeled), batch_size=64, shuffle=True)

    model = SemiSleepNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    best_acc = 0
    print(f"--- Starting Final Distillation Run ---")
    
    for epoch in range(args.epochs):
        model.train()
        total_l, total_u = 0, 0
        u_iter = iter(u_loader)

        for x_l, y_l in l_loader:
            try: x_u, _ = next(u_iter)
            except StopIteration: u_iter = iter(u_loader); x_u, _ = next(u_iter)
            x_l, y_l, x_u = x_l.to(device), y_l.to(device), x_u.to(device)

            # 1. Supervised Loss
            loss_sup = F.cross_entropy(model(x_l), y_l, label_smoothing=0.1)

            # 2. Consistency Distillation (Soft targets instead of hard labels)
            loss_unsup = torch.tensor(0.0).to(device)
            if epoch >= args.burn_in:
                with torch.no_grad():
                    teacher_logits = model(weak_aug(x_u))
                    teacher_probs = F.softmax(teacher_logits / 2.0, dim=1) # Temperature scaling
                
                student_logits = model(strong_aug(x_u))
                # KL Divergence forces student to match teacher's distribution
                loss_unsup = F.kl_div(F.log_softmax(student_logits, dim=1), teacher_probs, reduction='batchmean')

            loss = loss_sup + (0.5 * loss_unsup)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_l += loss.item()

        # Validation
        val_obj = torch.load(os.path.join(args.data_dir, str(args.target_fold), "val_set.pt"), weights_only=False)
        val_loader = DataLoader(ContextDataset(create_context([val_obj[i] for i in range(len(val_obj))])), batch_size=128)
        model.eval(); all_p, all_y = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                all_p.extend(model(vx.to(device)).argmax(1).cpu().tolist()); all_y.extend(vy.tolist())
        
        cur_acc = np.mean(np.array(all_p) == np.array(all_y))
        if cur_acc > best_acc:
            best_acc = cur_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"** NEW BEST: {best_acc:.4f} **")
            
        print(f"Ep {epoch+1} | Total Loss: {total_l/len(l_loader):.3f}")

    print("\n--- FINAL REPORT ---")
    model.load_state_dict(torch.load(args.save_path))
    model.eval(); all_p, all_y = [], []
    with torch.no_grad():
        for vx, vy in val_loader:
            all_p.extend(model(vx.to(device)).argmax(1).cpu().tolist()); all_y.extend(vy.tolist())
    print(classification_report(all_y, all_p, digits=4))
