import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURATION & ARGS ---
def get_args():
    parser = argparse.ArgumentParser(description="Narval Sleep Staging Job")
    parser.add_argument("--data_dir", type=str, 
                        default="/home/zhuoying/projects/def-xilinliu/data/extracted_data_2ch",
                        help="Path to the data directory")
    parser.add_argument("--scratch_dir", type=str, 
                        default=os.getenv("SCRATCH", "./"),
                        help="Path to save models")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()

# --- SANITY CHECKS (CRITICAL) ---
def perform_sanity_checks(labeled_data, test_set):
    print("\n" + "="*30)
    print("RUNNING DATA INTEGRITY CHECKS")
    print("="*30)

    # 1. Check for Patient Overlap (Subject Leakage)
    train_patients = set([item[3] for item in labeled_data])
    test_patients = set([item[3] for item in test_set])
    overlap = train_patients.intersection(test_patients)
    
    if overlap:
        print(f"CRITICAL WARNING: SUBJECT LEAKAGE DETECTED!")
        print(f"Patients found in both Train and Test: {overlap}")
        print("Result: Accuracy will be artificially high (near 100%).")
    else:
        print("Subject Split Check: PASSED (No patient overlap)")

    # 2. Check for Label Leakage in Channel 2
    # We check if Channel 2 is a flat line (constant value) which often means it's a mask/label
    suspicious_samples = 0
    for i in range(min(100, len(labeled_data))):
        ch2 = labeled_data[i][1]
        if torch.std(ch2) < 1e-4: # If standard deviation is near 0
            suspicious_samples += 1
            
    if suspicious_samples > 10:
        print(f"CRITICAL WARNING: SIGNAL LEAKAGE SUSPECTED!")
        print(f"Channel 2 has zero variance in {suspicious_samples}% of checked samples.")
        print("It might be a hardcoded label or artifact mask.")
    else:
        print("Signal Variance Check: PASSED (Channels look like EEG)")

    # 3. Check for Trivial Test Set (Class Imbalance)
    test_labels = [item[2] for item in test_set]
    unique_labels = set(test_labels)
    if len(unique_labels) == 1:
        print(f" WARNING: TRIVIAL TEST SET")
        print(f"Test set only contains Class {list(unique_labels)[0]}.")
        print("Model can get 100% by guessing one class.")
    else:
        print(f"Class Diversity Check: PASSED (Test set has {len(unique_labels)} classes)")
    print("="*30 + "\n")

# --- MODEL DEFINITION ---
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=2, n_classes=5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)
        
        # ADDED: Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)
        
        self.fc1 = nn.Linear(64 * 1920, 128) 
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x) # Dropout active during training
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# --- UTILS ---
def list_to_dataset(data_list):
    """Efficiently converts list of tuples to TensorDataset."""
    # data_list items: (ch1, ch2, label, patient_no)
    
    # Pre-allocate tensors for speed
    n_samples = len(data_list)
    c_len = data_list[0][0].shape[0] # Assuming 7680
    
    x_tensor = torch.zeros((n_samples, 2, c_len))
    y_tensor = torch.zeros((n_samples), dtype=torch.long)
    
    for i, (ch1, ch2, label, _) in enumerate(data_list):
        x_tensor[i, 0, :] = ch1
        x_tensor[i, 1, :] = ch2
        y_tensor[i] = label
        
    return TensorDataset(x_tensor, y_tensor)

def load_labeled_data(folds, base_path):
    data = []
    print(f"Loading folds: {folds}...")
    for fold in folds:
        path = os.path.join(base_path, str(fold), "train_set.pt")
        if os.path.exists(path):
            train_set = torch.load(path, weights_only=False) # Weights_only=False for legacy files
            data.extend(train_set)
        else:
            print(f"Warning: Path not found {path}")
    print(f"Loaded {len(data)} samples.")
    return data

def evaluate_model(model, test_dataset, device):
    model.eval()
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds = torch.argmax(out, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(y.tolist())

    print("\n--- FINAL EVALUATION ---")
    print(classification_report(all_labels, all_preds, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    return all_preds

# --- TRAINING LOOP ---
def train_model(model, train_dataset, device, epochs, lr=1e-3):
    # Split train into Train/Val for internal monitoring
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_sub, val_sub = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_sub, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_sub, batch_size=64, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Starting training on {len(train_dataset)} samples ({train_size} train, {val_size} val)")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation step
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_out = model(vx)
                v_pred = torch.argmax(v_out, dim=1)
                val_correct += (v_pred == vy).sum().item()
        
        val_acc = val_correct / len(val_sub)
        avg_loss = total_loss / len(train_loader)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

# --- MAIN ---
if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    labeled_folds = [0, 2, 4, 6, 8]
    test_fold = 10
    
    labeled_data = load_labeled_data(labeled_folds, args.data_dir)
    test_data_raw = torch.load(os.path.join(args.data_dir, str(test_fold), "val_set.pt"), weights_only=False)

    # 2. Run Sanity Checks (Don't skip this!)
    perform_sanity_checks(labeled_data, test_data_raw)

    # 3. Prepare Datasets
    print("Converting data to TensorDatasets...")
    train_dataset = list_to_dataset(labeled_data)
    test_dataset = list_to_dataset(test_data_raw)

    # 4. Init Model
    model = SimpleCNN().to(device)

    # 5. Train
    train_model(model, train_dataset, device, epochs=args.epochs, lr=args.lr)

    # 6. Evaluate
    evaluate_model(model, test_dataset, device)

    # 7. Save
    save_path = os.path.join(args.scratch_dir, "semi_model_CNN_final.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
