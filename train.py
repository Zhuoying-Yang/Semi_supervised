import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_preparation import SleepDataset_2_chan
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, test_set, device):
    model.eval()
    test_loader = DataLoader(test_set, batch_size=64)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for ch1, ch2, label, pno in test_set:
            x = torch.stack([ch1, ch2], dim=0).unsqueeze(0).to(device)
            out = model(x)
            pred = torch.argmax(out, dim=1).cpu().item()
            all_preds.append(pred)
            all_labels.append(label.item())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=2, n_classes=5):  # in_channels=2 for ch1 + ch2
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * 1920, 128)  # 7680/2/2=1920
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

      
def load_labeled_data(folds, base_path):
    data = []
    for fold in folds:
        train_set = torch.load(os.path.join(base_path, str(fold), "train_set.pt"), weights_only=False)
        data.extend(train_set)
    return data

def train_model(model, train_data, device, epochs=10, lr=1e-3):
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss = {total_loss:.4f}")

def generate_pseudo_labels(model, unlabeled_folds, base_path, device, threshold=0.9):
    pseudo_data = []
    model.eval()
    with torch.no_grad():
        for fold in unlabeled_folds:
            val_set = torch.load(os.path.join(base_path, str(fold), "val_set.pt"))
            for ch1, ch2, label, patient_no in val_set:
                x = torch.stack([ch1, ch2], dim=0).unsqueeze(0).to(device)  # [1, 2, 7680]
                out = model(x)
                prob = F.softmax(out, dim=1)
                confidence, pred = torch.max(prob, dim=1)
                if confidence.item() >= threshold:
                    pseudo_data.append((torch.stack([ch1, ch2], dim=0), pred.item()))
    return pseudo_data


def combine_and_retrain(model, labeled_data, pseudo_data, device):
    x_pseudo = torch.stack([x for x, y in pseudo_data])  # shape: [N, 2, 7680]
    y_pseudo = torch.tensor([y for x, y in pseudo_data])
    pseudo_dataset = TensorDataset(x_pseudo, y_pseudo)
    
    converted_labeled = []
    for ch1, ch2, label, _ in labeled_data:
        x = torch.stack([ch1, ch2], dim=0)
        converted_labeled.append((x, label))

    x_labeled = torch.stack([x for x, y in converted_labeled])
    y_labeled = torch.tensor([y for x, y in converted_labeled])
    labeled_dataset = TensorDataset(x_labeled, y_labeled)

    full_x = torch.cat([x_labeled, x_pseudo], dim=0)
    full_y = torch.cat([y_labeled, y_pseudo], dim=0)
    full_dataset = TensorDataset(full_x, full_y)

    train_loader = DataLoader(full_dataset, batch_size=64, shuffle=True)
    # Convert labeled_data to TensorDataset
    converted_labeled = []
    for ch1, ch2, label, _ in labeled_data:
        x = torch.stack([ch1, ch2], dim=0)
        converted_labeled.append((x, label))
    
    x_labeled = torch.stack([x for x, y in converted_labeled])
    y_labeled = torch.tensor([y for x, y in converted_labeled])
    labeled_dataset = TensorDataset(x_labeled, y_labeled)
    
    # Now pass this to train_model
    train_model(model, labeled_dataset, device)


if __name__ == "__main__":
    base_path = "/home/zhuoying/projects/def-xilinliu/data/extracted_data_2ch"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labeled_folds = [0, 2, 4, 6, 8]
    unlabeled_folds = [i for i in range(0, 62, 2) if i not in labeled_folds]

    labeled_data = load_labeled_data(labeled_folds, base_path)
    model = SimpleCNN().to(device)

    print("Step 1: Training initial model")
    train_model(model, labeled_data, device)

    print("Step 2: Generating pseudo labels")
    pseudo_data = generate_pseudo_labels(model, unlabeled_folds, base_path, device)

    print(f"Generated {len(pseudo_data)} pseudo-labeled samples")

    print("Step 3: Retraining on full data")
    combine_and_retrain(model, labeled_data, pseudo_data, device)

    print("Step 4: Evaluate on held-out test fold")
    test_fold = 10  
    test_set = torch.load(os.path.join(base_path, str(test_fold), "val_set.pt"))
    evaluate_model(model, test_set, device)
  
    print("Done!")
