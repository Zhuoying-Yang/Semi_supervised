import torch
import os
import numpy as np

# --- CONFIGURATION ---
base_path = "/home/zhuoying/projects/def-xilinliu/data/extracted_data_2ch"
train_folds = [0, 2, 4, 6, 8]
test_fold = 10

print("STARTING DEEP FORENSIC INVESTIGATION (V2)...")

# 1. Load Test Data
print(f"Loading Test Fold {test_fold}...")
test_data = torch.load(os.path.join(base_path, str(test_fold), "val_set.pt"), weights_only=False)

# --- FIX: Convert Tensors to Python Integers for Set Comparison ---
test_patients = set()
for item in test_data:
    pid = item[3]
    if isinstance(pid, torch.Tensor):
        pid = int(pid.item())
    test_patients.add(pid)

print(f" -> Test Patients IDs (Cleaned): {test_patients}")

# ==========================================
# CHECK 1: CORRELATION LEAKAGE (The "Hidden Pulse")
# ==========================================
print("\nCHECK 1: SIGNAL-LABEL CORRELATION")
# We check if the Mean or Std of the signal gives away the label
ch1_means = []
ch2_means = []
labels = []

for i in range(min(500, len(test_data))):
    ch1 = test_data[i][0]
    ch2 = test_data[i][1]
    lbl = test_data[i][2]
    
    ch1_means.append(ch1.mean().item())
    ch2_means.append(ch2.mean().item())
    
    if isinstance(lbl, torch.Tensor): lbl = lbl.item()
    labels.append(lbl)

# Calculate Correlation
corr_ch1 = np.corrcoef(ch1_means, labels)[0, 1]
corr_ch2 = np.corrcoef(ch2_means, labels)[0, 1]

print(f"   Correlation (Ch1 Mean vs Label): {corr_ch1:.4f}")
print(f"   Correlation (Ch2 Mean vs Label): {corr_ch2:.4f}")

if abs(corr_ch1) > 0.8 or abs(corr_ch2) > 0.8:
    print("ALARM: High correlation found! The label is likely embedded in the signal amplitude.")
    print("(e.g., Stage 2 has a higher voltage offset than Stage 1)")
else:
    print("VERDICT: No obvious statistical leakage found.")


# ==========================================
# CHECK 2: ROBUST SUBJECT LEAKAGE
# ==========================================
print("\nCHECK 2: SUBJECT OVERLAP (FIXED)")
overlap_found = False

for fold in train_folds:
    path = os.path.join(base_path, str(fold), "train_set.pt")
    if os.path.exists(path):
        train_part = torch.load(path, weights_only=False)
        
        train_patients = set()
        for item in train_part:
            pid = item[3]
            if isinstance(pid, torch.Tensor):
                pid = int(pid.item())
            train_patients.add(pid)
            
        # Check intersection
        overlap = train_patients.intersection(test_patients)
        if len(overlap) > 0:
            print(f"ALARM: Fold {fold} contains Patients {overlap} which are also in Test Fold 10!")
            overlap_found = True
            break
    else:
        print(f"   (Skipping Fold {fold})")

if not overlap_found:
    print("VERDICT: Patient IDs are unique.")


# ==========================================
# CHECK 3: DEEP DUPLICATE SEARCH
# ==========================================
print("\nCHECK 3: RANDOM SAMPLE MATCHING")
# Instead of checking sample 0, we check 5 random samples from Test
import random
indices_to_check = random.sample(range(len(test_data)), 3)
leak_found = False

for idx in indices_to_check:
    test_sig = test_data[idx][0]
    test_val = test_sig.sum().item()
    
    for fold in train_folds:
        path = os.path.join(base_path, str(fold), "train_set.pt")
        if os.path.exists(path):
            train_part = torch.load(path, weights_only=False)
            # Check first 200 samples of train
            for i in range(min(200, len(train_part))):
                train_val = train_part[i][0].sum().item()
                if abs(train_val - test_val) < 1e-4:
                    print(f"ALARM: Test Sample {idx} found in Fold {fold} at index {i}!")
                    leak_found = True
        if leak_found: break
    if leak_found: break

if not leak_found:
    print("VERDICT: No random duplicates found.")
    
print("\nINVESTIGATION V2 COMPLETE.")
