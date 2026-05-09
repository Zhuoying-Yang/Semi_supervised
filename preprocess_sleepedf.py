import os
import glob
import torch
import mne
import numpy as np

# Suppress MNE verbosity for cleaner logs
mne.set_log_level('ERROR')

def preprocess_sleepedf():
    data_dir = "/home/zhuoying/projects/def-xilinliu/data/sleepedf/sleep-cassette"
    save_path = "/home/zhuoying/projects/def-xilinliu/data/sleepedf_pretrain.pt"
    
    # Sleep-EDF AASM Stage Mapping
    # Merges Stage 3 and 4 into N3 (Label 3). Ignores Movement and Unknown.
    annot_mapping = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,
        'Sleep stage R': 4
    }

    # Find all PSG files
    psg_files = glob.glob(os.path.join(data_dir, "*PSG.edf"))
    psg_files.sort()
    
    all_data_list = []
    
    print(f"Found {len(psg_files)} PSG files. Starting extraction...")
    
    for i, psg_path in enumerate(psg_files):
        # The hypnogram file shares the first 7 characters (e.g., SC4001E)
        basename = os.path.basename(psg_path)
        subject_id = basename[:7]
        pid = int(basename[2:6]) # Extracts e.g., 4001 from SC4001E0
        
        hyp_path = glob.glob(os.path.join(data_dir, f"{subject_id}*Hypnogram.edf"))
        if not hyp_path:
            continue
        hyp_path = hyp_path[0]
        
        try:
            # Load raw data and annotations
            raw = mne.io.read_raw_edf(psg_path, preload=True)
            annot = mne.read_annotations(hyp_path)
            raw.set_annotations(annot)
            
            # Pick only the two EEG channels your model expects
            raw.pick_channels(['EEG Fpz-Cz', 'EEG Pz-Oz'])
            
            # Chunk the annotations into 30-second events
            events, _ = mne.events_from_annotations(raw, event_id=annot_mapping, chunk_duration=30.0)
            
            # Create 30s epochs. tmax=29.99 ensures exactly 3000 samples at 100Hz.
            epochs = mne.Epochs(raw, events, tmin=0, tmax=29.99, baseline=None, preload=True)
            
            data = epochs.get_data() # Shape: (n_epochs, 2, 3000)
            labels = epochs.events[:, -1]
            
            # Format to match your MASS SS3 sequence creator:
            # item = [ch1_tensor, ch2_tensor, label, pid]
            for j in range(len(epochs)):
                ch1 = torch.tensor(data[j, 0, :], dtype=torch.float32)
                ch2 = torch.tensor(data[j, 1, :], dtype=torch.float32)
                label = int(labels[j])
                all_data_list.append((ch1, ch2, label, pid))
                
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(psg_files)} subjects...")
                
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")

    print(f"Finished extracting! Total 30-second epochs: {len(all_data_list)}")
    print(f"Saving to {save_path} ...")
    torch.save(all_data_list, save_path)
    print("Done!")

if __name__ == "__main__":
    preprocess_sleepedf()
