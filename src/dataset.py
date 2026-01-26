import pickle
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader

class TCGADataset(Dataset):
    """
    Dataset class for TCGA embeddings and cancer type labels.
    Adapted to support patient-aware splits to prevent data leakage.
    """
    def __init__(self, pickle_path: str, csv_path: str, split_map_path: str = None, class_to_idx=None):
        """
        Args:
            pickle_path: Path to the .pkl file with embeddings.
            csv_path: Path to the full labels csv.
            split_map_path: (Optional) Path to a CSV containing ONLY the patient IDs for this split (train/val/test).
            class_to_idx: Dictionary mapping class names to integers (ensures consistency).
        """
        # 1. Load the embeddings
        # print(f"Loading embeddings from {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            self.embedding_data = pickle.load(f)
        
        # 2. Load the full labels dataframe
        self.labels_df = pd.read_csv(csv_path)
        
        # 3. Create or use existing class mapping
        if class_to_idx is None:
            self.cancer_types = sorted(self.labels_df['cancer_type'].unique())
            self.class_to_idx = {name: i for i, name in enumerate(self.cancer_types)}
        else:
            self.class_to_idx = class_to_idx

        # 4. Filter Patients based on the Split (CRITICAL for MLOps)
        # We start with the intersection of patients having both images and labels
        valid_intersection = set(self.embedding_data.keys()).intersection(set(self.labels_df['patient_id']))
        
        if split_map_path and os.path.exists(split_map_path):
            # If a split file is provided, we ONLY keep patients listed there
            split_df = pd.read_csv(split_map_path)
            split_patients = set(split_df['patient_id'])
            
            # The final valid list is: (Exists in PKL) AND (Exists in CSV) AND (Exists in Split)
            self.valid_patient_ids = sorted(list(valid_intersection.intersection(split_patients)))
            print(f"Loaded split from {split_map_path}: {len(self.valid_patient_ids)} samples.")
        else:
            # Fallback: load everything (only for inference/debugging)
            self.valid_patient_ids = sorted(list(valid_intersection))
            print(f"Warning: No split file provided. Loaded all {len(self.valid_patient_ids)} available samples.")
        
        # 5. Map Patient IDs to their cancer type labels for fast lookup
        # We assume one label per patient
        self.id_to_label = dict(zip(self.labels_df['patient_id'], self.labels_df['cancer_type']))

    def __len__(self):
        return len(self.valid_patient_ids)

    def __getitem__(self, idx):    
        pid = self.valid_patient_ids[idx]
        
        # Retrieve the embedding
        # The data structure is dict[pid] -> dict['embeddings'] -> list of arrays
        # We take the first embedding (standard strategy for this assignment)
        embedding_list = self.embedding_data[pid]['embeddings']
        embedding_vector = embedding_list[0] 
        
        # Convert to PyTorch Tensor
        x = torch.tensor(embedding_vector, dtype=torch.float32)
            
        # Get label and convert to integer index
        label_name = self.id_to_label[pid]
        y = torch.tensor(self.class_to_idx[label_name], dtype=torch.long)
        
        return x, y

# Function to easily get loaders for the rest of the team
def get_dataloaders(batch_size=32, num_workers=2):
    # Hardcoded paths relative to the project root
    raw_dir = 'data/raw'
    processed_dir = 'data/processed'
    
    pkl_path = os.path.join(raw_dir, 'tcga_titan_embeddings.pkl')
    labels_path = os.path.join(raw_dir, 'tcga_patient_to_cancer_type.csv')
    
    train_split = os.path.join(processed_dir, 'train_split.csv')
    val_split = os.path.join(processed_dir, 'val_split.csv')
    
    # 1. Create Train Dataset
    print("Initializing Train Dataset...")
    train_dataset = TCGADataset(pkl_path, labels_path, split_map_path=train_split)
    
    # 2. Create Validation Dataset (reuse class_to_idx from train to ensure mapping matches!)
    print("Initializing Val Dataset...")
    val_dataset = TCGADataset(pkl_path, labels_path, split_map_path=val_split, 
                              class_to_idx=train_dataset.class_to_idx)
    
    # 3. Create Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Small test block to verify everything works
    print("--- Testing Dataset Implementation ---")
    try:
        t_loader, v_loader = get_dataloaders()
        features, labels = next(iter(t_loader))
        print(f"\nSuccess! Got batch of shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
    except Exception as e:
        print(f"\nError: {e}")
        print("Did you run scripts/create_split.py first?")
