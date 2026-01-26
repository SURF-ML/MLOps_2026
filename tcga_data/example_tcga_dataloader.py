import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class TCGADataset(Dataset):
    """
    Dataset class for TCGA embeddings and cancer type labels.
    Expected data format: Pickled dictionary with patient IDs as keys.
    """
    def __init__(self, pickle_path: str, csv_path: str, class_to_idx=None):
    	# TODO: add the text embeddings! Make sure they are aligned with the pid!
        # 1. Load the noised embeddings
        with open(pickle_path, 'rb') as f:
            self.embedding_data = pickle.load(f)
        
        # 2. Load the labels
        self.labels_df = pd.read_csv(csv_path)
        
        # 3. Create or use existing class mapping (ensures train/test alignment)
        if class_to_idx is None:
            self.cancer_types = sorted(self.labels_df['cancer_type'].unique())
            self.class_to_idx = {name: i for i, name in enumerate(self.cancer_types)}
        else:
            self.class_to_idx = class_to_idx

        # 4. Identify patients that exist in BOTH the embeddings and the label file
        self.valid_patient_ids = sorted(list(
            set(self.embedding_data.keys()).intersection(set(self.labels_df['patient_id']))
        ))
        
        # 5. Map Patient IDs to their cancer type labels for fast lookup
        self.id_to_label = dict(zip(self.labels_df['patient_id'], self.labels_df['cancer_type']))

    def __len__(self):
        """Returns the total number of valid patients in the dataset."""
        return len(self.valid_patient_ids)

    def __getitem__(self, idx):	
        """
        Retrieves a single sample (embedding + label).
        Since data is pre-noised and mean-pooled, we take the first item 
        from the embeddings list.
        """
        pid = self.valid_patient_ids[idx]
        
        # Retrieve an embedding (768-dim vector)
        # TODO: think what you're going to do with the patients that have more than 1 embedding
	# That is, patients who have had more than 1 imaging done
        embedding_list = self.embedding_data[pid]['embeddings']
        embedding_vector = embedding_list[0] 
        
        # Convert to PyTorch Tensor
        x = torch.tensor(embedding_vector, dtype=torch.float32)
            
        # Get label and convert to integer index
        label_name = self.id_to_label[pid]
        y = torch.tensor(self.class_to_idx[label_name], dtype=torch.long)
        
        return x, y

def main():
    # --- Configuration ---
    TRAIN_PKL = 'tcga_titan_embeddings.pkl'
    LABEL_CSV = 'tcga_patient_to_cancer_type.csv'
    BATCH_SIZE = 32

    # --- Loading ---
    print("Loading TCGA Dataset...")
    train_dataset = TCGADataset(TRAIN_PKL, LABEL_CSV)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )

    # --- Verification ---
    print(f"Dataset loaded successfully!")
    print(f"Total Patients: {len(train_dataset)}")
    print(f"Number of Classes: {len(train_dataset.class_to_idx)}")
    
    # Peek at the first batch
    features, labels = next(iter(train_loader))
    print(f"\nBatch Information:")
    print(f"Feature shape: {features.shape} (Expected: [{BATCH_SIZE}, 768])")
    print(f"Labels shape:  {labels.shape}  (Expected: [{BATCH_SIZE}])")
    
    # Display the mapping to the students
    print("\nClass Mapping (first 5):")
    for i, (name, idx) in enumerate(train_dataset.class_to_idx.items()):
        print(f"  {name}: {idx}")
        if i == 4: break

if __name__ == "__main__":
    main()
