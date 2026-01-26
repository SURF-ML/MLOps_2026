import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split

RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
LABEL_FILE = os.path.join(RAW_DATA_DIR, 'tcga_patient_to_cancer_type.csv')
EMBEDDING_FILE = os.path.join(RAW_DATA_DIR, 'tcga_titan_embeddings.pkl')

def create_splits():
    print("Loading data...")
    df_labels = pd.read_csv(LABEL_FILE)
    
    with open(EMBEDDING_FILE, 'rb') as f:
        embeddings_dict = pickle.load(f)
    
    available_patients_with_images = set(embeddings_dict.keys())
    
    df_clean = df_labels[df_labels['patient_id'].isin(available_patients_with_images)].copy()
    
    print(f"Total patients in CSV: {len(df_labels)}")
    print(f"Total patients with images: {len(available_patients_with_images)}")
    print(f"Valid patients (intersection): {len(df_clean)}")
    
    X = df_clean['patient_id'].values
    y = df_clean['cancer_type'].values
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val
    )
    
    print("\n--- Split Resultaten ---")
    print(f"Train set: {len(X_train)} patiënten")
    print(f"Val set:   {len(X_val)} patiënten")
    print(f"Test set:  {len(X_test)} patiënten")
    
    assert len(set(X_train) & set(X_test)) == 0, "LEKKAGE: Train en Test overlappen!"
    assert len(set(X_train) & set(X_val)) == 0, "LEKKAGE: Train en Val overlappen!"
    assert len(set(X_val) & set(X_test)) == 0, "LEKKAGE: Val en Test overlappen!"
    print("Geen data leakage gevonden. Splits zijn geldig.")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    pd.DataFrame(X_train, columns=['patient_id']).to_csv(os.path.join(PROCESSED_DATA_DIR, 'train_split.csv'), index=False)
    pd.DataFrame(X_val, columns=['patient_id']).to_csv(os.path.join(PROCESSED_DATA_DIR, 'val_split.csv'), index=False)
    pd.DataFrame(X_test, columns=['patient_id']).to_csv(os.path.join(PROCESSED_DATA_DIR, 'test_split.csv'), index=False)
    
    print(f"\nSplits opgeslagen in {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    create_splits()
