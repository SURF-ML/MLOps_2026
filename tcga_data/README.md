# TCGA Data Loading Project

This directory contains scripts and data files for processing TCGA patient embeddings and clinical information.

## File Descriptions

* **example_tcga_dataloader.py** A Python script containing a custom PyTorch `Dataset` and `DataLoader` implementation designed to align patient embeddings with cancer type labels.

* **tcga_titan_embeddings.pkl** A pickled dictionary where keys are patient IDs and values contain lists of 768-dimensional feature vectors (embeddings).

* **tcga_patient_to_cancer_type.csv** A CSV file mapping patient IDs to their respective cancer type classifications, used as the ground-truth labels for training.

* **tcga_reports.jsonl** A JSON Lines file containing textual clinical or pathology reports for each patient, formatted for NLP tasks or multi-modal integration.
