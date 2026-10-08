"""
Dataset download and audit script for UCI Chronic Kidney Disease dataset.
This script downloads the dataset and performs a comprehensive audit.
"""

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def download_dataset():
    """Download the UCI Chronic Kidney Disease dataset."""
    print("Downloading UCI Chronic Kidney Disease dataset...")
    
    try:
        # Fetch dataset
        ckd = fetch_ucirepo(id=336)
        
        # Get data
        X = ckd.data.features
        y = ckd.data.targets
        
        # Combine into single dataframe
        df = pd.concat([X, y], axis=1)
        
        print(f"Dataset downloaded successfully!")
        print(f"Shape: {df.shape}")
        
        return df, ckd
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None, None

def audit_dataset(df):
    """Perform comprehensive dataset audit."""
    print("\n" + "="*60)
    print("DATASET AUDIT REPORT")
    print("="*60)
    
    # 1. Basic information
    print("\n1. BASIC INFORMATION")
    print("-" * 60)
    print(f"Shape: {df.shape}")
    print(f"Number of samples: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")
    
    # 2. Column names
    print("\n2. COLUMN NAMES")
    print("-" * 60)
    print("Columns:", df.columns.tolist())
    
    # 3. Data types
    print("\n3. DATA TYPES")
    print("-" * 60)
    print(df.dtypes)
    
    # 4. Missing values
    print("\n4. MISSING VALUES")
    print("-" * 60)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df)
    
    # 5. Unique values per column
    print("\n5. UNIQUE VALUES PER COLUMN")
    print("-" * 60)
    for col in df.columns:
        unique_count = df[col].nunique()
        unique_values = df[col].unique()
        print(f"\n{col}:")
        print(f"  Unique count: {unique_count}")
        print(f"  Unique values: {unique_values[:20]}")  # Show first 20
        if unique_count > 20:
            print(f"  ... (and {unique_count - 20} more)")
    
    # 6. Class distribution
    print("\n6. CLASS DISTRIBUTION")
    print("-" * 60)
    target_col = df.columns[-1]  # Assuming last column is target
    class_counts = df[target_col].value_counts()
    class_pct = (class_counts / len(df)) * 100
    print(f"Target column: {target_col}")
    print("Class distribution:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} ({class_pct[cls]:.2f}%)")
    
    # 7. Malformed values
    print("\n7. MALFORMED VALUES CHECK")
    print("-" * 60)
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check for whitespace issues
            whitespace_mask = df[col].str.contains(r'^\s+|\s+$', na=False)
            if whitespace_mask.any():
                print(f"  {col}: Has whitespace issues ({whitespace_mask.sum()} rows)")
            
            # Check for unusual characters
            try:
                # Try to convert numeric columns that might be stored as strings
                pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    
    # 8. Duplicate rows
    print("\n8. DUPLICATE ROWS")
    print("-" * 60)
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates} ({(duplicates/len(df))*100:.2f}%)")
    
    # 9. Statistical summary for numeric columns
    print("\n9. STATISTICAL SUMMARY (NUMERIC COLUMNS)")
    print("-" * 60)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    else:
        print("No numeric columns found")
    
    # 10. Target variable analysis
    print("\n10. TARGET VARIABLE ANALYSIS")
    print("-" * 60)
    print(f"Target column: {target_col}")
    print(f"Target type: {df[target_col].dtype}")
    print(f"Target unique values: {df[target_col].unique()}")
    
    return df

def save_raw_dataset(df, dataset_info):
    """Save the raw dataset and dataset info."""
    # Create directories if they don't exist
    os.makedirs(project_root + '/data/raw', exist_ok=True)
    os.makedirs(project_root + '/reports', exist_ok=True)
    
    # Save raw dataset
    raw_path = os.path.join(project_root, 'data/raw', 'chronic_kidney_disease_raw.csv')
    df.to_csv(raw_path, index=False)
    print(f"\nRaw dataset saved to: {raw_path}")
    
    # Save dataset metadata
    if dataset_info:
        metadata_path = os.path.join(project_root, 'data/raw', 'dataset_metadata.txt')
        with open(metadata_path, 'w') as f:
            f.write("UCI Chronic Kidney Disease Dataset Metadata\n")
            f.write("="*60 + "\n\n")
            f.write(f"UCI ID: {dataset_info.metadata['uci_id']}\n")
            f.write(f"Name: {dataset_info.metadata['name']}\n")
            f.write(f"Abstract: {dataset_info.metadata['abstract']}\n")
            f.write(f"\nFeatures: {len(dataset_info.data.features.columns)}\n")
            f.write(f"Targets: {len(dataset_info.data.targets.columns)}\n")
        print(f"Dataset metadata saved to: {metadata_path}")

def main():
    """Main execution function."""
    print("="*60)
    print("PHASE 1: DATASET DOWNLOAD AND AUDIT")
    print("="*60)
    
    # Download dataset
    df, dataset_info = download_dataset()
    
    if df is None:
        print("Failed to download dataset. Exiting.")
        return
    
    # Perform audit
    df = audit_dataset(df)
    
    # Save dataset
    save_raw_dataset(df, dataset_info)
    
    print("\n" + "="*60)
    print("PHASE 1 COMPLETE")
    print("="*60)
    print("Dataset audit complete. Check the output above for findings.")
    print("Raw dataset saved to data/raw/chronic_kidney_disease_raw.csv")

if __name__ == "__main__":
    main()
