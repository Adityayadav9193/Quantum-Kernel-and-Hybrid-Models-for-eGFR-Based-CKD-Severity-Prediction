"""
Data cleaning script for UCI Chronic Kidney Disease dataset.
This script addresses the issues identified in the data audit.
"""

import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def clean_whitespace(df):
    """Clean whitespace issues in categorical variables."""
    print("Cleaning whitespace issues...")
    
    # Fix target variable 'class'
    df['class'] = df['class'].str.strip()
    
    # Fix diabetes 'dm' column
    df['dm'] = df['dm'].str.strip()
    
    print("  - Fixed whitespace in 'class' and 'dm' columns")
    
    return df

def handle_missing_values(df):
    """Handle missing values with appropriate imputation strategies."""
    print("Handling missing values...")
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns
    
    # Impute numeric columns with median
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  - Imputed {col} with median: {median_val:.2f}")
    
    # Impute categorical columns with mode
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"  - Imputed {col} with mode: {mode_val}")
    
    return df

def investigate_outliers(df):
    """Investigate potential outliers."""
    print("\nInvestigating potential outliers...")
    
    # Check suspicious sodium value
    sod_min = df['sod'].min()
    sod_q1 = df['sod'].quantile(0.25)
    print(f"  - Sodium minimum: {sod_min} (Q1: {sod_q1})")
    
    if sod_min < 100:
        print(f"  - WARNING: Sodium value {sod_min} is suspiciously low")
        print(f"  - Number of samples with sod < 100: {(df['sod'] < 100).sum()}")
    
    # Check pediatric cases
    pediatric = df[df['age'] < 18]
    print(f"  - Pediatric cases (age < 18): {len(pediatric)}")
    
    return df

def verify_cleaning(df):
    """Verify that cleaning was successful."""
    print("\nVerifying data cleaning...")
    
    # Check for remaining missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("  - WARNING: Missing values still present:")
        print(missing[missing > 0])
    else:
        print("  - No missing values remaining")
    
    # Check target variable
    print(f"  - Target variable unique values: {df['class'].unique()}")
    print(f"  - Target variable distribution:")
    print(df['class'].value_counts())
    
    # Check dm variable
    print(f"  - Diabetes variable unique values: {df['dm'].unique()}")
    
    return df

def save_cleaned_dataset(df):
    """Save the cleaned dataset."""
    os.makedirs(project_root + '/data/processed', exist_ok=True)
    
    cleaned_path = os.path.join(project_root, 'data/processed', 'chronic_kidney_disease_cleaned.csv')
    df.to_csv(cleaned_path, index=False)
    print(f"\nCleaned dataset saved to: {cleaned_path}")
    
    return cleaned_path

def main():
    """Main execution function."""
    print("="*60)
    print("DATA CLEANING")
    print("="*60)
    
    # Load raw dataset
    raw_path = os.path.join(project_root, 'data/raw', 'chronic_kidney_disease_raw.csv')
    df = pd.read_csv(raw_path)
    print(f"Loaded raw dataset: {df.shape}")
    
    # Clean whitespace
    df = clean_whitespace(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Investigate outliers
    df = investigate_outliers(df)
    
    # Verify cleaning
    df = verify_cleaning(df)
    
    # Save cleaned dataset
    save_cleaned_dataset(df)
    
    print("\n" + "="*60)
    print("DATA CLEANING COMPLETE")
    print("="*60)
    print("Cleaned dataset saved to data/processed/chronic_kidney_disease_cleaned.csv")

if __name__ == "__main__":
    main()
