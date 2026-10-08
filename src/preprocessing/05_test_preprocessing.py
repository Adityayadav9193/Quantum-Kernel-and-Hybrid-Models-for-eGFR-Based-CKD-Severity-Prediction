"""
Corrected test script for preprocessing pipeline.
Tests the corrected pipeline with observed-only data and proper leakage control.
"""

import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing.preprocessing import CKDPreprocessor, create_experiment_feature_sets

def main():
    """Test the corrected preprocessing pipeline."""
    print("="*60)
    print("TESTING CORRECTED PREPROCESSING PIPELINE")
    print("="*60)
    
    # Load observed-only data (corrected dataset)
    data_path = os.path.join(project_root, 'data/processed', 'ckd_egfr_categories_observed_only.csv')
    df = pd.read_csv(data_path)
    print(f"Loaded observed-only data: {df.shape}")
    
    # Create preprocessor
    preprocessor = CKDPreprocessor(random_state=42)
    
    # Define feature sets
    experiments = create_experiment_feature_sets()
    
    print("\nAvailable Leakage Experiments:")
    for exp_id, exp in experiments.items():
        print(f"  {exp_id}: {exp['name']}")
        print(f"    Features: {len(exp['features'])}")
        print(f"    Excluded: {exp['excluded_features']}")
        print(f"    Leakage-aware: {exp['is_leakage_aware']}")
        print(f"    Clinical validity: {exp['clinical_validity']}")
        print()
    
    # Test with leakage-aware feature set (Experiment B)
    feature_columns = experiments['B_no_leakage']['features']
    
    print(f"Testing with: {experiments['B_no_leakage']['name']}")
    print(f"Features: {len(feature_columns)}")
    print(f"Excluded leakage features: {experiments['B_no_leakage']['excluded_features']}")
    
    # Split data (with corrected target column name)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        df, feature_columns, target_column='egfr_category'
    )
    
    # Preprocess training data (includes imputation fitted on train only)
    X_train_processed, y_train_encoded = preprocessor.preprocess_train(X_train, y_train)
    
    # Preprocess validation data (uses training parameters)
    X_val_processed, y_val_encoded = preprocessor.preprocess_test(X_val, y_val)
    
    # Preprocess test data (uses training parameters)
    X_test_processed, y_test_encoded = preprocessor.preprocess_test(X_test, y_test)
    
    print("\nPreprocessing complete!")
    print(f"Train shape: {X_train_processed.shape}")
    print(f"Validation shape: {X_val_processed.shape}")
    print(f"Test shape: {X_test_processed.shape}")
    
    # Validation checks
    print("\n" + "="*60)
    print("VALIDATION CHECKS")
    print("="*60)
    
    # Check no NaN values
    assert not X_train_processed.isnull().any().any(), "Train data contains NaN values"
    assert not X_val_processed.isnull().any().any(), "Validation data contains NaN values"
    assert not X_test_processed.isnull().any().any(), "Test data contains NaN values"
    print("✓ No NaN values in processed data")
    
    # Check target distribution
    print("\nTarget distribution:")
    print(f"Train: {pd.Series(y_train_encoded).value_counts().sort_index().to_dict()}")
    print(f"Validation: {pd.Series(y_val_encoded).value_counts().sort_index().to_dict()}")
    print(f"Test: {pd.Series(y_test_encoded).value_counts().sort_index().to_dict()}")
    
    # Check that leakage features are excluded
    assert 'sc' not in feature_columns, "sc (serum creatinine) should be excluded in leakage-aware experiment"
    assert 'age' not in feature_columns, "age should be excluded in leakage-aware experiment"
    print("✓ Leakage features correctly excluded")
    
    # Save preprocessor
    os.makedirs(project_root + '/models', exist_ok=True)
    preprocessor_path = os.path.join(project_root, 'models', 'preprocessor_corrected.pkl')
    preprocessor.save_preprocessor(preprocessor_path)
    
    print("\n" + "="*60)
    print("CORRECTED PREPROCESSING TEST COMPLETE")
    print("="*60)
    print("All validation checks passed.")
    print("Preprocessing pipeline is leakage-free.")

if __name__ == "__main__":
    main()
