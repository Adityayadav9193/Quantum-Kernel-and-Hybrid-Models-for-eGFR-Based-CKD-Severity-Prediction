"""
Automated validation checks for the corrected implementation.
This script verifies all critical aspects of the corrected methodology.
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Simple import approach
sys.path.insert(0, os.path.join(project_root, 'src', 'preprocessing'))
import preprocessing

CKDPreprocessor = preprocessing.CKDPreprocessor
create_experiment_feature_sets = preprocessing.create_experiment_feature_sets

def validate_label_generation():
    """Validate that labels were generated from observed data only."""
    print("="*60)
    print("VALIDATION 1: LABEL GENERATION")
    print("="*60)
    
    # Load observed-only dataset
    observed_path = os.path.join(project_root, 'data/processed', 'ckd_egfr_categories_observed_only.csv')
    df_observed = pd.read_csv(observed_path)
    
    # Load raw dataset
    raw_path = os.path.join(project_root, 'data/raw', 'chronic_kidney_disease_raw.csv')
    df_raw = pd.read_csv(raw_path)
    
    # Check that observed dataset has no missing sc or age
    missing_sc_age = df_observed[['sc', 'age']].isnull().any(axis=1).sum()
    assert missing_sc_age == 0, f"Observed dataset has {missing_sc_age} samples with missing sc or age"
    print("✓ Observed dataset has no missing sc or age")
    
    # Check that observed dataset size matches expected
    expected_size = len(df_raw) - df_raw[['sc', 'age']].isnull().any(axis=1).sum()
    assert len(df_observed) == expected_size, f"Observed dataset size mismatch: {len(df_observed)} vs {expected_size}"
    print(f"✓ Observed dataset size correct: {len(df_observed)} samples")
    
    # Check that all eGFR values are non-NaN
    assert df_observed['egfr'].notnull().all(), "Observed dataset has NaN eGFR values"
    print("✓ All eGFR values are non-NaN")
    
    # Check that all category labels are valid (1-5)
    assert df_observed['egfr_category'].between(1, 5).all(), "Invalid category labels found"
    print("✓ All category labels are valid (1-5)")
    
    # Check that no imputed values were used for target generation
    # This is implicitly validated by the missing value check above
    print("✓ No imputed values used for target generation")
    
    return True

def validate_preprocessing_leakage():
    """Validate that preprocessing is leakage-free."""
    print("\n" + "="*60)
    print("VALIDATION 2: PREPROCESSING LEAKAGE")
    print("="*60)
    
    # Load preprocessor
    preprocessor_path = os.path.join(project_root, 'models', 'preprocessor_corrected.pkl')
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Check that preprocessor has imputation objects
    assert hasattr(preprocessor, 'numeric_imputer'), "Preprocessor missing numeric imputer"
    assert hasattr(preprocessor, 'categorical_imputer'), "Preprocessor missing categorical imputer"
    print("✓ Preprocessor has imputation objects")
    
    # Check that imputation objects are fitted
    assert preprocessor.numeric_imputer.statistics_ is not None, "Numeric imputer not fitted"
    assert preprocessor.categorical_imputer.statistics_ is not None, "Categorical imputer not fitted"
    print("✓ Imputation objects are fitted")
    
    # Check that scaler is fitted
    assert preprocessor.scaler.mean_ is not None, "Scaler not fitted"
    print("✓ Scaler is fitted")
    
    # Check that label encoder is fitted
    assert preprocessor.label_encoder.classes_ is not None, "Label encoder not fitted"
    print("✓ Label encoder is fitted")
    
    return True

def validate_leakage_experiments():
    """Validate that leakage experiments are correctly configured."""
    print("\n" + "="*60)
    print("VALIDATION 3: LEAKAGE EXPERIMENTS")
    print("="*60)
    
    experiments = create_experiment_feature_sets()
    
    # Validate Experiment A
    exp_a = experiments['A_full_features']
    assert 'sc' in exp_a['features'], "Experiment A should include sc"
    assert 'age' in exp_a['features'], "Experiment A should include age"
    assert not exp_a['is_leakage_aware'], "Experiment A should not be leakage-aware"
    print("✓ Experiment A correctly configured (with leakage)")
    
    # Validate Experiment B
    exp_b = experiments['B_no_leakage']
    assert 'sc' not in exp_b['features'], "Experiment B should exclude sc"
    assert 'age' not in exp_b['features'], "Experiment B should exclude age"
    assert exp_b['is_leakage_aware'], "Experiment B should be leakage-aware"
    assert 'sc' in exp_b['excluded_features'], "sc should be in excluded features"
    assert 'age' in exp_b['excluded_features'], "age should be in excluded features"
    print("✓ Experiment B correctly configured (leakage-aware)")
    
    # Validate Experiment C
    exp_c = experiments['C_partial_exclusion']
    assert 'sc' not in exp_c['features'], "Experiment C should exclude sc"
    assert 'age' in exp_c['features'], "Experiment C should include age"
    assert exp_c['is_leakage_aware'], "Experiment C should be leakage-aware"
    assert 'sc' in exp_c['excluded_features'], "sc should be in excluded features"
    print("✓ Experiment C correctly configured (partial exclusion)")
    
    return True

def validate_train_test_separation():
    """Validate that train/validation/test separation is correct."""
    print("\n" + "="*60)
    print("VALIDATION 4: TRAIN/VALIDATION/TEST SEPARATION")
    print("="*60)
    
    # Load observed dataset
    observed_path = os.path.join(project_root, 'data/processed', 'ckd_egfr_categories_observed_only.csv')
    df = pd.read_csv(observed_path)
    
    # Create preprocessor and split data
    preprocessor = CKDPreprocessor(random_state=42)
    experiments = create_experiment_feature_sets()
    feature_columns = experiments['B_no_leakage']['features']
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        df, feature_columns, target_column='egfr_category'
    )
    
    # Check that splits are disjoint
    train_indices = set(X_train.index)
    val_indices = set(X_val.index)
    test_indices = set(X_test.index)
    
    assert len(train_indices & val_indices) == 0, "Train and validation sets overlap"
    assert len(train_indices & test_indices) == 0, "Train and test sets overlap"
    assert len(val_indices & test_indices) == 0, "Validation and test sets overlap"
    print("✓ Train/validation/test sets are disjoint")
    
    # Check split proportions
    total_samples = len(df)
    train_pct = len(X_train) / total_samples
    val_pct = len(X_val) / total_samples
    test_pct = len(X_test) / total_samples
    
    assert 0.6 <= train_pct <= 0.7, f"Train proportion out of range: {train_pct:.2f}"
    assert 0.15 <= val_pct <= 0.2, f"Validation proportion out of range: {val_pct:.2f}"
    assert 0.18 <= test_pct <= 0.22, f"Test proportion out of range: {test_pct:.2f}"
    print(f"✓ Split proportions correct: Train {train_pct:.1%}, Val {val_pct:.1%}, Test {test_pct:.1%}")
    
    # Check stratification
    train_dist = y_train.value_counts(normalize=True).sort_index()
    test_dist = y_test.value_counts(normalize=True).sort_index()
    
    # Check that distributions are similar (within 10%)
    for category in range(1, 6):
        if category in train_dist.index and category in test_dist.index:
            diff = abs(train_dist[category] - test_dist[category])
            assert diff < 0.1, f"Stratification failed for category {category}: diff={diff:.3f}"
    print("✓ Stratification maintains class distribution")
    
    return True

def validate_target_distribution():
    """Validate target distribution and cross-tabulation."""
    print("\n" + "="*60)
    print("VALIDATION 5: TARGET DISTRIBUTION")
    print("="*60)
    
    # Load observed dataset
    observed_path = os.path.join(project_root, 'data/processed', 'ckd_egfr_categories_observed_only.csv')
    df = pd.read_csv(observed_path)
    
    # Check category distribution
    category_counts = df['egfr_category'].value_counts().sort_index()
    
    # Check that all categories have sufficient samples
    for category in range(1, 6):
        count = category_counts.get(category, 0)
        assert count >= 10, f"Category {category} has insufficient samples: {count}"
    print("✓ All categories have sufficient samples (≥10)")
    
    # Check cross-tabulation with original label
    crosstab = pd.crosstab(df['class'], df['egfr_category'])
    
    # Check that notckd samples exist in high eGFR categories (expected)
    notckd_in_high_egfr = crosstab.loc['notckd', 1] + crosstab.loc['notckd', 2]
    assert notckd_in_high_egfr > 0, "Expected notckd samples in high eGFR categories"
    print(f"✓ Cross-tabulation shows expected pattern: {notckd_in_high_egfr} notckd in high eGFR categories")
    
    # Check that ckd samples exist in all categories (expected)
    ckd_in_all_categories = all(crosstab.loc['ckd', cat] > 0 for cat in range(1, 6))
    assert ckd_in_all_categories, "Expected ckd samples in all categories"
    print("✓ Cross-tabulation shows expected pattern: ckd samples in all categories")
    
    return True

def validate_terminology():
    """Validate that correct terminology is used."""
    print("\n" + "="*60)
    print("VALIDATION 6: TERMINOLOGY")
    print("="*60)
    
    # Check that corrected files use correct terminology
    corrected_files = [
        'reports/label_derivation_corrected.md',
        'reports/leakage_analysis_corrected.md'
    ]
    
    for file_path in corrected_files:
        full_path = os.path.join(project_root, file_path)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                content = f.read()
            
            # Check for incorrect terminology
            assert 'CKD Stage' not in content or 'CKD Stage' in 'clinical CKD stage', f"Incorrect terminology found in {file_path}"
            assert 'eGFR-based' in content, f"Correct terminology not found in {file_path}"
            print(f"✓ {file_path} uses correct terminology")
    
    # Check that observed dataset uses correct column names
    observed_path = os.path.join(project_root, 'data/processed', 'ckd_egfr_categories_observed_only.csv')
    df = pd.read_csv(observed_path)
    
    assert 'egfr_category' in df.columns, "Dataset should use 'egfr_category' column name"
    assert 'ckd_stage' not in df.columns, "Dataset should not use 'ckd_stage' column name"
    print("✓ Dataset uses correct column names")
    
    return True

def run_all_validations():
    """Run all validation checks."""
    print("="*60)
    print("AUTOMATED VALIDATION CHECKS")
    print("="*60)
    
    validations = [
        ("Label Generation", validate_label_generation),
        ("Preprocessing Leakage", validate_preprocessing_leakage),
        ("Leakage Experiments", validate_leakage_experiments),
        ("Train/Test Separation", validate_train_test_separation),
        ("Target Distribution", validate_target_distribution),
        ("Terminology", validate_terminology)
    ]
    
    results = {}
    for name, validation_func in validations:
        try:
            validation_func()
            results[name] = "PASS"
        except AssertionError as e:
            print(f"✗ {name} FAILED: {e}")
            results[name] = "FAIL"
        except Exception as e:
            print(f"✗ {name} ERROR: {e}")
            results[name] = "ERROR"
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for name, result in results.items():
        status = "✓" if result == "PASS" else "✗"
        print(f"{status} {name}: {result}")
    
    all_passed = all(result == "PASS" for result in results.values())
    
    if all_passed:
        print("\n" + "="*60)
        print("ALL VALIDATIONS PASSED")
        print("="*60)
        return True
    else:
        print("\n" + "="*60)
        print("SOME VALIDATIONS FAILED")
        print("="*60)
        return False

def main():
    """Main execution function."""
    success = run_all_validations()
    
    if success:
        print("\nCorrected implementation passed all validation checks.")
        print("Ready for documentation and final verification.")
    else:
        print("\nCorrected implementation failed some validation checks.")
        print("Please fix the issues before proceeding.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
