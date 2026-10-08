"""
Phase 5: Classical Baselines
Uses corrected methodology from Phases 1-4.
Primary experiment: Experiment B (leakage-aware exploratory configuration).
Selected features: bgr, bu, pot, hemo, pcv, rbcc, sg, sod
"""

import pandas as pd
import numpy as np
import os
import json
import time
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Use current working directory as project root
project_root = os.getcwd()

def load_corrected_data():
    """Load the corrected observed-only dataset."""
    data_path = os.path.join(project_root, 'data/processed', 'ckd_egfr_categories_observed_only.csv')
    df = pd.read_csv(data_path)
    print(f"Loaded corrected dataset: {df.shape}")
    return df

def create_established_split(df, features, target='egfr_category'):
    """
    Create the ESTABLISHED Phase 3 split (239/60/75).
    This is the CORRECT methodology.
    """
    print("\n" + "="*60)
    print("CREATING ESTABLISHED PHASE 3 SPLIT")
    print("="*60)
    
    X = df[features].copy()
    y = df[target].copy()
    
    # First split: train+val vs test (20% test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: train vs val (20% of remaining for validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )
    
    print(f"Training data: {len(X_train)} samples (from Phase 3)")
    print(f"Validation data: {len(X_val)} samples (from Phase 3)")
    print(f"Test data: {len(X_test)} samples (from Phase 3)")
    
    # Verify target distributions
    print("\nTarget Distribution:")
    print(f"Training: {y_train.value_counts().sort_index().to_dict()}")
    print(f"Validation: {y_val.value_counts().sort_index().to_dict()}")
    print(f"Test: {y_test.value_counts().sort_index().to_dict()}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_preprocessing_pipeline(X_train, features):
    """
    Create preprocessing pipeline fitted ONLY on training data.
    For the selected 8 features.
    """
    print("\n" + "="*60)
    print("CREATING PREPROCESSING PIPELINE")
    print("="*60)
    
    # Identify numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()
    
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    # Fit imputers on training data only
    numeric_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    
    # Fit scalers on training data only
    scaler = StandardScaler()
    
    # Fit encoders on training data only
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        label_encoders[col] = le
    
    # Fit imputation on training data
    X_train[numeric_cols] = numeric_imputer.fit_transform(X_train[numeric_cols])
    if categorical_cols:
        X_train[categorical_cols] = categorical_imputer.fit_transform(X_train[categorical_cols])
    
    # Fit scaling on training data
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    
    preprocessing_pipeline = {
        'numeric_imputer': numeric_imputer,
        'categorical_imputer': categorical_imputer,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }
    
    print("✓ Preprocessing pipeline fitted on training data only")
    print("✓ Imputation fitted on training data only")
    print("✓ Scaling fitted on training data only")
    print("✓ Encoding fitted on training data only")
    
    return preprocessing_pipeline

def apply_preprocessing(X, preprocessing_pipeline):
    """Apply fitted preprocessing pipeline to data."""
    X_processed = X.copy()
    
    numeric_cols = preprocessing_pipeline['numeric_cols']
    categorical_cols = preprocessing_pipeline['categorical_cols']
    
    # Apply encoding
    for col in categorical_cols:
        le = preprocessing_pipeline['label_encoders'][col]
        X_processed[col] = le.transform(X_processed[col].astype(str))
    
    # Apply imputation
    X_processed[numeric_cols] = preprocessing_pipeline['numeric_imputer'].transform(X_processed[numeric_cols])
    if categorical_cols:
        X_processed[categorical_cols] = preprocessing_pipeline['categorical_imputer'].transform(X_processed[categorical_cols])
    
    # Apply scaling
    X_processed[numeric_cols] = preprocessing_pipeline['scaler'].transform(X_processed[numeric_cols])
    
    return X_processed

def train_model_with_cv(X_train, y_train, model_name, cv_folds=5):
    """
    Train model with cross-validation on training data.
    Returns model and CV results.
    """
    print(f"\n" + "="*60)
    print(f"TRAINING {model_name} WITH CROSS-VALIDATION")
    print("="*60)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42
        )
    }
    
    model = models[model_name]
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_scores = []
    cv_f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # Clone model for this fold
        from sklearn.base import clone
        model_fold = clone(model)
        
        model_fold.fit(X_fold_train, y_fold_train)
        y_pred = model_fold.predict(X_fold_val)
        
        fold_acc = accuracy_score(y_fold_val, y_pred)
        fold_f1 = f1_score(y_fold_val, y_pred, average='macro')
        
        cv_scores.append(fold_acc)
        cv_f1_scores.append(fold_f1)
        
        print(f"Fold {fold+1}: Accuracy={fold_acc:.4f}, Macro-F1={fold_f1:.4f}")
    
    # Train final model on full training data
    model.fit(X_train, y_train)
    
    cv_results = {
        'mean_accuracy': np.mean(cv_scores),
        'std_accuracy': np.std(cv_scores),
        'mean_macro_f1': np.mean(cv_f1_scores),
        'std_macro_f1': np.std(cv_f1_scores)
    }
    
    print(f"\nCV Results:")
    print(f"  Mean Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
    print(f"  Mean Macro-F1: {cv_results['mean_macro_f1']:.4f} ± {cv_results['std_macro_f1']:.4f}")
    
    return model, cv_results

def evaluate_model(model, X, y, model_name, split_name):
    """Evaluate model on given data."""
    y_pred = model.predict(X)
    
    accuracy = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average='macro')
    weighted_f1 = f1_score(y, y_pred, average='weighted')
    
    # Per-class metrics
    precision = precision_score(y, y_pred, average=None, zero_division=0)
    recall = recall_score(y, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y, y_pred, average=None, zero_division=0)
    
    # ROC-AUC (one-vs-rest)
    try:
        y_proba = model.predict_proba(X)
        roc_auc = roc_auc_score(y, y_proba, multi_class='ovr', average='macro')
    except:
        roc_auc = None
    
    print(f"\n{model_name} - {split_name} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro-F1: {macro_f1:.4f}")
    print(f"  Weighted-F1: {weighted_f1:.4f}")
    if roc_auc is not None:
        print(f"  ROC-AUC: {roc_auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    results = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'roc_auc': float(roc_auc) if roc_auc is not None else None,
        'confusion_matrix': cm.tolist(),
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1_per_class.tolist()
    }
    
    return results

def hyperparameter_tuning(X_train, y_train, X_val, y_val, model_name):
    """
    Lightweight hyperparameter tuning on validation set.
    """
    print(f"\n" + "="*60)
    print(f"HYPERPARAMETER TUNING FOR {model_name}")
    print("="*60)
    
    # Define parameter grids (lightweight)
    param_grids = {
        'Logistic Regression': {
            'C': [0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'saga']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None]
        },
        'MLP': {
            'hidden_layer_sizes': [(32,), (64,), (64, 32)],
            'alpha': [0.0001, 0.001]
        }
    }
    
    base_models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'MLP': MLPClassifier(
            max_iter=500,
            random_state=42
        )
    }
    
    model = base_models[model_name]
    param_grid = param_grids[model_name]
    
    best_score = 0
    best_params = None
    best_model = None
    
    # Generate all parameter combinations
    from itertools import product
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    for combination in product(*values):
        params = dict(zip(keys, combination))
        try:
            model.set_params(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average='macro')
            
            if score > best_score:
                best_score = score
                best_params = params
                best_model = model
                
            print(f"Params: {params}, Validation Macro-F1: {score:.4f}")
        except Exception as e:
            print(f"Params: {params} failed: {e}")
            continue
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best validation Macro-F1: {best_score:.4f}")
    
    return best_model, best_params, best_score

def main():
    """Main execution function."""
    print("="*60)
    print("PHASE 5: CLASSICAL BASELINES")
    print("="*60)
    
    # Load corrected data
    df = load_corrected_data()
    
    # Load selected features
    json_path = os.path.join(project_root, 'results/figures/feature_selection/selected_features_corrected.json')
    with open(json_path, 'r') as f:
        feature_results = json.load(f)
    
    selected_features = feature_results['selected_features']
    
    print(f"\nSelected Features (8): {selected_features}")
    print(f"Experiment: B_leakage_aware_exploratory")
    
    # Verify leakage features are excluded
    assert 'sc' not in selected_features, "sc should be excluded"
    assert 'age' not in selected_features, "age should be excluded"
    print("✓ Leakage features (sc, age) excluded")
    
    # Create established split
    X_train, X_val, X_test, y_train, y_val, y_test = create_established_split(df, selected_features)
    
    # Create preprocessing pipeline
    preprocessing_pipeline = create_preprocessing_pipeline(X_train, selected_features)
    
    # Apply preprocessing
    X_train_processed = apply_preprocessing(X_train, preprocessing_pipeline)
    X_val_processed = apply_preprocessing(X_val, preprocessing_pipeline)
    X_test_processed = apply_preprocessing(X_test, preprocessing_pipeline)
    
    print("\n✓ Preprocessing applied to all splits")
    print("✓ Preprocessing fitted on training data only")
    
    # Train models with CV
    models_to_train = ['Logistic Regression', 'Random Forest', 'MLP']
    cv_results = {}
    trained_models = {}
    
    for model_name in models_to_train:
        model, cv_result = train_model_with_cv(X_train_processed, y_train, model_name)
        trained_models[model_name] = model
        cv_results[model_name] = cv_result
    
    # Evaluate on validation set
    print("\n" + "="*60)
    print("VALIDATION SET EVALUATION")
    print("="*60)
    
    val_results = {}
    for model_name in models_to_train:
        model = trained_models[model_name]
        results = evaluate_model(model, X_val_processed, y_val, model_name, "Validation")
        val_results[model_name] = results
    
    # Select best model based on validation Macro-F1
    best_model_name = max(val_results, key=lambda x: val_results[x]['macro_f1'])
    best_model = trained_models[best_model_name]
    
    print(f"\n" + "="*60)
    print(f"SELECTED MODEL: {best_model_name}")
    print("="*60)
    print(f"Validation Macro-F1: {val_results[best_model_name]['macro_f1']:.4f}")
    
    # Final test evaluation
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    
    test_results = evaluate_model(best_model, X_test_processed, y_test, best_model_name, "Test")
    
    # Save results
    all_results = {
        'cv_results': cv_results,
        'val_results': val_results,
        'test_results': test_results,
        'best_model': best_model_name,
        'selected_features': selected_features,
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test)
    }
    
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = os.path.join(results_dir, 'classical_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Save preprocessing pipeline
    preprocessing_path = os.path.join(project_root, 'models/classical_preprocessor.pkl')
    with open(preprocessing_path, 'wb') as f:
        pickle.dump(preprocessing_pipeline, f)
    
    print(f"Preprocessing pipeline saved to: {preprocessing_path}")
    
    # Save best model
    model_path = os.path.join(project_root, 'models/best_classical_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"Best model saved to: {model_path}")
    
    print("\n" + "="*60)
    print("PHASE 5 COMPLETE")
    print("="*60)
    print("Next: Create visualizations and detailed report")

if __name__ == "__main__":
    main()
