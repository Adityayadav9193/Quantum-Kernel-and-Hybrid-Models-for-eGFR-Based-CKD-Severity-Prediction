"""
Repeated Stratified 5-Fold Cross-Validation Evaluation Pipeline.
Evaluates eGFR 5-Category Prediction on 374-Sample CKD Dataset.

Methodology:
- Total Dataset: 374 observed-only samples.
- Locked Test Split: 20% (75 samples), stratified, random_state=42.
- Development Set: 299 samples.
- Repeated Stratified 5-Fold CV: 5 repeats (25 folds total), varied random_states.
- Preprocessing: Imputation (median) and Standardization strictly inside each fold via sklearn Pipeline.
- Metrics Reported:
  1. Macro-F1
  2. Accuracy
  3. Quadratic Weighted Kappa (QWK)
  4. Mean Absolute Error (MAE) of category index
  5. Within ±1 category accuracy
- Uncertainty: Mean ± Std and 95% Bootstrap Confidence Intervals (B=2000).
- Locked Test Evaluation: Final pipeline fit on 299 dev samples, evaluated once on 75 locked test samples.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    cohen_kappa_score,
    mean_absolute_error,
    confusion_matrix,
    classification_report
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

FEATURES = ["bgr", "bu", "pot", "hemo", "pcv", "rbcc", "sg", "sod"]
TARGET = "egfr_category"
NUM_CLASSES = 5
N_SPLITS = 5
N_REPEATS = 5
TOTAL_FOLDS = N_SPLITS * N_REPEATS
BOOTSTRAP_ROUNDS = 2000
RANDOM_SEED = 42


def compute_metrics(y_true, y_pred):
    """Compute all required metrics for 5-class ordinal eGFR prediction."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    mae = mean_absolute_error(y_true, y_pred)
    within_1 = float(np.mean(np.abs(y_true - y_pred) <= 1))
    
    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
        "quadratic_weighted_kappa": float(qwk),
        "mae_category": float(mae),
        "within_1_category_accuracy": float(within_1)
    }


def bootstrap_ci_mean(values, rounds=BOOTSTRAP_ROUNDS, seed=RANDOM_SEED):
    """Compute 95% percentile bootstrap confidence interval for the mean of values."""
    np.random.seed(seed)
    values = np.asarray(values)
    n = len(values)
    boot_means = [np.mean(np.random.choice(values, size=n, replace=True)) for _ in range(rounds)]
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return ci_low, ci_high


def bootstrap_ci_test_samples(y_true, y_pred, metric_name, rounds=BOOTSTRAP_ROUNDS, seed=RANDOM_SEED):
    """Compute 95% percentile bootstrap CI by resampling test samples."""
    np.random.seed(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    boot_scores = []
    
    for _ in range(rounds):
        idx = np.random.choice(n, size=n, replace=True)
        yt_b, yp_b = y_true[idx], y_pred[idx]
        m = compute_metrics(yt_b, yp_b)[metric_name]
        boot_scores.append(m)
        
    ci_low = float(np.percentile(boot_scores, 2.5))
    ci_high = float(np.percentile(boot_scores, 97.5))
    return ci_low, ci_high


def main():
    print("=" * 70)
    print("REPEATED STRATIFIED 5-FOLD CV EVALUATION PIPELINE")
    print("=" * 70)
    
    # 1. Load data
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "ckd_egfr_categories_observed_only.csv")
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    
    X = df[FEATURES]
    y = df[TARGET]
    
    # 2. Split: 20% locked test set (75 samples), 80% development set (299 samples)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Locked Test Set   : {len(X_test)} samples (20.0%) - KEPT UNTOUCHED")
    print(f"Development Set   : {len(X_dev)} samples (80.0%)")
    print(f"Target distribution in test:\n{y_test.value_counts().sort_index().to_dict()}")
    
    # 3. Models to evaluate with leakage-free Pipeline
    model_configs = {
        "Logistic Regression (Balanced, Baseline)": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED, solver="lbfgs"
        ),
        "Random Forest (Balanced, Comparison)": RandomForestClassifier(
            class_weight="balanced", n_estimators=100, random_state=RANDOM_SEED
        )
    }
    
    # Repeated Stratified K-Fold generator (5 repeats x 5 folds = 25 folds)
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)
    
    all_results = {}
    
    for model_name, clf in model_configs.items():
        print("\n" + "-" * 70)
        print(f"Running Repeated Stratified CV for: {model_name}")
        print("-" * 70)
        
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", clf)
        ])
        
        fold_metrics = {
            "macro_f1": [],
            "accuracy": [],
            "quadratic_weighted_kappa": [],
            "mae_category": [],
            "within_1_category_accuracy": []
        }
        
        # Cross-validation loop
        for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X_dev, y_dev)):
            X_tr, y_tr = X_dev.iloc[train_idx], y_dev.iloc[train_idx]
            X_va, y_va = X_dev.iloc[val_idx], y_dev.iloc[val_idx]
            
            # Fit pipeline strictly on fold training data
            pipeline.fit(X_tr, y_tr)
            preds = pipeline.predict(X_va)
            
            scores = compute_metrics(y_va, preds)
            for k, v in scores.items():
                fold_metrics[k].append(v)
                
        # Aggregate CV statistics
        cv_summary = {}
        print(f"\nResults across {TOTAL_FOLDS} folds (5 repeats x 5 folds):")
        for metric, vals in fold_metrics.items():
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            ci_low, ci_high = bootstrap_ci_mean(vals, rounds=BOOTSTRAP_ROUNDS)
            cv_summary[metric] = {
                "mean": mean_val,
                "std": std_val,
                "ci_95": [ci_low, ci_high],
                "fold_values": vals
            }
            print(f"  {metric:<28}: {mean_val:.4f} ± {std_val:.4f} | 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
            
        # 4. Final Fit on all 299 development samples -> Evaluate on locked 75-sample test set
        pipeline.fit(X_dev, y_dev)
        test_preds = pipeline.predict(X_test)
        test_scores = compute_metrics(y_test, test_preds)
        test_cm = confusion_matrix(y_test, test_preds).tolist()
        
        test_summary = {}
        print(f"\nLocked Test Evaluation (75 samples):")
        for metric, val in test_scores.items():
            ci_low, ci_high = bootstrap_ci_test_samples(y_test, test_preds, metric, rounds=BOOTSTRAP_ROUNDS)
            test_summary[metric] = {
                "point_estimate": val,
                "ci_95": [ci_low, ci_high]
            }
            print(f"  {metric:<28}: {val:.4f} | 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
            
        all_results[model_name] = {
            "cv_repeated_kfold": cv_summary,
            "locked_test_evaluation": {
                "metrics": test_summary,
                "confusion_matrix": test_cm,
                "classification_report": classification_report(y_test, test_preds, output_dict=True, zero_division=0)
            }
        }
        
    # Save results
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, "repeated_cv_results.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved repeated CV results to: {out_file}")

if __name__ == "__main__":
    main()
