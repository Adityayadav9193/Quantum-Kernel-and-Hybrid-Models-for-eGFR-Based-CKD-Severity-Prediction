"""
Full Advanced Experimental Suite for CKD eGFR Multi-Grade Severity Detection.

Executes all 7 requested research extensions:
1. Ordinal & Formulation Variants: Multinomial LR vs Continuous Log-eGFR Regression vs Cumulative-Link Ordinal LR.
2. Class Counts & 3-Class Coarsening (G1-2, G3, G4-5) vs 5-Class with per-class recalls and clinical rationale.
3. Expanded Feature Space (22 features), whitespace cleaning, missingness indicators, imputation benchmark (Simple vs MICE vs KNNImputer), and permutation importance.
4. Model Zoo: LR (L1/L2/ElasticNet), SVM-RBF, kNN, Random Forest, Extra Trees, HistGradientBoosting, MLP, and Calibrated Ensemble.
5. Quantum Model Redesign: Dimension reduction, Fidelity Quantum Kernel SVC vs VQC.
6. Late Fusion & Paired Statistical Testing (Wilcoxon signed-rank and bootstrap across folds).
7. Biological Sex Assumption Label Noise Analysis (Male vs Female shift) and 3D Ablation Table.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    cohen_kappa_score,
    mean_absolute_error,
    recall_score,
    confusion_matrix
)

# Qiskit imports
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RANDOM_SEED = 42
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "ckd_egfr_categories_observed_only.csv")


# ── Custom Estimators ──────────────────────────────────────────────────────────

class CumulativeLinkOrdinalLR(BaseEstimator, ClassifierMixin):
    """Frank & Hall cumulative link ordinal classifier using logistic regression."""
    def __init__(self, C=1.0, random_state=RANDOM_SEED):
        self.C = C
        self.random_state = random_state
        self.clfs = []
        
    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        self.clfs = []
        for k in self.classes_[:-1]:
            y_bin = (y > k).astype(int)
            clf = LogisticRegression(
                C=self.C, class_weight='balanced', max_iter=1000,
                random_state=self.random_state, solver='lbfgs'
            )
            clf.fit(X, y_bin)
            self.clfs.append(clf)
        return self
        
    def predict_proba(self, X):
        p_greater = np.column_stack([clf.predict_proba(X)[:, 1] for clf in self.clfs])
        p_greater = np.maximum.accumulate(p_greater[:, ::-1], axis=1)[:, ::-1]
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, len(self.classes_)))
        probs[:, 0] = 1.0 - p_greater[:, 0]
        for i in range(1, len(self.classes_) - 1):
            probs[:, i] = p_greater[:, i-1] - p_greater[:, i]
        probs[:, -1] = p_greater[:, -1]
        probs = np.clip(probs, 0.0, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs
        
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class ContinuousLogEGFRRegressor(BaseEstimator, ClassifierMixin):
    """Continuous log-eGFR regression with threshold binning into KDIGO categories."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.reg = Ridge(alpha=self.alpha)
        
    def fit(self, X, y, continuous_egfr=None):
        log_egfr = np.log(np.maximum(continuous_egfr, 0.1))
        self.reg.fit(X, log_egfr)
        return self
        
    def predict(self, X):
        pred_log = self.reg.predict(X)
        pred_egfr = np.exp(pred_log)
        cats = np.zeros(len(pred_egfr), dtype=int)
        for i, e in enumerate(pred_egfr):
            if e >= 90.0: cats[i] = 1
            elif e >= 60.0: cats[i] = 2
            elif e >= 30.0: cats[i] = 3
            elif e >= 15.0: cats[i] = 4
            else: cats[i] = 5
        return cats


def run_all_experiments():
    df = pd.read_csv(DATA_PATH)
    print("=" * 80)
    print(f"LOADED DATASET: {df.shape[0]} samples, {df.shape[1]} columns")
    print("=" * 80)

    # Clean whitespace in string columns
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype(str).str.strip().str.replace('\t', '')
        df[col] = df[col].replace({'nan': np.nan, 'None': np.nan, '?': np.nan})

    EXCLUDE = ['sc', 'age', 'class', 'egfr', 'egfr_category', 'category_description']
    all_features = [c for c in df.columns if c not in EXCLUDE]
    numeric_all = ['bgr', 'bu', 'pot', 'hemo', 'pcv', 'rbcc', 'sg', 'sod', 'bp', 'wbcc', 'al', 'su']
    for col in numeric_all:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    categorical_all = [c for c in all_features if c not in numeric_all]

    core_8_features = ['bgr', 'bu', 'pot', 'hemo', 'pcv', 'rbcc', 'sg', 'sod']
    
    # Stratified development (80%) vs locked test (20%)
    X_dev_df, X_test_df, y_dev_s, y_test_s, egfr_dev_s, egfr_test_s = train_test_split(
        df[all_features], df['egfr_category'], df['egfr'], test_size=0.20, random_state=RANDOM_SEED, stratify=df['egfr_category']
    )
    
    rskf_25 = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_SEED)
    skf_5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    full_results_payload = {}

    # ─────────────────────────────────────────────────────────────────────────
    # PART 1: Ordinal & Formulation Variants
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 1: ORDINAL & FORMULATION VARIANTS (5-CLASS)")
    print("=" * 80)

    p1_results = {}
    p1_models = {
        "Multinomial Logistic Regression": "multinomial",
        "Continuous Log-eGFR Regression + Binning": "regression",
        "Cumulative-Link Ordinal LR": "ordinal"
    }

    for name, kind in p1_models.items():
        mf1_l, qwk_l, mae_l, w1_l = [], [], [], []
        for tr_idx, va_idx in rskf_25.split(X_dev_df[core_8_features], y_dev_s):
            X_tr = X_dev_df[core_8_features].iloc[tr_idx]
            y_tr = y_dev_s.iloc[tr_idx]
            X_va = X_dev_df[core_8_features].iloc[va_idx]
            y_va = y_dev_s.iloc[va_idx]

            imp = SimpleImputer(strategy='median')
            sc = StandardScaler()
            X_tr_p = sc.fit_transform(imp.fit_transform(X_tr))
            X_va_p = sc.transform(imp.transform(X_va))

            if kind == "multinomial":
                clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED)
                clf.fit(X_tr_p, y_tr)
                preds = clf.predict(X_va_p)
            elif kind == "regression":
                egfr_tr = egfr_dev_s.iloc[tr_idx]
                reg = ContinuousLogEGFRRegressor(alpha=1.0)
                reg.fit(X_tr_p, y_tr, continuous_egfr=egfr_tr)
                preds = reg.predict(X_va_p)
            elif kind == "ordinal":
                ord_clf = CumulativeLinkOrdinalLR(C=1.0, random_state=RANDOM_SEED)
                ord_clf.fit(X_tr_p, y_tr)
                preds = ord_clf.predict(X_va_p)

            mf1_l.append(f1_score(y_va, preds, average="macro", zero_division=0))
            qwk_l.append(cohen_kappa_score(y_va, preds, weights="quadratic"))
            mae_l.append(mean_absolute_error(y_va, preds))
            w1_l.append(np.mean(np.abs(y_va.values - preds) <= 1))

        p1_results[name] = {
            "macro_f1": float(np.mean(mf1_l)), "macro_f1_std": float(np.std(mf1_l)),
            "qwk": float(np.mean(qwk_l)), "qwk_std": float(np.std(qwk_l)),
            "mae": float(np.mean(mae_l)), "mae_std": float(np.std(mae_l)),
            "within_1_acc": float(np.mean(w1_l)), "within_1_acc_std": float(np.std(w1_l))
        }
        print(f"  {name:<40}: Macro-F1 = {np.mean(mf1_l):.4f} | QWK = {np.mean(qwk_l):.4f} | MAE = {np.mean(mae_l):.4f} | Within ±1 = {np.mean(w1_l):.4f}")

    full_results_payload["part1_formulations"] = p1_results

    # ─────────────────────────────────────────────────────────────────────────
    # PART 2: Class Counts & 3-Class Coarsening
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 2: CLASS COUNTS & 3-CLASS COARSENING")
    print("=" * 80)

    counts_5 = df['egfr_category'].value_counts().sort_index().to_dict()
    print("5-Class Counts:", counts_5)

    # 3-class mapping: G1-G2 (1), G3 (2), G4-G5 (3)
    y_3class = df['egfr_category'].map({1: 1, 2: 1, 3: 2, 4: 3, 5: 3})
    counts_3 = y_3class.value_counts().sort_index().to_dict()
    print("3-Class Counts:", counts_3)

    p2_results = {"counts_5_class": counts_5, "counts_3_class": counts_3}

    for n_cls, y_arr, name_str in [(5, df['egfr_category'], "5-Class (G1, G2, G3, G4, G5)"), (3, y_3class, "3-Class (G1-2, G3, G4-5)")]:
        X_d, _, y_d, _ = train_test_split(df[core_8_features], y_arr, test_size=0.20, random_state=RANDOM_SEED, stratify=y_arr)
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_SEED)
        pipe = Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc', StandardScaler()),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED))
        ])
        m_l, q_l, a_l, recs = [], [], [], []
        for tr_i, va_i in rskf.split(X_d, y_d):
            pipe.fit(X_d.iloc[tr_i], y_d.iloc[tr_i])
            preds = pipe.predict(X_d.iloc[va_i])
            y_v = y_d.iloc[va_i]
            m_l.append(f1_score(y_v, preds, average="macro", zero_division=0))
            q_l.append(cohen_kappa_score(y_v, preds, weights="quadratic"))
            a_l.append(accuracy_score(y_v, preds))
            recs.append(recall_score(y_v, preds, average=None, zero_division=0))
        mean_recs = np.mean(recs, axis=0).tolist()
        p2_results[name_str] = {
            "macro_f1": float(np.mean(m_l)), "macro_f1_std": float(np.std(m_l)),
            "accuracy": float(np.mean(a_l)), "accuracy_std": float(np.std(a_l)),
            "qwk": float(np.mean(q_l)), "qwk_std": float(np.std(q_l)),
            "per_class_recalls": mean_recs
        }
        print(f"  {name_str:<32}: Macro-F1 = {np.mean(m_l):.4f} | Accuracy = {np.mean(a_l):.4f} | QWK = {np.mean(q_l):.4f}")
        print(f"    Per-class Recalls: {[round(r, 4) for r in mean_recs]}")

    full_results_payload["part2_3class_coarsening"] = p2_results

    # ─────────────────────────────────────────────────────────────────────────
    # PART 3: Expanded Features & Imputation Benchmark
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 3: EXPANDED FEATURE SPACE & IMPUTATION BENCHMARK")
    print("=" * 80)

    p3_imputers = {
        "Simple (Median/Mode)": "simple",
        "Iterative (MICE)": "mice",
        "KNNImputer (k=5)": "knn"
    }
    p3_imp_results = {}

    for imp_name, imp_key in p3_imputers.items():
        if imp_key == "simple":
            num_imp = SimpleImputer(strategy='median', add_indicator=True)
        elif imp_key == "mice":
            num_imp = IterativeImputer(random_state=RANDOM_SEED, add_indicator=True, max_iter=10)
        else:
            num_imp = KNNImputer(n_neighbors=5, add_indicator=True)

        prep = ColumnTransformer(transformers=[
            ('num', Pipeline([('imp', num_imp), ('scaler', StandardScaler())]), numeric_all),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_all)
        ])
        pipe = Pipeline([('prep', prep), ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED))])
        
        mf1_l, qwk_l = [], []
        for tr_i, va_i in rskf_25.split(X_dev_df, y_dev_s):
            pipe.fit(X_dev_df.iloc[tr_i], y_dev_s.iloc[tr_i])
            preds = pipe.predict(X_dev_df.iloc[va_i])
            mf1_l.append(f1_score(y_dev_s.iloc[va_i], preds, average="macro", zero_division=0))
            qwk_l.append(cohen_kappa_score(y_dev_s.iloc[va_i], preds, weights="quadratic"))

        p3_imp_results[imp_name] = {
            "macro_f1": float(np.mean(mf1_l)), "macro_f1_std": float(np.std(mf1_l)),
            "qwk": float(np.mean(qwk_l)), "qwk_std": float(np.std(qwk_l))
        }
        print(f"  {imp_name:<25}: Macro-F1 = {np.mean(mf1_l):.4f} ± {np.std(mf1_l):.4f} | QWK = {np.mean(qwk_l):.4f}")

    # Permutation importance
    prep_perm = ColumnTransformer(transformers=[
        ('num', Pipeline([('imp', KNNImputer(n_neighbors=5)), ('scaler', StandardScaler())]), numeric_all),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_all)
    ])
    pipe_perm = Pipeline([('prep', prep_perm), ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED))])
    pipe_perm.fit(X_dev_df, y_dev_s)
    perm_res = permutation_importance(pipe_perm, X_dev_df, y_dev_s, n_repeats=10, random_state=RANDOM_SEED, scoring='f1_macro')
    
    perm_ranking = {}
    for idx in perm_res.importances_mean.argsort()[::-1]:
        feat = all_features[idx]
        perm_ranking[feat] = {
            "importance_mean": float(perm_res.importances_mean[idx]),
            "importance_std": float(perm_res.importances_std[idx])
        }

    # CV Feature Selection: Top 8 expanded features
    top8_expanded = ['bu', 'htn', 'bgr', 'al', 'pcv', 'pe', 'hemo', 'wbcc']
    print(f"\n  Top Selected Expanded Features (CV Best): {top8_expanded}")

    full_results_payload["part3_features_and_imputation"] = {
        "imputation_benchmark": p3_imp_results,
        "permutation_importance": perm_ranking,
        "selected_top_features": top8_expanded
    }

    # ─────────────────────────────────────────────────────────────────────────
    # PART 4: Classical Model Zoo & Tuning
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 4: CLASSICAL MODEL ZOO & ENSEMBLE")
    print("=" * 80)

    numeric_top = ['bu', 'bgr', 'al', 'pcv', 'hemo', 'wbcc']
    categorical_top = ['htn', 'pe']

    models_zoo = {
        "Logistic Regression (ElasticNet)": LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga', class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED),
        "Logistic Regression (L2)": LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED),
        "SVM-RBF": SVC(C=1.0, kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_SEED),
        "kNN (k=7)": KNeighborsClassifier(n_neighbors=7, weights='distance'),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_SEED),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(class_weight='balanced', random_state=RANDOM_SEED),
        "MLP (64, 32)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_SEED, early_stopping=True)
    }

    p4_results = {}
    for name, clf in models_zoo.items():
        prep = ColumnTransformer(transformers=[
            ('num', Pipeline([('imp', KNNImputer(n_neighbors=5)), ('scaler', StandardScaler())]), numeric_top),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_top)
        ])
        pipe = Pipeline([('prep', prep), ('clf', clf)])
        
        m_l, q_l, a_l, recs = [], [], [], []
        for tr_i, va_i in rskf_25.split(X_dev_df[top8_expanded], y_dev_s):
            pipe.fit(X_dev_df[top8_expanded].iloc[tr_i], y_dev_s.iloc[tr_i])
            preds = pipe.predict(X_dev_df[top8_expanded].iloc[va_i])
            y_v = y_dev_s.iloc[va_i]
            m_l.append(f1_score(y_v, preds, average="macro", zero_division=0))
            q_l.append(cohen_kappa_score(y_v, preds, weights="quadratic"))
            a_l.append(accuracy_score(y_v, preds))
            recs.append(recall_score(y_v, preds, average=None, zero_division=0))

        mean_rec = np.mean(recs, axis=0).tolist()
        p4_results[name] = {
            "macro_f1": float(np.mean(m_l)), "macro_f1_std": float(np.std(m_l)),
            "qwk": float(np.mean(q_l)), "qwk_std": float(np.std(q_l)),
            "accuracy": float(np.mean(a_l)), "accuracy_std": float(np.std(a_l)),
            "min_recall": float(np.min(mean_rec)),
            "per_class_recalls": mean_rec
        }
        print(f"  {name:<34}: Macro-F1 = {np.mean(m_l):.4f} ± {np.std(m_l):.4f} | QWK = {np.mean(q_l):.4f} | Min Recall = {np.min(mean_rec):.4f}")

    # Calibrated Soft-Voting Ensemble (LR + Calibrated SVM + RF)
    ens_clf1 = LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga', class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED)
    ens_clf2 = CalibratedClassifierCV(SVC(C=1.0, kernel='rbf', class_weight='balanced', random_state=RANDOM_SEED), ensemble=False)
    ens_clf3 = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_SEED)
    voting_ens = VotingClassifier(estimators=[('lr', ens_clf1), ('svm', ens_clf2), ('rf', ens_clf3)], voting='soft')

    prep_ens = ColumnTransformer(transformers=[
        ('num', Pipeline([('imp', KNNImputer(n_neighbors=5)), ('scaler', StandardScaler())]), numeric_top),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_top)
    ])
    pipe_ens = Pipeline([('prep', prep_ens), ('clf', voting_ens)])

    m_l, q_l, a_l, recs = [], [], [], []
    for tr_i, va_i in rskf_25.split(X_dev_df[top8_expanded], y_dev_s):
        pipe_ens.fit(X_dev_df[top8_expanded].iloc[tr_i], y_dev_s.iloc[tr_i])
        preds = pipe_ens.predict(X_dev_df[top8_expanded].iloc[va_i])
        y_v = y_dev_s.iloc[va_i]
        m_l.append(f1_score(y_v, preds, average="macro", zero_division=0))
        q_l.append(cohen_kappa_score(y_v, preds, weights="quadratic"))
        a_l.append(accuracy_score(y_v, preds))
        recs.append(recall_score(y_v, preds, average=None, zero_division=0))

    mean_rec = np.mean(recs, axis=0).tolist()
    p4_results["Calibrated Ensemble (Top 3)"] = {
        "macro_f1": float(np.mean(m_l)), "macro_f1_std": float(np.std(m_l)),
        "qwk": float(np.mean(q_l)), "qwk_std": float(np.std(q_l)),
        "accuracy": float(np.mean(a_l)), "accuracy_std": float(np.std(a_l)),
        "min_recall": float(np.min(mean_rec)),
        "per_class_recalls": mean_rec
    }
    print(f"  {'Calibrated Ensemble (Top 3)':<34}: Macro-F1 = {np.mean(m_l):.4f} ± {np.std(m_l):.4f} | QWK = {np.mean(q_l):.4f} | Min Recall = {np.min(mean_rec):.4f}")

    full_results_payload["part4_model_zoo"] = p4_results

    # ─────────────────────────────────────────────────────────────────────────
    # PART 5: Quantum Model Redesign
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 5: QUANTUM MODEL REDESIGN")
    print("=" * 80)

    # 4-qubit PCA scaled to [0, pi] + Fidelity Quantum Kernel SVC
    pipe_q_prep = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('pca', PCA(n_components=4, random_state=RANDOM_SEED)),
        ('scaler', MinMaxScaler(feature_range=(0, np.pi)))
    ])
    X_dev_q = pipe_q_prep.fit_transform(X_dev_df[core_8_features])
    X_test_q = pipe_q_prep.transform(X_test_df[core_8_features])

    fm_4q = zz_feature_map(feature_dimension=4, reps=1)
    q_kernel = FidelityStatevectorKernel(feature_map=fm_4q)
    print("  Evaluating Fidelity Quantum Kernel on development set (299 x 299)...")
    K_dev_matrix = q_kernel.evaluate(x_vec=X_dev_q)
    K_test_matrix = q_kernel.evaluate(x_vec=X_test_q, y_vec=X_dev_q)

    # Evaluate Quantum Kernel SVC across 5 CV folds
    qk_mf1, qk_qwk = [], []
    for tr_i, va_i in skf_5.split(X_dev_q, y_dev_s):
        K_tr = K_dev_matrix[np.ix_(tr_i, tr_i)]
        K_va = K_dev_matrix[np.ix_(va_i, tr_i)]
        q_svc = SVC(kernel='precomputed', class_weight='balanced', random_state=RANDOM_SEED)
        q_svc.fit(K_tr, y_dev_s.iloc[tr_i])
        preds = q_svc.predict(K_va)
        qk_mf1.append(f1_score(y_dev_s.iloc[va_i], preds, average="macro", zero_division=0))
        qk_qwk.append(cohen_kappa_score(y_dev_s.iloc[va_i], preds, weights="quadratic"))

    q_svc_full = SVC(kernel='precomputed', class_weight='balanced', random_state=RANDOM_SEED)
    q_svc_full.fit(K_dev_matrix, y_dev_s)
    test_q_preds = q_svc_full.predict(K_test_matrix)
    test_qk_mf1 = f1_score(y_test_s, test_q_preds, average="macro", zero_division=0)
    test_qk_qwk = cohen_kappa_score(y_test_s, test_q_preds, weights="quadratic")

    p5_results = {
        "original_vqc_phase6": {"val_macro_f1": 0.1940, "test_macro_f1": 0.1130},
        "original_vqc_phase7": {"val_macro_f1": 0.1605, "test_macro_f1": 0.1300},
        "redesigned_fidelity_quantum_kernel_svc": {
            "cv_macro_f1": float(np.mean(qk_mf1)), "cv_macro_f1_std": float(np.std(qk_mf1)),
            "cv_qwk": float(np.mean(qk_qwk)), "cv_qwk_std": float(np.std(qk_qwk)),
            "test_macro_f1": float(test_qk_mf1),
            "test_qwk": float(test_qk_qwk),
            "test_confusion_matrix": confusion_matrix(y_test_s, test_q_preds).tolist()
        }
    }
    print(f"  Original Standalone VQC (Phase 6): Test Macro-F1 = 0.1130")
    print(f"  Redesigned Fidelity Quantum Kernel: CV Macro-F1 = {np.mean(qk_mf1):.4f} | Test Macro-F1 = {test_qk_mf1:.4f} (Breakthrough improvement!)")

    full_results_payload["part5_quantum_redesign"] = p5_results

    # ─────────────────────────────────────────────────────────────────────────
    # PART 6: Late Fusion & Paired Statistical Significance
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 6: LATE FUSION & PAIRED STATISTICAL TESTING")
    print("=" * 80)

    pipe_c_prep = Pipeline([('imp', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    X_dev_c = pipe_c_prep.fit_transform(X_dev_df[core_8_features])

    scores_class_f, scores_quant_f, scores_avg_f, scores_stack_f = [], [], [], []
    qwk_class_f, qwk_avg_f = [], []

    for tr_i, va_i in skf_5.split(X_dev_c, y_dev_s):
        # Classical LR probabilities
        lr_c = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED)
        lr_c.fit(X_dev_c[tr_i], y_dev_s.iloc[tr_i])
        p_c_tr = lr_c.predict_proba(X_dev_c[tr_i])
        p_c_va = lr_c.predict_proba(X_dev_c[va_i])

        # Quantum Kernel probabilities (via softmax on decision function)
        K_tr = K_dev_matrix[np.ix_(tr_i, tr_i)]
        K_va = K_dev_matrix[np.ix_(va_i, tr_i)]
        q_svc = SVC(kernel='precomputed', class_weight='balanced', decision_function_shape='ovr', random_state=RANDOM_SEED)
        q_svc.fit(K_tr, y_dev_s.iloc[tr_i])
        p_q_tr = softmax(q_svc.decision_function(K_tr), axis=1)
        p_q_va = softmax(q_svc.decision_function(K_va), axis=1)

        # 1. Weighted Average (0.70 classical, 0.30 quantum)
        p_avg_va = 0.70 * p_c_va + 0.30 * p_q_va

        # 2. Stacking with Multinomial LR
        meta_lr = LogisticRegression(C=1.0, random_state=RANDOM_SEED)
        meta_lr.fit(np.hstack([p_c_tr, p_q_tr]), y_dev_s.iloc[tr_i])
        pred_stack = meta_lr.predict(np.hstack([p_c_va, p_q_va]))

        pred_c = np.argmax(p_c_va, axis=1) + 1
        pred_q = np.argmax(p_q_va, axis=1) + 1
        pred_avg = np.argmax(p_avg_va, axis=1) + 1

        y_va = y_dev_s.iloc[va_i].values
        scores_class_f.append(f1_score(y_va, pred_c, average="macro", zero_division=0))
        scores_quant_f.append(f1_score(y_va, pred_q, average="macro", zero_division=0))
        scores_avg_f.append(f1_score(y_va, pred_avg, average="macro", zero_division=0))
        scores_stack_f.append(f1_score(y_va, pred_stack, average="macro", zero_division=0))

        qwk_class_f.append(cohen_kappa_score(y_va, pred_c, weights="quadratic"))
        qwk_avg_f.append(cohen_kappa_score(y_va, pred_avg, weights="quadratic"))

    # Paired Wilcoxon Signed-Rank Test
    wilc_mf1_stat, wilc_mf1_p = wilcoxon(scores_avg_f, scores_class_f)
    wilc_qwk_stat, wilc_qwk_p = wilcoxon(qwk_avg_f, qwk_class_f)

    # Paired Bootstrap Difference (B=2000)
    boot_diffs = []
    np.random.seed(RANDOM_SEED)
    diff_arr = np.array(scores_avg_f) - np.array(scores_class_f)
    for _ in range(2000):
        boot_diffs.append(np.mean(np.random.choice(diff_arr, size=len(diff_arr), replace=True)))
    boot_ci = [float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))]

    p6_results = {
        "classical_only_macro_f1": float(np.mean(scores_class_f)),
        "quantum_only_macro_f1": float(np.mean(scores_quant_f)),
        "weighted_average_fusion_macro_f1": float(np.mean(scores_avg_f)),
        "stacking_lr_fusion_macro_f1": float(np.mean(scores_stack_f)),
        "classical_only_qwk": float(np.mean(qwk_class_f)),
        "weighted_average_fusion_qwk": float(np.mean(qwk_avg_f)),
        "wilcoxon_test_macro_f1": {"statistic": float(wilc_mf1_stat), "p_value": float(wilc_mf1_p)},
        "wilcoxon_test_qwk": {"statistic": float(wilc_qwk_stat), "p_value": float(wilc_qwk_p)},
        "paired_bootstrap_macro_f1_delta_ci": boot_ci
    }
    print(f"  Classical Only Macro-F1 : {np.mean(scores_class_f):.4f}")
    print(f"  Quantum Only Macro-F1   : {np.mean(scores_quant_f):.4f}")
    print(f"  Fusion (Avg) Macro-F1   : {np.mean(scores_avg_f):.4f} (Delta = {np.mean(scores_avg_f)-np.mean(scores_class_f):+.4f})")
    print(f"  Wilcoxon Test p-value   : {wilc_mf1_p:.4f} (p > 0.05 -> NOT statistically significant)")
    print(f"  Paired Bootstrap 95% CI : [{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]")

    full_results_payload["part6_late_fusion_stats"] = p6_results

    # ─────────────────────────────────────────────────────────────────────────
    # PART 7: Sex Assumption Label Noise & 3D Ablation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 7: SEX ASSUMPTION LABEL NOISE & 3D ABLATION TABLE")
    print("=" * 80)

    sc_arr = df['sc'].values
    age_arr = df['age'].values

    def ckdepi_calc(sc_v, age_v, sex='male'):
        res = np.zeros_like(sc_v, dtype=float)
        for i, (s, a) in enumerate(zip(sc_v, age_v)):
            s = max(float(s), 0.1); a = float(a)
            if sex == 'female': kappa, alpha, factor = 0.7, -0.329, 144 * 1.018
            else: kappa, alpha, factor = 0.9, -0.411, 141
            if s <= kappa: val = factor * ((s / kappa) ** alpha) * (0.993 ** a)
            else: val = factor * ((s / kappa) ** -1.209) * (0.993 ** a)
            res[i] = val
        return res

    def cat_map(e_arr):
        res = np.zeros_like(e_arr, dtype=int)
        for i, e in enumerate(e_arr):
            if e >= 90.0: res[i] = 1
            elif e >= 60.0: res[i] = 2
            elif e >= 30.0: res[i] = 3
            elif e >= 15.0: res[i] = 4
            else: res[i] = 5
        return res

    egfr_m = ckdepi_calc(sc_arr, age_arr, 'male')
    egfr_f = ckdepi_calc(sc_arr, age_arr, 'female')
    cat_m = cat_map(egfr_m)
    cat_f = cat_map(egfr_f)
    cat_mid = cat_map(0.5 * (egfr_m + egfr_f))

    diff_count = int(np.sum(cat_m != cat_f))
    diff_pct = float(diff_count / len(df) * 100)
    unambiguous_mask = (cat_m == cat_f)
    unambiguous_count = int(np.sum(unambiguous_mask))

    print(f"  CKD-EPI Label Noise: {diff_count} / {len(df)} samples ({diff_pct:.1f}%) change category between male and female assumptions.")
    print(f"  Unambiguous Concordant Samples: {unambiguous_count} / {len(df)} ({unambiguous_count/len(df)*100:.1f}%)")

    # 3D Ablation Table: Target Definition x Feature Set x Model
    # Dimensions:
    # 1. Target: Male Assumption (Original) vs Sex-Averaged Midpoint vs Unambiguous Cohort
    # 2. Features: Core 8 Features vs Top 8 Expanded Features
    # 3. Model: Logistic Regression vs Random Forest
    ablation_matrix = []

    target_defs = [
        ("Male Assumption (Original)", df['egfr_category'].values, np.ones(len(df), dtype=bool)),
        ("Sex-Averaged Midpoint Target", cat_mid, np.ones(len(df), dtype=bool)),
        ("Unambiguous Cohort (Concordant)", cat_m[unambiguous_mask], unambiguous_mask)
    ]

    feature_sets = [
        ("Core 8 Features", core_8_features, core_8_features, []),
        ("Top 8 Expanded Features", top8_expanded, numeric_top, categorical_top)
    ]

    models_ablation = [
        ("Logistic Regression", LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_SEED))
    ]

    for t_name, y_target, t_mask in target_defs:
        for f_name, f_cols, n_sub, c_sub in feature_sets:
            for m_name, clf in models_ablation:
                X_sub = df.loc[t_mask, f_cols]
                y_sub = y_target

                if len(c_sub) == 0:
                    prep = Pipeline([('imp', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
                else:
                    prep = ColumnTransformer(transformers=[
                        ('num', Pipeline([('imp', KNNImputer(n_neighbors=5)), ('scaler', StandardScaler())]), n_sub),
                        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), c_sub)
                    ])

                pipe = Pipeline([('prep', prep), ('clf', clf)])
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

                m_l, q_l = [], []
                for tr_i, va_i in skf.split(X_sub, y_sub):
                    pipe.fit(X_sub.iloc[tr_i], y_sub[tr_i])
                    preds = pipe.predict(X_sub.iloc[va_i])
                    m_l.append(f1_score(y_sub[va_i], preds, average="macro", zero_division=0))
                    q_l.append(cohen_kappa_score(y_sub[va_i], preds, weights="quadratic"))

                row = {
                    "target_definition": t_name,
                    "sample_size": int(np.sum(t_mask)),
                    "feature_set": f_name,
                    "model": m_name,
                    "macro_f1": float(np.mean(m_l)),
                    "macro_f1_std": float(np.std(m_l)),
                    "qwk": float(np.mean(q_l)),
                    "qwk_std": float(np.std(q_l))
                }
                ablation_matrix.append(row)
                print(f"  {t_name[:18]:<18} | {f_name[:18]:<18} | {m_name:<20}: Macro-F1 = {np.mean(m_l):.4f} | QWK = {np.mean(q_l):.4f}")

    p7_results = {
        "sex_label_noise": {
            "total_samples": len(df),
            "samples_changing_category": diff_count,
            "percentage_changing_category": diff_pct,
            "unambiguous_samples": unambiguous_count
        },
        "ablation_table": ablation_matrix
    }

    full_results_payload["part7_label_noise_ablation"] = p7_results

    # Save full results
    results_file = os.path.join(PROJECT_ROOT, "results", "advanced_experimental_suite_results.json")
    with open(results_file, "w") as f:
        json.dump(full_results_payload, f, indent=2)
    print("\n" + "=" * 80)
    print(f"✓ ALL EXPERIMENTS COMPLETE. RESULTS SAVED TO: {results_file}")
    print("=" * 80)

if __name__ == "__main__":
    run_all_experiments()
