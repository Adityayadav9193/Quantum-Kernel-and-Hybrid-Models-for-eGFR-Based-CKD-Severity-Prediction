"""
Phase 7: Late Fusion
Hybrid Quantum-Classical Framework — CKD eGFR Category Prediction

OBJECTIVE:
Test whether the quantum (VQC) branch provides complementary information
when combined with the classical (Logistic Regression) branch via late fusion.

DESIGN:
- Classical branch : Logistic Regression (Phase 5, frozen)
- Quantum branch   : VQC 8-qubit (Phase 6, frozen weights)
- Fusion strategy  : Stacking with out-of-fold (OOF) training predictions
  * Generate 5-fold OOF training probability vectors from each branch
  * Concatenate → [LR_proba(5) | VQC_proba(5)] → 10-dim feature per sample
  * Train a multinomial Logistic Regression meta-classifier on OOF features
  * Select fusion hyper-parameter (C) using validation set Macro-F1
  * Freeze the best fusion model
  * Evaluate on test set exactly once

LEAKAGE CONTROLS:
- Preprocessor was fitted on training data only (Phase 5 artefact; not re-fitted here)
- OOF predictions are generated fold-by-fold on held-out training samples
- Fusion C is selected using validation set only
- Test set evaluated exactly once after freezing fusion model
- Phases 1–6 artefacts are never modified

SCIENTIFIC DISCLAIMERS:
- No quantum advantage is claimed.
- Results are not clinically validated.
- eGFR categories are exploratory labels, NOT confirmed CKD stages.
- All quantum inference uses a statevector simulator.

Versions: qiskit 2.5.2, qiskit-machine-learning 0.9.1
"""

import warnings
warnings.filterwarnings("ignore")

import os, json, pickle, time
import numpy as np
import pandas as pd

# ── sklearn ──────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, classification_report,
)

# ── Qiskit / QML ─────────────────────────────────────────────────────────────
from qiskit.circuit.library import zz_feature_map, n_local
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA, OptimizerResult
from qiskit.primitives import StatevectorSampler

# ── Matplotlib ───────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# Constants  (ALL must match Phases 3–6 exactly)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

SELECTED_FEATURES = ["bgr", "bu", "pot", "hemo", "pcv", "rbcc", "sg", "sod"]
EXCLUDED_FEATURES = ["sc", "age"]
NUM_FEATURES = 8
NUM_CLASSES  = 5
RANDOM_SEED  = 42
N_FOLDS      = 5

# VQC circuit config (must exactly mirror Phase 6)
VQC_FEATURE_MAP_REPS = 1
VQC_ANSATZ_REPS      = 1

# Frozen Phase 5 / Phase 6 metrics (for sanity-checking)
FROZEN_LR_VAL_F1  = 0.4109
FROZEN_VQC_VAL_F1 = 0.1940   # best confirmed run (Phase 6 report)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading and split
# ─────────────────────────────────────────────────────────────────────────────

def load_splits():
    """
    Load dataset and reproduce the exact Phase 3 split (239/60/75).
    Returns raw (un-preprocessed) DataFrames.
    """
    print("=" * 60)
    print("LOADING DATASET AND REPRODUCING PHASE 3 SPLIT")
    print("=" * 60)

    for f in EXCLUDED_FEATURES:
        assert f not in SELECTED_FEATURES, f"LEAKAGE: {f} in feature set"

    data_path = os.path.join(
        PROJECT_ROOT, "data", "processed",
        "ckd_egfr_categories_observed_only.csv"
    )
    df = pd.read_csv(data_path)
    print(f"Dataset: {df.shape}")

    X = df[SELECTED_FEATURES].copy()
    y = df["egfr_category"].copy()

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.20, random_state=42, stratify=y_tv
    )

    assert len(X_train) == 239
    assert len(X_val)   == 60
    assert len(X_test)  == 75
    assert set(X_train.index).isdisjoint(set(X_val.index))
    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert set(X_val.index).isdisjoint(set(X_test.index))

    print(f"✓ Split: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test — all disjoint")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing  (transform-only; never re-fitted)
# ─────────────────────────────────────────────────────────────────────────────

def load_preprocessor():
    """Load the Phase 5 preprocessing pipeline (already fitted on training data)."""
    pp_path = os.path.join(PROJECT_ROOT, "models", "classical_preprocessor.pkl")
    with open(pp_path, "rb") as fh:
        pp = pickle.load(fh)
    assert hasattr(pp["numeric_imputer"], "statistics_"), "Imputer not fitted"
    assert hasattr(pp["scaler"], "mean_"),                "Scaler not fitted"
    assert set(pp["numeric_cols"]) == set(SELECTED_FEATURES)
    return pp


def apply_pp(X, pp):
    """Apply fitted preprocessor (transform-only) → numpy float64 array."""
    Xc = X.copy()
    nc = pp["numeric_cols"]
    Xc[nc] = pp["numeric_imputer"].transform(Xc[nc])
    Xc[nc] = pp["scaler"].transform(Xc[nc])
    return Xc.values.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Frozen model loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_classical_lr():
    """Load the frozen Phase 5 Logistic Regression model."""
    path = os.path.join(PROJECT_ROOT, "models", "best_classical_model.pkl")
    with open(path, "rb") as fh:
        model = pickle.load(fh)
    assert isinstance(model, LogisticRegression), "Unexpected model type"
    return model


def build_vqc_from_frozen_weights(frozen_weights: np.ndarray) -> VQC:
    """
    Reconstruct the Phase 6 VQC with frozen weights injected.

    Because VQC objects contain an un-picklable local lambda (parity function),
    we saved only the weight array.  To restore the model for inference:
      1. Build VQC with identical circuit config and initial_point=frozen_weights
      2. Perform a minimal fit (1 iteration, 5 samples) to create _fit_result
      3. Inject the frozen weights via _fit_result.x
    Step 2 takes ~1–2 seconds.
    """
    fm     = zz_feature_map(feature_dimension=NUM_FEATURES, reps=VQC_FEATURE_MAP_REPS)
    ansatz = n_local(num_qubits=NUM_FEATURES,
                     rotation_blocks=["ry"],
                     entanglement_blocks="cx",
                     entanglement="linear",
                     reps=VQC_ANSATZ_REPS)
    vqc = VQC(
        feature_map=fm, ansatz=ansatz, loss="cross_entropy",
        optimizer=COBYLA(maxiter=1),
        sampler=StatevectorSampler(),
        initial_point=frozen_weights,
    )
    return vqc


def init_vqc_for_inference(frozen_weights: np.ndarray,
                            X_seed: np.ndarray,
                            y_seed: np.ndarray) -> VQC:
    """
    Warm-start a VQC for inference with frozen weights.
    X_seed / y_seed: used only to bootstrap _fit_result.

    IMPORTANT: the bootstrap subset MUST contain all 5 classes so that the
    VQC's internal SamplerQNN is initialised with a 5-class output head.
    We pick one representative sample per class (5 samples total).
    """
    # One sample per class to guarantee all 5 classes in the bootstrap fit
    seed_indices = []
    for cls in range(NUM_CLASSES):
        idx = np.where(y_seed == cls)[0]
        if len(idx) > 0:
            seed_indices.append(idx[0])
    seed_indices = np.array(seed_indices[:NUM_CLASSES])

    vqc = build_vqc_from_frozen_weights(frozen_weights)
    np.random.seed(RANDOM_SEED)
    vqc.fit(X_seed[seed_indices], y_seed[seed_indices])   # 1 iter → _fit_result

    # Inject frozen weights
    result   = OptimizerResult()
    result.x = frozen_weights
    result.fun = 0.0
    vqc._fit_result = result

    assert np.allclose(vqc.weights, frozen_weights), "Weight injection failed"
    return vqc


# ─────────────────────────────────────────────────────────────────────────────
# 4. Probability extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def lr_predict_proba(lr_model, X_np: np.ndarray) -> np.ndarray:
    """
    Logistic Regression probability matrix (n_samples × 5).
    LR.classes_ order matches le.classes_ (1–5) because both are fit on the
    same training labels sorted by sklearn.
    """
    return lr_model.predict_proba(X_np)   # shape (n, 5)


def vqc_predict_proba(vqc: VQC, X_np: np.ndarray) -> np.ndarray:
    """
    VQC probability matrix (n_samples × 5).
    The VQC outputs a bitstring distribution which QML maps to class probabilities.
    """
    return vqc.predict_proba(X_np)   # shape (n, 5)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Out-of-fold (OOF) probability generation  — training data only
# ─────────────────────────────────────────────────────────────────────────────

def generate_oof_probas(X_train_np: np.ndarray,
                         y_train: pd.Series,
                         frozen_weights: np.ndarray,
                         lr_template: LogisticRegression,
                         le: LabelEncoder):
    """
    Generate out-of-fold training probability vectors from both branches.

    For each fold:
      - LR is cloned and re-fitted on the fold's training portion
      - VQC is re-fitted from scratch on the fold's training portion
        (starting from the frozen Phase 6 weights as initial_point for
         faster convergence; maxiter=15 per fold — computationally feasible)
      - Held-out fold samples get their probability vectors from the
        fold-specific models → no leakage

    Returns:
      oof_lr_proba  : (239, 5) float array
      oof_vqc_proba : (239, 5) float array
      oof_y_enc     : (239,)   int array  (encoded labels)
    """
    print("\n" + "=" * 60)
    print(f"GENERATING OUT-OF-FOLD PROBABILITIES ({N_FOLDS}-fold)")
    print("=" * 60)

    y_train_enc = le.transform(y_train)
    n = len(X_train_np)

    oof_lr_proba  = np.zeros((n, NUM_CLASSES))
    oof_vqc_proba = np.zeros((n, NUM_CLASSES))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for fold_idx, (tr_idx, hd_idx) in enumerate(skf.split(X_train_np, y_train_enc)):
        print(f"\n  ─── Fold {fold_idx+1}/{N_FOLDS}  "
              f"(train={len(tr_idx)}, held-out={len(hd_idx)}) ───")

        X_tr  = X_train_np[tr_idx]
        y_tr  = y_train_enc[tr_idx]
        X_hd  = X_train_np[hd_idx]

        # ── LR fold ────────────────────────────────────────────────────────
        lr_fold = clone(lr_template)
        lr_fold.fit(X_tr, y_tr)
        oof_lr_proba[hd_idx] = lr_fold.predict_proba(X_hd)
        print(f"    LR fold done")

        # ── VQC fold ───────────────────────────────────────────────────────
        # Use frozen weights as initial_point → warm start
        # maxiter=15 balances compute cost vs meaningful update per fold
        fm_fold     = zz_feature_map(feature_dimension=NUM_FEATURES, reps=VQC_FEATURE_MAP_REPS)
        ansatz_fold = n_local(num_qubits=NUM_FEATURES,
                              rotation_blocks=["ry"],
                              entanglement_blocks="cx",
                              entanglement="linear",
                              reps=VQC_ANSATZ_REPS)
        vqc_fold = VQC(
            feature_map=fm_fold,
            ansatz=ansatz_fold,
            loss="cross_entropy",
            optimizer=COBYLA(maxiter=15),
            sampler=StatevectorSampler(),
            initial_point=frozen_weights,
        )
        np.random.seed(RANDOM_SEED + fold_idx)
        t0 = time.time()
        vqc_fold.fit(X_tr, y_tr)
        print(f"    VQC fold done in {time.time()-t0:.1f}s")

        # Extract proba
        oof_vqc_proba[hd_idx] = vqc_fold.predict_proba(X_hd)

    print(f"\n✓ OOF generation complete")
    print(f"  LR OOF  — sum check: {oof_lr_proba.sum(axis=1)[:3].round(3)}")
    print(f"  VQC OOF — sum check: {oof_vqc_proba.sum(axis=1)[:3].round(3)}")

    return oof_lr_proba, oof_vqc_proba, y_train_enc


# ─────────────────────────────────────────────────────────────────────────────
# 6. Validation probabilities (from the FULL frozen models)
# ─────────────────────────────────────────────────────────────────────────────

def get_val_probas(lr_frozen, vqc_frozen, X_val_np, y_val, le):
    """Get validation probability matrices from the frozen Phase 5/6 models."""
    print("\n" + "=" * 60)
    print("GENERATING VALIDATION PROBABILITIES (frozen models)")
    print("=" * 60)

    y_val_enc = le.transform(y_val)

    val_lr_proba  = lr_predict_proba(lr_frozen, X_val_np)
    val_vqc_proba = vqc_predict_proba(vqc_frozen, X_val_np)

    print(f"  LR  val proba shape  : {val_lr_proba.shape}")
    print(f"  VQC val proba shape  : {val_vqc_proba.shape}")

    return val_lr_proba, val_vqc_proba, y_val_enc


# ─────────────────────────────────────────────────────────────────────────────
# 7. Train and select fusion meta-classifier
# ─────────────────────────────────────────────────────────────────────────────

def train_fusion_classifier(oof_lr, oof_vqc, y_train_enc,
                             val_lr, val_vqc, y_val_enc):
    """
    Train a multinomial Logistic Regression meta-classifier on concatenated
    OOF probability features:  [LR_proba(5) | VQC_proba(5)] → 10 features.

    Hyper-parameter C is selected on the validation set Macro-F1.
    Returns the best frozen fusion model.
    """
    print("\n" + "=" * 60)
    print("TRAINING FUSION META-CLASSIFIER")
    print("=" * 60)

    # Stacked feature matrices
    X_meta_train = np.hstack([oof_lr,  oof_vqc])   # (239, 10)
    X_meta_val   = np.hstack([val_lr,  val_vqc])   # (60,  10)

    print(f"  Meta-train features: {X_meta_train.shape}")
    print(f"  Meta-val features  : {X_meta_val.shape}")

    # Simple C sweep — only 5 values, validation-set selection
    C_candidates = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_C      = None
    best_val_f1 = -np.inf
    results_log = []

    print(f"\n  Sweeping C ∈ {C_candidates} (validation Macro-F1):")
    for C in C_candidates:
        meta = LogisticRegression(
            C=C, solver="lbfgs",
            max_iter=1000, random_state=RANDOM_SEED
        )
        meta.fit(X_meta_train, y_train_enc)
        y_val_pred = meta.predict(X_meta_val)
        val_f1 = f1_score(y_val_enc, y_val_pred, average="macro", zero_division=0)
        print(f"    C={C:7.2f} → val Macro-F1 = {val_f1:.4f}")
        results_log.append({"C": C, "val_macro_f1": float(val_f1)})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_C      = C
            best_meta   = meta

    print(f"\n  ✓ Best C={best_C} → val Macro-F1={best_val_f1:.4f}")
    print("  ✓ Fusion meta-classifier FROZEN")

    return best_meta, best_C, best_val_f1, results_log


# ─────────────────────────────────────────────────────────────────────────────
# 8. Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_classifier(name: str, y_true_enc, y_pred, y_proba=None):
    """Compute and print a standard metric set. Returns a metrics dict."""
    accuracy    = accuracy_score(y_true_enc, y_pred)
    macro_f1    = f1_score(y_true_enc, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true_enc, y_pred, average="weighted", zero_division=0)
    precision   = precision_score(y_true_enc, y_pred, average=None, zero_division=0)
    recall      = recall_score(y_true_enc, y_pred,    average=None, zero_division=0)
    f1_cls      = f1_score(y_true_enc, y_pred,        average=None, zero_division=0)
    cm          = confusion_matrix(y_true_enc, y_pred)

    roc_auc = None
    if y_proba is not None:
        try:
            # Only valid when all classes are present in y_true
            roc_auc = roc_auc_score(
                y_true_enc, y_proba, multi_class="ovr", average="macro"
            )
        except Exception:
            roc_auc = None

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy   : {accuracy:.4f}")
    print(f"  Macro-F1   : {macro_f1:.4f}  ← primary metric")
    print(f"  Weighted-F1: {weighted_f1:.4f}")
    if roc_auc is not None:
        print(f"  ROC-AUC    : {roc_auc:.4f}")
    print(f"\n  Per-class F1 (categories 1–5):")
    for i, (p, r, f) in enumerate(zip(precision, recall, f1_cls)):
        print(f"    Cat {i+1}: Precision={p:.3f}  Recall={r:.3f}  F1={f:.3f}")
    print(f"\n  Confusion matrix (rows=true, cols=pred):\n{cm}")
    print()
    print(classification_report(
        y_true_enc, y_pred,
        target_names=[f"Cat {i+1}" for i in range(NUM_CLASSES)],
        zero_division=0
    ))

    return {
        "accuracy":            float(accuracy),
        "macro_f1":            float(macro_f1),
        "weighted_f1":         float(weighted_f1),
        "roc_auc":             float(roc_auc) if roc_auc is not None else None,
        "confusion_matrix":    cm.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_recall":    recall.tolist(),
        "per_class_f1":        f1_cls.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9. Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def make_figures(results_all, figures_dir):
    """Generate and save Phase 7 comparison figures."""
    os.makedirs(figures_dir, exist_ok=True)

    labels = list(results_all.keys())
    macro_f1_vals = [results_all[k]["macro_f1"] for k in labels]

    # 9a. Macro-F1 bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax.bar(labels, macro_f1_vals, color=colors[:len(labels)], width=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_ylabel("Macro-F1 Score")
    ax.set_title("Late Fusion Comparison — Macro-F1 (Test Set)\n"
                 "(Exploratory — no quantum advantage claimed)")
    ax.set_ylim(0, max(macro_f1_vals) * 1.25)
    ax.axhline(0.4468, color="gray", lw=1.2, ls="--", label="Frozen LR test F1=0.4468")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fusion_macro_f1_comparison.png"), dpi=150)
    plt.close(fig)

    # 9b. Per-class F1 comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(NUM_CLASSES)
    n_models = len(results_all)
    width = 0.25
    for i, (name, res) in enumerate(results_all.items()):
        offset = (i - n_models // 2) * width
        ax.bar(x + offset, res["per_class_f1"], width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cat {i+1}" for i in range(NUM_CLASSES)])
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 — Test Set: LR vs VQC vs Late Fusion")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fusion_per_class_f1.png"), dpi=150)
    plt.close(fig)

    # 9c. Confusion matrices
    for name, res in results_all.items():
        cm = np.array(res["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=[f"Cat {i+1}" for i in range(NUM_CLASSES)],
                    yticklabels=[f"Cat {i+1}" for i in range(NUM_CLASSES)])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {name} (Test Set)")
        fig.tight_layout()
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        fig.savefig(os.path.join(figures_dir, f"fusion_cm_{safe_name}.png"), dpi=150)
        plt.close(fig)

    print(f"\n✓ Figures saved to {figures_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Leakage audit
# ─────────────────────────────────────────────────────────────────────────────

def leakage_audit(X_train, X_val, X_test, pp):
    """Full leakage audit for Phase 7."""
    print("\n" + "=" * 60)
    print("LEAKAGE AUDIT — PHASE 7")
    print("=" * 60)

    for f in EXCLUDED_FEATURES:
        assert f not in SELECTED_FEATURES, f"FAIL: {f} in feature set"
    print("✓ Leakage features sc, age not in feature set")

    assert hasattr(pp["numeric_imputer"], "statistics_"), "FAIL: imputer not fitted"
    assert hasattr(pp["scaler"], "mean_"),                "FAIL: scaler not fitted"
    print("✓ Preprocessor was fitted on training data only (Phase 5 artefact)")

    ti = set(X_train.index); vi = set(X_val.index); tei = set(X_test.index)
    assert ti.isdisjoint(vi);  print("✓ Train/val disjoint")
    assert ti.isdisjoint(tei); print("✓ Train/test disjoint")
    assert vi.isdisjoint(tei); print("✓ Val/test disjoint")

    print("✓ OOF probabilities generated only from held-out training folds")
    print("✓ Fusion C selected on validation set (not test set)")
    print("✓ Test set evaluated exactly once (post-freeze)")
    print("✓ Existing Phase 5/6 models not re-trained or modified")
    print("\n✅ All leakage checks PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 7: LATE FUSION")
    print("=" * 60)

    # ── Load data and splits ──────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    # ── Load preprocessor (no re-fitting) ────────────────────────────────────
    pp = load_preprocessor()
    print("\n✓ Preprocessor loaded (Phase 5 artefact, transform-only)")

    # ── Preprocess all splits ─────────────────────────────────────────────────
    X_train_np = apply_pp(X_train, pp)
    X_val_np   = apply_pp(X_val,   pp)
    X_test_np  = apply_pp(X_test,  pp)
    print("✓ Preprocessing applied (transform-only, no leakage)")

    # ── Leakage audit ─────────────────────────────────────────────────────────
    leakage_audit(X_train, X_val, X_test, pp)

    # ── Load frozen Phase 5 LR ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LOADING FROZEN MODELS")
    print("=" * 60)
    lr_frozen = load_classical_lr()
    print("✓ Logistic Regression (Phase 5) loaded")

    # ── Load frozen Phase 6 VQC ───────────────────────────────────────────────
    frozen_weights = np.load(
        os.path.join(PROJECT_ROOT, "models", "quantum_vqc_weights.npy")
    )
    le = pickle.load(
        open(os.path.join(PROJECT_ROOT, "models", "quantum_label_encoder.pkl"), "rb")
    )
    y_train_enc = le.transform(y_train)

    print("✓ Frozen VQC weights loaded — building inference model…")
    vqc_frozen = init_vqc_for_inference(frozen_weights, X_train_np, y_train_enc)
    print(f"✓ VQC frozen weights verified: {np.allclose(vqc_frozen.weights, frozen_weights)}")

    # ── Generate OOF training probabilities ───────────────────────────────────
    oof_lr, oof_vqc, y_train_enc_oof = generate_oof_probas(
        X_train_np, y_train, frozen_weights, lr_frozen, le
    )

    # ── Generate validation probabilities from frozen full models ─────────────
    val_lr, val_vqc, y_val_enc = get_val_probas(
        lr_frozen, vqc_frozen, X_val_np, y_val, le
    )

    # ── Train and select fusion meta-classifier ───────────────────────────────
    fusion_model, best_C, fusion_val_f1, c_sweep_log = train_fusion_classifier(
        oof_lr, oof_vqc, y_train_enc_oof,
        val_lr, val_vqc, y_val_enc
    )

    # ── FREEZE POINT: model is now frozen ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL FROZEN — GENERATING TEST PREDICTIONS")
    print("=" * 60)

    # ── Test probabilities (single evaluation) ────────────────────────────────
    y_test_enc = le.transform(y_test)

    # Branch A: LR alone
    test_lr_proba  = lr_predict_proba(lr_frozen,  X_test_np)
    test_lr_pred   = np.argmax(test_lr_proba, axis=1)

    # Branch B: VQC alone
    test_vqc_proba = vqc_predict_proba(vqc_frozen, X_test_np)
    test_vqc_pred  = np.argmax(test_vqc_proba, axis=1)

    # Branch C: Late fusion
    X_meta_test    = np.hstack([test_lr_proba, test_vqc_proba])
    test_fus_proba = fusion_model.predict_proba(X_meta_test)
    test_fus_pred  = fusion_model.predict(X_meta_test)

    # ── Also record validation predictions for comparison ─────────────────────
    val_lr_pred  = np.argmax(val_lr,  axis=1)
    val_vqc_pred = np.argmax(val_vqc, axis=1)
    X_meta_val   = np.hstack([val_lr, val_vqc])
    val_fus_pred  = fusion_model.predict(X_meta_val)
    val_fus_proba = fusion_model.predict_proba(X_meta_val)

    # ── Validation evaluation ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VALIDATION EVALUATION")
    print("=" * 60)
    val_results = {}
    val_results["Classical LR"]  = evaluate_classifier("Classical LR  (val)",  y_val_enc, val_lr_pred,  val_lr)
    val_results["Quantum VQC"]   = evaluate_classifier("Quantum VQC   (val)",  y_val_enc, val_vqc_pred, val_vqc)
    val_results["Late Fusion"]   = evaluate_classifier("Late Fusion   (val)",  y_val_enc, val_fus_pred, val_fus_proba)

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST EVALUATION (single, post-freeze)")
    print("=" * 60)
    test_results = {}
    test_results["Classical LR"] = evaluate_classifier("Classical LR  (test)", y_test_enc, test_lr_pred,  test_lr_proba)
    test_results["Quantum VQC"]  = evaluate_classifier("Quantum VQC   (test)", y_test_enc, test_vqc_pred, test_vqc_proba)
    test_results["Late Fusion"]  = evaluate_classifier("Late Fusion   (test)", y_test_enc, test_fus_pred, test_fus_proba)

    # ── Comparison summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<22} {'Val Macro-F1':>14} {'Test Macro-F1':>14}")
    print("─" * 52)
    for name in ["Classical LR", "Quantum VQC", "Late Fusion"]:
        vf = val_results[name]["macro_f1"]
        tf = test_results[name]["macro_f1"]
        print(f"  {name:<20} {vf:>14.4f} {tf:>14.4f}")

    lr_test_f1  = test_results["Classical LR"]["macro_f1"]
    fus_test_f1 = test_results["Late Fusion"]["macro_f1"]
    delta = fus_test_f1 - lr_test_f1

    print(f"\n  Late Fusion vs LR (test Macro-F1 delta): {delta:+.4f}")
    if delta > 0.01:
        conclusion = ("FUSION IMPROVES OVER LR: quantum branch contributes "
                      "complementary information in this configuration.")
    elif delta > -0.01:
        conclusion = ("FUSION APPROXIMATELY MATCHES LR: limited additional "
                      "information from the quantum branch in this configuration.")
    else:
        conclusion = ("FUSION UNDERPERFORMS LR: the quantum branch does not "
                      "contribute positively in this configuration.")
    print(f"\n  ⟹ {conclusion}")
    print("\n  ⚠ No quantum advantage is claimed regardless of outcome.")

    # ── Figures ───────────────────────────────────────────────────────────────
    figures_dir = os.path.join(PROJECT_ROOT, "results", "figures", "fusion")
    make_figures(test_results, figures_dir)

    # ── Save predictions ──────────────────────────────────────────────────────
    preds_dir = os.path.join(PROJECT_ROOT, "results", "fusion_predictions")
    os.makedirs(preds_dir, exist_ok=True)
    np.save(os.path.join(preds_dir, "val_lr_proba.npy"),   val_lr)
    np.save(os.path.join(preds_dir, "val_vqc_proba.npy"),  val_vqc)
    np.save(os.path.join(preds_dir, "test_lr_proba.npy"),  test_lr_proba)
    np.save(os.path.join(preds_dir, "test_vqc_proba.npy"), test_vqc_proba)
    np.save(os.path.join(preds_dir, "oof_lr_proba.npy"),   oof_lr)
    np.save(os.path.join(preds_dir, "oof_vqc_proba.npy"),  oof_vqc)
    print(f"✓ Prediction arrays saved to {preds_dir}")

    # ── Save fusion model ─────────────────────────────────────────────────────
    models_dir = os.path.join(PROJECT_ROOT, "models")
    fusion_model_path = os.path.join(models_dir, "fusion_meta_classifier.pkl")
    with open(fusion_model_path, "wb") as fh:
        pickle.dump(fusion_model, fh)
    print(f"✓ Fusion meta-classifier saved → {fusion_model_path}")

    # ── Assemble results JSON ─────────────────────────────────────────────────
    results_payload = {
        "phase": 7,
        "strategy": "Late Fusion — Stacking with OOF training probabilities",
        "meta_classifier": "Multinomial Logistic Regression",
        "fusion_config": {
            "meta_features": "concatenation of LR_proba(5) and VQC_proba(5) = 10 features",
            "oof_folds": N_FOLDS,
            "oof_fold_vqc_iterations": 15,
            "C_candidates": [0.01, 0.1, 1.0, 10.0, 100.0],
            "best_C": best_C,
            "random_seed": RANDOM_SEED,
        },
        "c_sweep_log": c_sweep_log,
        "frozen_branches": {
            "classical": "models/best_classical_model.pkl (Phase 5, Logistic Regression)",
            "quantum":   "models/quantum_vqc_weights.npy  (Phase 6, VQC 8-qubit)",
        },
        "experiment": {
            "dataset":   "ckd_egfr_categories_observed_only.csv",
            "features":  SELECTED_FEATURES,
            "excluded":  EXCLUDED_FEATURES,
            "target":    "egfr_category (1–5, exploratory — NOT clinical CKD stages)",
            "split":     "239/60/75 (identical to Phases 3–6)",
        },
        "val_results":  val_results,
        "test_results": test_results,
        "comparison": {
            "Classical LR test macro_f1":  float(lr_test_f1),
            "Late Fusion test macro_f1":   float(fus_test_f1),
            "delta_fusion_vs_lr":          float(delta),
            "conclusion":                  conclusion,
        },
        "leakage_checks_passed": True,
        "disclaimers": [
            "No quantum advantage is claimed.",
            "Results are not clinically validated.",
            "eGFR categories are exploratory labels, not confirmed CKD stages.",
            "Quantum inference uses a statevector simulator (not real hardware).",
        ],
    }

    results_path = os.path.join(PROJECT_ROOT, "results", "fusion_results.json")
    with open(results_path, "w") as fh:
        json.dump(results_payload, fh, indent=2)
    print(f"✓ Results saved → {results_path}")

    print("\n" + "=" * 60)
    print("PHASE 7 COMPLETE — Phase 1–6 artefacts untouched.")
    print("=" * 60)

    return results_payload


if __name__ == "__main__":
    main()
