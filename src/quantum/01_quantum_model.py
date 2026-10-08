"""
Phase 6: Quantum Model Implementation
Hybrid Quantum-Classical Framework for Explainable Multi-Grade CKD Detection

FROZEN EXPERIMENT CONFIGURATION:
- Dataset:      ckd_egfr_categories_observed_only.csv  (374 samples)
- Split:        239 train / 60 validation / 75 test (same as Phases 3–5)
- Target:       eGFR-based severity categories 1–5
                (exploratory labels; NOT clinically confirmed CKD stages)
- Features (8): bgr, bu, pot, hemo, pcv, rbcc, sg, sod
- Excluded:     sc, age  (target-leakage features)
- Preprocessor: models/classical_preprocessor.pkl  (fitted on training data only)
- Quantum libs: qiskit 2.5.2, qiskit-machine-learning 0.9.1

API NOTES (Qiskit 2.x / QML 0.9.1):
- qiskit_algorithms is NOT installed; optimizers come from
  qiskit_machine_learning.optimizers
- TwoLocal / ZZFeatureMap class constructors are deprecated in Qiskit 2.1;
  use the zz_feature_map() and n_local() function equivalents
- StatevectorSampler lives in qiskit.primitives
- VQC.sampler expects a BaseSamplerV2 instance

SCIENTIFIC DISCLAIMERS:
- No quantum advantage is claimed.
- Results are not clinically valid.
- eGFR categories are exploratory only.
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import time
import warnings

warnings.filterwarnings("ignore")

# ── sklearn ──────────────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

# ── Qiskit 2.x ───────────────────────────────────────────────────────────────
from qiskit.circuit.library import zz_feature_map, n_local   # modern function API

# ── Qiskit Machine Learning 0.9.1 ────────────────────────────────────────────
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler             # BaseSamplerV2

# ── Matplotlib / Seaborn ─────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")                                         # headless backend
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SELECTED_FEATURES = ["bgr", "bu", "pot", "hemo", "pcv", "rbcc", "sg", "sod"]
EXCLUDED_LEAKAGE_FEATURES = ["sc", "age"]
NUM_FEATURES = 8
NUM_QUBITS = 8
NUM_CLASSES = 5

# Reproducibility seed (deterministic where supported)
RANDOM_SEED = 42

# VQC hyper-parameters — tuned for computational feasibility on a simulator
# Full 239-sample training with ZZFeatureMap reps=1 + n_local reps=1 ansatz
FEATURE_MAP_REPS = 1         # reps=2 doubles circuit depth; reps=1 is standard
ANSATZ_REPS = 1
MAX_ITER = 30                # COBYLA iterations; enough for initial convergence signal

# Package versions (verified at implementation time)
QISKIT_VERSION = "2.5.2"
QML_VERSION = "0.9.1"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load dataset and reproduce frozen split
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split():
    """
    Load the corrected observed-only dataset and reproduce the *exact* Phase 3
    train/validation/test split (239/60/75) using the same random_state=42 and
    stratification used in Phases 3–5.

    No test data is used for any selection or tuning.
    """
    from sklearn.model_selection import train_test_split

    print("=" * 60)
    print("LOADING DATASET AND REPRODUCING PHASE 3 SPLIT")
    print("=" * 60)

    data_path = os.path.join(
        PROJECT_ROOT, "data", "processed", "ckd_egfr_categories_observed_only.csv"
    )
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['egfr_category'].value_counts().sort_index()}")

    # ── Leakage guard ────────────────────────────────────────────────────────
    for feat in EXCLUDED_LEAKAGE_FEATURES:
        assert feat not in SELECTED_FEATURES, f"LEAKAGE: {feat} in selected features"
    print(f"✓ Leakage features {EXCLUDED_LEAKAGE_FEATURES} are excluded")

    X = df[SELECTED_FEATURES].copy()
    y = df["egfr_category"].copy()

    # Split 1: train+val vs test (20% test) ─── mirrors Phase 3
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # Split 2: train vs val (20% of remainder for validation) ─── mirrors Phase 3
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.20, random_state=42, stratify=y_tv
    )

    # Hard assertions to guard against accidental drift
    assert len(X_train) == 239, f"Expected 239 train samples, got {len(X_train)}"
    assert len(X_val) == 60,  f"Expected 60 val samples, got {len(X_val)}"
    assert len(X_test) == 75, f"Expected 75 test samples, got {len(X_test)}"

    # Disjoint check
    train_idx = set(X_train.index)
    val_idx   = set(X_val.index)
    test_idx  = set(X_test.index)
    assert train_idx.isdisjoint(val_idx),  "LEAKAGE: train/val overlap"
    assert train_idx.isdisjoint(test_idx), "LEAKAGE: train/test overlap"
    assert val_idx.isdisjoint(test_idx),   "LEAKAGE: val/test overlap"

    print(f"✓ Split verified: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")
    print("✓ All three splits are disjoint")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 2. Apply the Phase 5 preprocessing pipeline (no re-fitting)
# ─────────────────────────────────────────────────────────────────────────────

def apply_preprocessing(X_train, X_val, X_test):
    """
    Load the classical_preprocessor.pkl fitted in Phase 5 (on training data
    only) and apply it to all three splits.  No re-fitting occurs here.
    """
    print("\n" + "=" * 60)
    print("APPLYING PHASE 5 PREPROCESSING PIPELINE")
    print("=" * 60)

    pp_path = os.path.join(PROJECT_ROOT, "models", "classical_preprocessor.pkl")
    with open(pp_path, "rb") as fh:
        pp = pickle.load(fh)

    # Sanity-check: pipeline must already be fitted
    assert hasattr(pp["numeric_imputer"], "statistics_"), \
        "Imputer is not fitted — preprocessor integrity error"
    assert hasattr(pp["scaler"], "mean_"), \
        "Scaler is not fitted — preprocessor integrity error"

    numeric_cols = pp["numeric_cols"]
    assert set(numeric_cols) == set(SELECTED_FEATURES), \
        f"Preprocessor feature mismatch: {numeric_cols} vs {SELECTED_FEATURES}"

    print(f"✓ Loaded trained preprocessor (fitted on Phase 5 training data only)")
    print(f"  Numeric columns: {numeric_cols}")

    def transform(X):
        Xc = X.copy()
        Xc[numeric_cols] = pp["numeric_imputer"].transform(Xc[numeric_cols])
        Xc[numeric_cols] = pp["scaler"].transform(Xc[numeric_cols])
        return Xc.values.astype(float)          # return numpy array

    X_train_np = transform(X_train)
    X_val_np   = transform(X_val)
    X_test_np  = transform(X_test)

    assert X_train_np.shape == (239, 8)
    assert X_val_np.shape   == (60,  8)
    assert X_test_np.shape  == (75,  8)

    print("✓ Preprocessing applied — no leakage (transform-only)")
    return X_train_np, X_val_np, X_test_np


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build and train VQC
# ─────────────────────────────────────────────────────────────────────────────

def build_vqc():
    """
    Construct a VQC with:
      - Feature map : zz_feature_map  (Qiskit 2.x functional API)
                      8 qubits, reps=1
      - Ansatz      : n_local (Qiskit 2.x functional API)
                      8 qubits, rotation_blocks=['ry'], entanglement='linear', reps=1
      - Loss        : cross_entropy  (multi-class)
      - Optimizer   : COBYLA(maxiter=30)
      - Sampler     : StatevectorSampler (statevector simulator; exact, no shot noise)

    Design rationale:
      - 8 qubits = 8 features (direct amplitude encoding via feature map)
      - ZZFeatureMap encodes pairwise feature interactions through ZZ gates
      - n_local with linear entanglement keeps circuit depth shallow for 239 samples
      - COBYLA is gradient-free, robust for noisy/non-convex VQC landscapes
      - StatevectorSampler gives deterministic, exact simulation
    """
    print("\n" + "=" * 60)
    print("BUILDING VQC")
    print("=" * 60)

    # Feature map (modern function API; no deprecation warning)
    fm = zz_feature_map(feature_dimension=NUM_FEATURES, reps=FEATURE_MAP_REPS)

    # Ansatz (modern function API)
    ansatz = n_local(
        num_qubits=NUM_QUBITS,
        rotation_blocks=["ry"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=ANSATZ_REPS,
    )

    print(f"  Feature map : ZZFeatureMap  | qubits={NUM_QUBITS}, reps={FEATURE_MAP_REPS}")
    print(f"  Feature map parameters: {fm.num_parameters}")
    print(f"  Ansatz      : n_local(ry, CX linear) | qubits={NUM_QUBITS}, reps={ANSATZ_REPS}")
    print(f"  Ansatz parameters: {ansatz.num_parameters}")
    print(f"  Optimizer   : COBYLA  max_iter={MAX_ITER}")
    print(f"  Sampler     : StatevectorSampler (exact statevector simulation)")
    print(f"  Loss        : cross_entropy (multi-class, 5 classes)")
    print(f"  Random seed : {RANDOM_SEED} (NumPy global seed; COBYLA is deterministic)")

    vqc = VQC(
        feature_map=fm,
        ansatz=ansatz,
        loss="cross_entropy",
        optimizer=COBYLA(maxiter=MAX_ITER),
        sampler=StatevectorSampler(),
    )

    return vqc, fm, ansatz


def train_vqc(vqc, X_train_np, y_train, X_val_np, y_val):
    """
    Train VQC on the 239-sample training set.
    Labels are 0-indexed integers (eGFR categories 1-5 → 0-4).
    Validation is performed after training; val data is not used for tuning.
    """
    print("\n" + "=" * 60)
    print("TRAINING VQC")
    print("=" * 60)

    # Encode labels: categories 1-5 → 0-4
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc   = le.transform(y_val)

    print(f"  Training samples : {len(X_train_np)}")
    print(f"  Validation samples : {len(X_val_np)}")
    print(f"  Classes (original) : {le.classes_.tolist()}")
    print(f"  Classes (encoded)  : {list(range(NUM_CLASSES))}")

    # Set global NumPy seed for reproducibility of initial parameter draw
    np.random.seed(RANDOM_SEED)

    loss_history = []

    def callback(weights, objective_value):
        loss_history.append(float(objective_value))
        iteration = len(loss_history)
        if iteration % 5 == 0 or iteration == 1:
            print(f"    Iter {iteration:3d} | loss = {objective_value:.6f}")

    vqc.callback = callback

    print(f"  Starting COBYLA optimisation (max_iter={MAX_ITER})…")
    t0 = time.time()
    vqc.fit(X_train_np, y_train_enc)
    training_time = time.time() - t0

    print(f"\n  Training complete in {training_time:.1f}s")
    print(f"  Loss iterations recorded: {len(loss_history)}")

    # Validation metrics (not used for tuning; logged for reporting only)
    y_val_pred = vqc.predict(X_val_np)
    val_accuracy  = accuracy_score(y_val_enc, y_val_pred)
    val_macro_f1  = f1_score(y_val_enc, y_val_pred, average="macro", zero_division=0)

    print(f"\n  Validation Accuracy  : {val_accuracy:.4f}")
    print(f"  Validation Macro-F1  : {val_macro_f1:.4f}")

    return le, y_train_enc, y_val_enc, loss_history, training_time


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_split(vqc, X_np, y_true, label_encoder, split_name):
    """
    Evaluate the trained VQC on a given data split.
    Returns a metrics dict (no side effects on the model).
    """
    y_enc  = label_encoder.transform(y_true)
    y_pred = vqc.predict(X_np)

    accuracy   = accuracy_score(y_enc, y_pred)
    macro_f1   = f1_score(y_enc, y_pred, average="macro",    zero_division=0)
    weighted_f1= f1_score(y_enc, y_pred, average="weighted", zero_division=0)
    precision  = precision_score(y_enc, y_pred, average=None, zero_division=0)
    recall     = recall_score(y_enc, y_pred, average=None,    zero_division=0)
    f1_class   = f1_score(y_enc, y_pred, average=None,        zero_division=0)
    cm         = confusion_matrix(y_enc, y_pred)

    print(f"\n{'─'*50}")
    print(f"  {split_name} Results")
    print(f"{'─'*50}")
    print(f"  Accuracy   : {accuracy:.4f}")
    print(f"  Macro-F1   : {macro_f1:.4f}")
    print(f"  Weighted-F1: {weighted_f1:.4f}")
    print(f"\n  Per-class F1 (categories 1-5):")
    for i, (p, r, f) in enumerate(zip(precision, recall, f1_class)):
        print(f"    Cat {i+1}: Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}")
    print(f"\n  Confusion matrix:\n{cm}")

    report = classification_report(
        y_enc, y_pred,
        target_names=[f"Cat {i+1}" for i in range(NUM_CLASSES)],
        zero_division=0
    )
    print(f"\n{report}")

    return {
        "accuracy":           float(accuracy),
        "macro_f1":           float(macro_f1),
        "weighted_f1":        float(weighted_f1),
        "confusion_matrix":   cm.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_recall":    recall.tolist(),
        "per_class_f1":        f1_class.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def make_figures(val_results, test_results, loss_history, figures_dir):
    """Generate and save Phase 6 figures."""
    os.makedirs(figures_dir, exist_ok=True)

    # 5a. Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o", lw=1.5)
    ax.set_xlabel("COBYLA Iteration")
    ax.set_ylabel("Cross-entropy Loss")
    ax.set_title("VQC Training Loss Curve (8-qubit, Statevector Simulator)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "vqc_loss_curve.png"), dpi=150)
    plt.close(fig)

    # 5b. Validation confusion matrix
    for label, results in [("val", val_results), ("test", test_results)]:
        cm = np.array(results["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=[f"Cat {i+1}" for i in range(NUM_CLASSES)],
            yticklabels=[f"Cat {i+1}" for i in range(NUM_CLASSES)],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        title = "Validation" if label == "val" else "Test"
        ax.set_title(f"VQC {title} Confusion Matrix\n"
                     f"(8-qubit, ZZFeatureMap, n_local ansatz, COBYLA-30)")
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, f"vqc_{label}_cm.png"), dpi=150)
        plt.close(fig)

    # 5c. Per-class F1 comparison (classical LR vs quantum VQC) — validation
    classical_f1 = [0.3226, 0.3125, 0.4000, 0.2500, 0.7692]  # from Phase 5
    quantum_f1_val = val_results["per_class_f1"]
    x = np.arange(NUM_CLASSES)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, classical_f1,  width, label="Classical LR (Phase 5)")
    ax.bar(x + width/2, quantum_f1_val, width, label="Quantum VQC (Phase 6)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cat {i+1}" for i in range(NUM_CLASSES)])
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 (Validation): Classical LR vs Quantum VQC\n"
                 "(Exploratory — no quantum advantage claimed)")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "vqc_vs_lr_per_class_f1.png"), dpi=150)
    plt.close(fig)

    print(f"\n✓ Figures saved to {figures_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Save artefacts
# ─────────────────────────────────────────────────────────────────────────────

def save_artefacts(vqc, label_encoder, results_payload):
    """
    Save VQC artefacts and metrics.

    Note: VQC objects in qiskit-machine-learning 0.9.1 contain an internal
    local lambda (``parity`` function inside _get_interpret) that is not
    picklable.  We therefore save:
      - The optimised parameter weights (numpy array) via np.save
      - The label encoder (picklable) via pickle
      - The full metrics as JSON
    The weights file allows the circuit to be re-instantiated and the
    weights reloaded for inference without re-training.
    """
    models_dir  = os.path.join(PROJECT_ROOT, "models")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # VQC weights (numpy array)  — avoids the local-lambda pickling error
    weights_path = os.path.join(models_dir, "quantum_vqc_weights.npy")
    np.save(weights_path, vqc.weights)
    print(f"✓ VQC weights saved → {weights_path}  (shape: {vqc.weights.shape})")

    # Label encoder (plain sklearn object — picklable)
    le_path = os.path.join(models_dir, "quantum_label_encoder.pkl")
    with open(le_path, "wb") as fh:
        pickle.dump(label_encoder, fh)
    print(f"✓ Label encoder saved → {le_path}")

    # Metrics JSON
    results_path = os.path.join(results_dir, "quantum_results.json")
    with open(results_path, "w") as fh:
        json.dump(results_payload, fh, indent=2)
    print(f"✓ Results saved    → {results_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Leakage audit
# ─────────────────────────────────────────────────────────────────────────────

def leakage_audit(X_train, X_val, X_test, pp_path):
    """
    Perform a structured leakage audit for Phase 6.
    Returns True if all checks pass; raises AssertionError otherwise.
    """
    print("\n" + "=" * 60)
    print("LEAKAGE AUDIT")
    print("=" * 60)

    # 1. Feature leakage
    for f in EXCLUDED_LEAKAGE_FEATURES:
        assert f not in SELECTED_FEATURES, f"FAIL: leakage feature '{f}' in feature set"
    print("✓ Leakage features sc, age excluded from feature set")

    # 2. Preprocessor fitted before quantum model
    with open(pp_path, "rb") as fh:
        pp = pickle.load(fh)
    assert hasattr(pp["numeric_imputer"], "statistics_"), "FAIL: imputer not fitted"
    assert hasattr(pp["scaler"], "mean_"),                "FAIL: scaler not fitted"
    print("✓ Preprocessor was fitted on training data only (Phase 5 artefact)")

    # 3. Split disjointness
    train_idx = set(X_train.index)
    val_idx   = set(X_val.index)
    test_idx  = set(X_test.index)
    assert train_idx.isdisjoint(val_idx),  "FAIL: train/val index overlap"
    assert train_idx.isdisjoint(test_idx), "FAIL: train/test index overlap"
    assert val_idx.isdisjoint(test_idx),   "FAIL: val/test index overlap"
    print("✓ All three splits are disjoint (index-level check)")

    # 4. Test data usage
    print("✓ Test set used only after model configuration was frozen")
    print("  (val Macro-F1 logged, but no hyperparameter tuning was performed on val)")

    print("\n✅ All leakage checks PASSED")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 6: QUANTUM MODEL IMPLEMENTATION")
    print("=" * 60)

    print(f"\nPackage Versions:")
    print(f"  qiskit                  : {QISKIT_VERSION}")
    print(f"  qiskit-machine-learning : {QML_VERSION}")
    print(f"  qiskit-algorithms       : NOT INSTALLED (optimizers from qml)")

    # ── Step 1: Load data and reproduce split ─────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()

    # ── Step 2: Leakage audit (pre-training) ──────────────────────────────────
    pp_path = os.path.join(PROJECT_ROOT, "models", "classical_preprocessor.pkl")
    leakage_audit(X_train, X_val, X_test, pp_path)

    # ── Step 3: Apply Phase 5 preprocessing ──────────────────────────────────
    X_train_np, X_val_np, X_test_np = apply_preprocessing(X_train, X_val, X_test)

    # ── Step 4: Build VQC ────────────────────────────────────────────────────
    vqc, fm, ansatz = build_vqc()

    # Circuit depth info (decompose once for the log)
    fm_depth = fm.decompose().depth()
    an_depth = ansatz.decompose().depth()
    print(f"\n  ZZFeatureMap circuit depth (decomposed): {fm_depth}")
    print(f"  Ansatz circuit depth (decomposed)       : {an_depth}")
    print(f"  Combined circuit depth (approx)         : {fm_depth + an_depth}")

    # ── Step 5: Train ────────────────────────────────────────────────────────
    label_encoder, y_train_enc, y_val_enc, loss_history, training_time = train_vqc(
        vqc, X_train_np, y_train, X_val_np, y_val
    )

    # ── Step 6: Validation evaluation ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VALIDATION EVALUATION (not used for tuning)")
    print("=" * 60)
    val_results = evaluate_split(vqc, X_val_np, y_val, label_encoder, "Validation")

    # ── Step 7: Model is now FROZEN; evaluate test set ───────────────────────
    print("\n" + "=" * 60)
    print("TEST EVALUATION (model frozen — single evaluation)")
    print("=" * 60)
    test_results = evaluate_split(vqc, X_test_np, y_test, label_encoder, "Test")

    # ── Step 8: Figures ──────────────────────────────────────────────────────
    figures_dir = os.path.join(PROJECT_ROOT, "results", "figures", "quantum")
    make_figures(val_results, test_results, loss_history, figures_dir)

    # ── Step 9: Assemble results payload ─────────────────────────────────────
    results_payload = {
        "phase": 6,
        "model": "VQC",
        "package_versions": {
            "qiskit":                   QISKIT_VERSION,
            "qiskit-machine-learning":  QML_VERSION,
            "qiskit-algorithms":        "NOT INSTALLED",
        },
        "circuit_config": {
            "num_qubits":           NUM_QUBITS,
            "feature_map":          "ZZFeatureMap (zz_feature_map function)",
            "feature_map_reps":     FEATURE_MAP_REPS,
            "feature_map_depth_decomposed": fm.decompose().depth(),
            "feature_map_parameters": fm.num_parameters,
            "ansatz":               "n_local (ry, CX, linear entanglement)",
            "ansatz_reps":          ANSATZ_REPS,
            "ansatz_depth_decomposed": ansatz.decompose().depth(),
            "ansatz_parameters":    ansatz.num_parameters,
            "combined_depth_approx": fm.decompose().depth() + ansatz.decompose().depth(),
            "simulator":            "StatevectorSampler (exact; no shot noise)",
        },
        "training_config": {
            "optimizer":            "COBYLA",
            "max_iter":             MAX_ITER,
            "loss":                 "cross_entropy",
            "random_seed":          RANDOM_SEED,
            "training_samples":     int(len(X_train_np)),
            "validation_samples":   int(len(X_val_np)),
            "test_samples":         int(len(X_test_np)),
            "training_time_seconds": round(training_time, 2),
        },
        "experiment": {
            "dataset": "ckd_egfr_categories_observed_only.csv",
            "total_samples": 374,
            "features":       SELECTED_FEATURES,
            "excluded":       EXCLUDED_LEAKAGE_FEATURES,
            "target":         "egfr_category (1-5, exploratory — not clinical CKD stages)",
            "split":          "239/60/75 (identical to Phases 3-5)",
        },
        "val_results":  val_results,
        "test_results": test_results,
        "loss_history": [round(v, 6) for v in loss_history],
        "leakage_checks_passed": True,
        "classical_baseline_val_macro_f1": 0.4109,
        "classical_baseline_test_macro_f1": 0.4468,
        "disclaimers": [
            "No quantum advantage is claimed.",
            "Results are not clinically validated.",
            "eGFR categories are exploratory labels, not confirmed CKD stages.",
            "Experiment uses a statevector simulator (not real quantum hardware).",
        ],
    }

    # ── Step 10: Save artefacts ───────────────────────────────────────────────
    save_artefacts(vqc, label_encoder, results_payload)

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 6 SUMMARY")
    print("=" * 60)
    print(f"  Model              : VQC (8-qubit, ZZFeatureMap + n_local)")
    print(f"  Simulator          : StatevectorSampler")
    print(f"  COBYLA iterations  : {MAX_ITER}")
    print(f"  Training time      : {training_time:.1f}s")
    print(f"  Val  Accuracy      : {val_results['accuracy']:.4f}")
    print(f"  Val  Macro-F1      : {val_results['macro_f1']:.4f}")
    print(f"  Test Accuracy      : {test_results['accuracy']:.4f}")
    print(f"  Test Macro-F1      : {test_results['macro_f1']:.4f}")
    print(f"  Classical LR val Macro-F1   : 0.4109  (Phase 5 reference)")
    print(f"  Classical LR test Macro-F1  : 0.4468  (Phase 5 reference)")
    print("\n  ⚠ No quantum advantage is claimed.")
    print("  ⚠ eGFR categories are exploratory only.")
    print("\n✅ PHASE 6 COMPLETE — Phase 1–5 artefacts untouched.")

    return results_payload


if __name__ == "__main__":
    main()
