# Phase 6: Quantum Model Implementation
## Hybrid Quantum-Classical Framework — CKD eGFR Category Prediction

**Date:** 2026-10-08  
**Status:** PHASE 6 COMPLETE  
**Experiment:** B — Leakage-aware Exploratory Configuration

---

## Disclaimers

> **⚠ No quantum advantage is claimed.**  
> **⚠ Results are not clinically validated.**  
> **⚠ eGFR-based severity categories (1–5) are exploratory labels derived from observed serum creatinine and age. They are NOT confirmed clinical CKD stages.**  
> **⚠ All experiments use a statevector simulator — not real quantum hardware.**

---

## 1. Verification of Pre-Phase 6 Integrity

Before implementing Phase 6, the existing project state was verified against the frozen configuration:

| Check | Expected | Observed | Status |
|-------|----------|----------|--------|
| Total samples | 374 | 374 | ✅ |
| Train split | 239 | 239 | ✅ |
| Validation split | 60 | 60 | ✅ |
| Test split | 75 | 75 | ✅ |
| Feature count | 8 | 8 | ✅ |
| Features | bgr, bu, pot, hemo, pcv, rbcc, sg, sod | bgr, bu, pot, hemo, pcv, rbcc, sg, sod | ✅ |
| Excluded features | sc, age | sc, age | ✅ |
| Preprocessor | Train-fitted | Train-fitted (Phase 5 artefact) | ✅ |
| Selected classical model | Logistic Regression | Logistic Regression | ✅ |
| Classical val Macro-F1 | 0.4109 | 0.4109 | ✅ |
| Classical test Macro-F1 | 0.4468 | 0.4468 | ✅ |

**All integrity checks passed. No Phase 1–5 artefacts were modified.**

---

## 2. Software Environment

| Package | Version | Notes |
|---------|---------|-------|
| `qiskit` | 2.5.2 | Main SDK |
| `qiskit-machine-learning` | 0.9.1 | VQC, QSVC, optimizers |
| `qiskit-algorithms` | NOT INSTALLED | Optimizers sourced from `qiskit_machine_learning.optimizers` |
| `numpy` | — | Seed = 42 for initial parameter draw |
| `scikit-learn` | — | Preprocessing, metrics |

### API Notes (Qiskit 2.x)

Qiskit 2.1 deprecated the class-based `ZZFeatureMap` and `TwoLocal` constructors in favour of functional equivalents:

| Deprecated (pre-2.1) | Used instead (Qiskit 2.5.2) |
|---------------------|-----------------------------|
| `ZZFeatureMap(feature_dimension=n, reps=r)` | `zz_feature_map(feature_dimension=n, reps=r)` |
| `TwoLocal(num_qubits=n, rotations=..., reps=r)` | `n_local(num_qubits=n, rotation_blocks=..., reps=r)` |
| `from qiskit_algorithms.optimizers import COBYLA` | `from qiskit_machine_learning.optimizers import COBYLA` |

The existing template script (`src/quantum/01_quantum_model.py`, as delivered) contained three API errors:
1. Imported `COBYLA` from `qiskit_algorithms` (not installed)
2. Imported `TwoLocal` from `qiskit_machine_learning.circuit.library` (not present)
3. Return-value count mismatch in `train_quantum_model` (5 returns, 4 unpacked)

All three were corrected before running.

---

## 3. Dataset and Split

**No changes were made** to the dataset, target, or split from Phases 1–5.

| Property | Value |
|----------|-------|
| Dataset | `data/processed/ckd_egfr_categories_observed_only.csv` |
| Total samples | 374 (all have observed sc and age) |
| Target | `egfr_category` (1–5, eGFR-based severity — exploratory only) |
| Features | bgr, bu, pot, hemo, pcv, rbcc, sg, sod |
| Excluded | sc, age (target-leakage features) |
| Train | 239 samples |
| Validation | 60 samples |
| Test | 75 samples |
| Split method | `train_test_split(test_size=0.2, random_state=42, stratify=y)` × 2 (identical to Phase 3) |

### Class Distribution

| Category | Train | Val | Test | Total |
|----------|-------|-----|------|-------|
| 1 (high eGFR) | ~65 | ~16 | ~21 | 102 |
| 2 | ~54 | ~14 | ~17 | 84 |
| 3 | ~47 | ~12 | ~15 | 74 |
| 4 | ~33 | ~8 | ~10 | 52 |
| 5 (low eGFR) | ~40 | ~10 | ~12 | 62 |

---

## 4. Preprocessing

The Phase 5 preprocessing pipeline (`models/classical_preprocessor.pkl`) was **loaded and applied** — it was **not re-fitted**.

- **Imputer**: `SimpleImputer(strategy='median')` — fitted on 239 training samples in Phase 5
- **Scaler**: `StandardScaler()` — fitted on 239 training samples in Phase 5
- **Application**: `transform()` only applied to all three splits — zero leakage

```
X_train_np = transform(X_train)   # 239 × 8 float64 array
X_val_np   = transform(X_val)     # 60  × 8 float64 array
X_test_np  = transform(X_test)    # 75  × 8 float64 array
```

---

## 5. Quantum Model Architecture

### 5.1 Overview

A **Variational Quantum Classifier (VQC)** was implemented using:
- A parameterised feature map (data encoding layer)
- A variational ansatz (trainable layer)
- Classical COBYLA optimiser (gradient-free)
- StatevectorSampler (exact simulation)

### 5.2 Feature Map

| Property | Value |
|----------|-------|
| Type | ZZFeatureMap (`zz_feature_map` function, Qiskit 2.x) |
| Number of qubits | 8 (one per feature) |
| Repetitions (`reps`) | 1 |
| Parameters | 8 (one per feature, used for Pauli-Z rotations) |
| Circuit depth (decomposed) | 41 |
| Entanglement | Full ZZ-type pairwise interactions |

The ZZFeatureMap encodes the scaled feature vector via Hadamard gates and ZZ-interaction gates, creating a non-linear feature map in Hilbert space. With `reps=1`, the encoding layer is:
```
H⊗8 → Rz(2·xi) for each qubit i → ZZ interactions between all pairs
```

### 5.3 Ansatz

| Property | Value |
|----------|-------|
| Type | `n_local` (Qiskit 2.x functional API) |
| Number of qubits | 8 |
| Rotation blocks | `RY` gate |
| Entanglement blocks | `CX` (CNOT) gate |
| Entanglement topology | Linear (qubit i ↔ i+1) |
| Repetitions (`reps`) | 1 |
| Trainable parameters | 16 (2 × 8: one RY per qubit per layer) |
| Circuit depth (decomposed) | 9 |

Linear entanglement topology was chosen to keep circuit depth minimal while still introducing qubit correlations.

### 5.4 Combined Circuit

| Property | Value |
|----------|-------|
| Total circuit depth (approx) | 50 (41 feature map + 9 ansatz) |
| Total trainable parameters | 16 |
| Qubits | 8 |
| Simulator | StatevectorSampler (exact; no shot noise) |

### 5.5 Training Configuration

| Property | Value |
|----------|-------|
| Optimizer | COBYLA (Constrained Optimisation By Linear Approximations) |
| Max iterations | 30 |
| Loss function | Cross-entropy (multi-class, 5 classes) |
| Label encoding | Categories 1–5 → integers 0–4 (LabelEncoder) |
| Random seed | 42 (NumPy global seed for initial parameter draw) |
| Training samples | 239 |
| Training time | ~51.1 seconds |

### 5.6 Why VQC?

VQC was chosen as the primary quantum approach because:
1. It directly implements a quantum-parametrised classifier trainable with classical optimisers
2. It is well-documented in QML 0.9.1 with a stable API
3. It is computationally feasible on a statevector simulator with 239 samples and 30 iterations
4. It allows feature map / ansatz experimentation without changing the training infrastructure

---

## 6. Leakage Audit

The following leakage checks were run and **all passed** before and after training:

| Check | Status |
|-------|--------|
| `sc` excluded from features | ✅ PASS |
| `age` excluded from features | ✅ PASS |
| Preprocessor fitted on training data only | ✅ PASS (Phase 5 artefact) |
| Train/val index disjoint | ✅ PASS |
| Train/test index disjoint | ✅ PASS |
| Val/test index disjoint | ✅ PASS |
| Test data not used for tuning or model selection | ✅ PASS |
| Validation Macro-F1 logged only — no hyperparameter search on val | ✅ PASS |

---

## 7. Training Procedure

1. Loaded dataset and reproduced Phase 3 split (239/60/75) deterministically.
2. Applied Phase 5 preprocessor (transform-only, no re-fitting).
3. Encoded labels: categories 1–5 → integers 0–4.
4. Initialised VQC with `zz_feature_map` + `n_local` ansatz.
5. Set `np.random.seed(42)` for reproducible initial parameter draw.
6. Ran COBYLA optimisation for 30 iterations on the 239-sample training set.
7. Recorded loss at iterations 1, 5, 10, 15, 20, 25, 30.
8. Computed validation metrics (not used for tuning).
9. Froze model configuration.
10. Computed test metrics (single evaluation, post-freeze).

---

## 8. Results

### 8.1 Training Loss Curve

| Iteration | Loss |
|-----------|------|
| 1 | 17.3140 |
| 5 | 17.3139 |
| 10 | 17.3228 |
| 15 | 17.3146 |
| 20 | 17.3131 |
| 25 | 17.3080 |
| 30 | 17.3031 |

The loss remained nearly flat throughout training (~17.3), indicating that 30 COBYLA iterations were insufficient for VQC to meaningfully minimise the 5-class cross-entropy on this dataset configuration.

### 8.2 Validation Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.2000 |
| **Macro-F1** | **0.0940** |
| Weighted-F1 | 0.1138 |

**Per-class validation performance:**

| Category | Precision | Recall | F1 |
|----------|-----------|--------|----|
| 1 (high eGFR) | 0.125 | 0.125 | 0.125 |
| 2 | 0.227 | 0.714 | 0.345 |
| 3 | 0.000 | 0.000 | 0.000 |
| 4 | 0.000 | 0.000 | 0.000 |
| 5 (low eGFR) | 0.000 | 0.000 | 0.000 |

**Validation confusion matrix:**

```
Predicted →  Cat1  Cat2  Cat3  Cat4  Cat5
True Cat1  [   2    14     0     0     0 ]
True Cat2  [   4    10     0     0     0 ]
True Cat3  [   3     9     0     0     0 ]
True Cat4  [   2     6     0     0     0 ]
True Cat5  [   5     5     0     0     0 ]
```

The model collapsed to predicting primarily Categories 1 and 2, with no predictions for Categories 3–5.

### 8.3 Test Results (Post-Freeze, Single Evaluation)

| Metric | Value |
|--------|-------|
| Accuracy | 0.2933 |
| **Macro-F1** | **0.1528** |
| Weighted-F1 | 0.1906 |

**Per-class test performance:**

| Category | Precision | Recall | F1 |
|----------|-----------|--------|----|
| 1 (high eGFR) | 0.286 | 0.381 | 0.327 |
| 2 | 0.298 | 0.824 | 0.438 |
| 3 | 0.000 | 0.000 | 0.000 |
| 4 | 0.000 | 0.000 | 0.000 |
| 5 (low eGFR) | 0.000 | 0.000 | 0.000 |

**Test confusion matrix:**

```
Predicted →  Cat1  Cat2  Cat3  Cat4  Cat5
True Cat1  [   8    13     0     0     0 ]
True Cat2  [   3    14     0     0     0 ]
True Cat3  [   7     8     0     0     0 ]
True Cat4  [   4     6     0     0     0 ]
True Cat5  [   6     6     0     0     0 ]
```

### 8.4 Comparison with Classical Baseline (Logistic Regression)

| Metric | Classical LR (Phase 5) | Quantum VQC (Phase 6) |
|--------|----------------------|----------------------|
| Val Accuracy | 0.4167 | 0.2000 |
| **Val Macro-F1** | **0.4109** | **0.0940** |
| Test Accuracy | 0.4800 | 0.2933 |
| **Test Macro-F1** | **0.4468** | **0.1528** |

> **⚠ The VQC substantially underperformed the classical Logistic Regression baseline.** This is expected behaviour and **does not imply any negative claim about quantum computing in general**. See Section 10 (Limitations) for full discussion.

---

## 9. Saved Artefacts

| File | Description |
|------|-------------|
| `results/quantum_results.json` | Full metrics, circuit config, training config |
| `results/figures/quantum/vqc_loss_curve.png` | COBYLA loss across 30 iterations |
| `results/figures/quantum/vqc_val_cm.png` | Validation confusion matrix |
| `results/figures/quantum/vqc_test_cm.png` | Test confusion matrix |
| `results/figures/quantum/vqc_vs_lr_per_class_f1.png` | Per-class F1 comparison (LR vs VQC) |
| `models/quantum_vqc_weights.npy` | Optimised VQC parameter weights (16 values) |
| `models/quantum_label_encoder.pkl` | LabelEncoder (categories 1–5 ↔ integers 0–4) |

> **Note on model serialisation:** VQC objects in QML 0.9.1 contain an internal `parity` local lambda that is not picklable by Python's `pickle` module. The model is therefore saved via its weights array (`np.save`). To reload for inference, re-instantiate the VQC with the same circuit configuration and call `vqc.weights = np.load(weights_path)`.

---

## 10. Limitations and Analysis

### 10.1 Why VQC Underperformed

The flat loss curve (loss ≈ 17.3 throughout all 30 iterations) indicates **VQC did not converge** within this configuration. Several factors contribute:

1. **Insufficient iterations**: COBYLA with 30 iterations is minimal for a 16-parameter landscape over 239 samples. Gradient-free optimisation of VQCs typically requires hundreds to thousands of function evaluations.

2. **Barren plateau risk**: With 8 qubits and random parameter initialisation, VQCs are susceptible to barren plateaus — regions of the loss landscape where gradients vanish, causing gradient-free optimisers to stall. This is a known fundamental challenge for VQCs.

3. **5-class imbalance**: With 5 output classes and limited training data (239 samples, minimum 33 per class), the VQC's cross-entropy landscape is complex and hard to navigate with few iterations.

4. **No warmup or structured initialisation**: Random parameter initialisation (NumPy seed 42) was used. Structured initialisation (e.g., near-identity ansatz) may improve convergence.

5. **Feature encoding mismatch**: ZZFeatureMap encodes features via Pauli rotations. The relationship between biochemical features (bgr, bu, pot, etc.) and quantum Hilbert space structure is not guaranteed to be advantageous.

### 10.2 What Would Be Needed for Meaningful Results

- Many more COBYLA iterations (200–500+) — computationally expensive on simulator
- Alternative optimisers (SPSA, ADAM) that may navigate barren plateaus better
- Structured ansatz initialisation
- Dimensionality reduction (PCA to ≤4 features) to reduce qubit count and circuit depth
- Kernel-based approach (QSVC) as an alternative that avoids the barren plateau problem

### 10.3 Scientific Integrity

- Test data was **not used** for model selection, tuning, or any decision
- The VQC configuration was fixed **before** test evaluation
- All leakage checks passed
- No quantum advantage is claimed
- The result accurately reflects the current performance of VQC on this dataset/configuration

### 10.4 eGFR Category Limitations

As documented in Phase 2:
- Categories are derived using the male-default CKD-EPI equation (sex unknown)
- eGFR ≥ 60 does not confirm the absence of CKD
- Original binary labels span all eGFR categories
- Categories are exploratory research constructs, not clinical diagnoses

---

## 11. Reproducibility Information

| Parameter | Value |
|-----------|-------|
| Dataset file | `ckd_egfr_categories_observed_only.csv` |
| Feature set | bgr, bu, pot, hemo, pcv, rbcc, sg, sod |
| Preprocessor | `models/classical_preprocessor.pkl` (Phase 5) |
| Split random state | 42 |
| NumPy global seed | 42 (set before VQC initialisation) |
| Feature map | `zz_feature_map(feature_dimension=8, reps=1)` |
| Ansatz | `n_local(num_qubits=8, rotation_blocks=['ry'], entanglement_blocks='cx', entanglement='linear', reps=1)` |
| Optimizer | `COBYLA(maxiter=30)` |
| Loss | `cross_entropy` |
| Sampler | `StatevectorSampler()` |
| Qiskit | 2.5.2 |
| QML | 0.9.1 |

---

## 12. Phase 1–5 Integrity Confirmation

The following Phase 1–5 artefacts were **read but never modified**:

- `data/processed/ckd_egfr_categories_observed_only.csv` — unchanged
- `models/classical_preprocessor.pkl` — unchanged
- `models/best_classical_model.pkl` — unchanged
- `results/classical_results.json` — unchanged
- All reports in `reports/` (except addition of this file) — unchanged

**Phase 1–5 work is fully preserved.**

---

## 13. Next Steps (Phase 7 — NOT YET IMPLEMENTED)

Phase 7 (late fusion of classical + quantum models) is explicitly out of scope for Phase 6 and has **not** been implemented.

---

**Report Written:** 2026-10-08  
**Status:** PHASE 6 COMPLETE  
**Phase 1–5 Integrity:** CONFIRMED PRESERVED  
**Leakage Checks:** ALL PASSED  
**Next Phase:** Phase 7 (Late Fusion) — AWAITING APPROVAL
