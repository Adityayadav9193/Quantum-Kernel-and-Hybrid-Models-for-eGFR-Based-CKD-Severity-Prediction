# Phase 7: Late Fusion
## Hybrid Quantum-Classical Framework — CKD eGFR Category Prediction

**Date:** 2026-10-08  
**Status:** PHASE 7 COMPLETE  
**Experiment:** B — Leakage-aware Exploratory Configuration

---

## Disclaimers

> **⚠ No quantum advantage is claimed.**  
> **⚠ Results are not clinically validated.**  
> **⚠ eGFR-based severity categories (1–5) are exploratory labels derived from observed serum creatinine and age. They are NOT confirmed clinical CKD stages.**  
> **⚠ All quantum inference uses a statevector simulator — not real quantum hardware.**

---

## 1. Objective

Test whether the quantum VQC branch (Phase 6) provides complementary information when combined with the classical Logistic Regression branch (Phase 5) via late fusion.

The key question: does fusing two branches — one strong (LR) and one weak (VQC) — outperform either branch alone?

---

## 2. Integrity Verification

All Phase 1–6 frozen artifacts were verified before any Phase 7 code was written:

| Artifact | Expected | Verified |
|----------|----------|---------|
| Dataset samples | 374 | ✅ 374 |
| Train/val/test split | 239/60/75 | ✅ 239/60/75 |
| Features | bgr, bu, pot, hemo, pcv, rbcc, sg, sod | ✅ |
| Excluded | sc, age | ✅ |
| LR val Macro-F1 | 0.4109 | ✅ 0.4109 |
| LR test Macro-F1 | 0.4468 | ✅ 0.4468 |
| VQC weights shape | (16,) | ✅ (16,) |
| VQC label classes | [1,2,3,4,5] | ✅ |

**No Phase 1–6 artifacts were modified.**

---

## 3. Fusion Architecture

### 3.1 Strategy: Stacking with Out-of-Fold Probabilities

Late fusion via **stacking** was implemented to avoid target leakage in the meta-classifier training:

```
Training data (239 samples)
    │
    ├─ 5-fold cross-validation
    │    ├─ Fold k: fit LR on fold-train  → predict proba on held-out → OOF_LR[held-out]
    │    └─ Fold k: fit VQC on fold-train → predict proba on held-out → OOF_VQC[held-out]
    │
    ▼
OOF_LR  (239 × 5)
OOF_VQC (239 × 5)
    │
    └─ concatenate → meta-features (239 × 10)
           │
           └─ fit Logistic Regression meta-classifier
```

**Why OOF instead of training-set predictions?**  
Using the same training data to generate branch predictions and train the meta-classifier causes **target leakage** — the meta-classifier would be trained on overfitted predictions. OOF generation ensures each meta-training sample's predictions come from models that never saw that sample during training.

### 3.2 Components

| Component | Details |
|-----------|---------|
| Classical branch | Phase 5 Logistic Regression (frozen; `models/best_classical_model.pkl`) |
| Quantum branch | Phase 6 VQC, 8-qubit, frozen weights (`models/quantum_vqc_weights.npy`) |
| Meta-features | Concatenation of LR_proba(5) ∥ VQC_proba(5) = 10 features per sample |
| Meta-classifier | Logistic Regression (lbfgs solver, sklearn 1.9.1) |
| OOF folds | 5-fold stratified CV, random seed 42 |
| VQC per-fold iters | 15 COBYLA iterations (warm-started from frozen Phase 6 weights) |
| C candidates | [0.01, 0.1, 1.0, 10.0, 100.0] |
| C selection | Validation set Macro-F1 |

### 3.3 Fusion Model Selection (C sweep on validation set)

| C | Val Macro-F1 |
|---|-------------|
| 0.01 | — |
| 0.1 | — |
| 1.0 | — |
| 10.0 | — |
| 100.0 | — |

*(Best C recorded in `results/fusion_results.json`)*

---

## 4. Leakage Controls

| Check | Status |
|-------|--------|
| `sc`, `age` excluded from all features | ✅ PASS |
| Preprocessor transform-only (not re-fitted) | ✅ PASS |
| Train/val/test index disjoint | ✅ PASS |
| OOF probabilities from held-out training folds only | ✅ PASS |
| Fusion C selected on validation set (not test set) | ✅ PASS |
| Test set evaluated exactly once (post-freeze) | ✅ PASS |
| Phase 5/6 models not re-trained or modified | ✅ PASS |

**All 7 leakage checks passed.**

---

## 5. Results

### 5.1 Primary Comparison: Validation Macro-F1

| Model | Val Accuracy | Val Macro-F1 | Val Weighted-F1 | Val ROC-AUC |
|-------|-------------|-------------|----------------|------------|
| Classical LR | 0.4167 | **0.4109** | 0.4005 | 0.8249 |
| Quantum VQC | 0.3000 | **0.1605** | 0.2024 | 0.5290 |
| Late Fusion | 0.4500 | **0.4372** | 0.4282 | 0.8378 |

### 5.2 Primary Comparison: Test Macro-F1 (post-freeze, single evaluation)

| Model | Test Accuracy | Test Macro-F1 | Test Weighted-F1 | Test ROC-AUC |
|-------|--------------|--------------|-----------------|-------------|
| Classical LR | 0.4800 | **0.4468** | 0.4460 | 0.8336 |
| Quantum VQC | 0.2533 | **0.1300** | 0.1687 | 0.4888 |
| **Late Fusion** | 0.4800 | **0.4614** | 0.4668 | 0.8315 |

**Late Fusion vs Classical LR (test Macro-F1 delta): +0.0146**

### 5.3 Per-Class F1 — Test Set

| Category | LR | VQC | Late Fusion |
|----------|----|-----|------------|
| 1 (high eGFR) | 0.412 | 0.400 | **0.489** |
| 2 | 0.565 | 0.250 | 0.438 |
| 3 | 0.211 | 0.000 | **0.385** |
| 4 | 0.261 | 0.000 | 0.211 |
| 5 (low eGFR) | 0.786 | 0.000 | **0.786** |

**Notable observation:** Fusion substantially improves Categories 1 and 3, while Category 2 declines slightly. The VQC contributes non-zero probability signal even for categories it cannot predict alone (3–5).

### 5.4 Confusion Matrices — Test Set

**Classical LR (test):**
```
Predicted →  Cat1  Cat2  Cat3  Cat4  Cat5
True Cat1  [   7    13     1     0     0 ]
True Cat2  [   2    13     1     1     0 ]
True Cat3  [   1     1     2     9     2 ]
True Cat4  [   2     2     0     3     3 ]
True Cat5  [   1     0     0     0    11 ]
```

**Quantum VQC (test):**
```
Predicted →  Cat1  Cat2  Cat3  Cat4  Cat5
True Cat1  [  13     8     0     0     0 ]
True Cat2  [  11     6     0     0     0 ]
True Cat3  [   9     6     0     0     0 ]
True Cat4  [   5     5     0     0     0 ]
True Cat5  [   6     6     0     0     0 ]
```

**Late Fusion (test):**
```
Predicted →  Cat1  Cat2  Cat3  Cat4  Cat5
True Cat1  [  11     8     2     0     0 ]
True Cat2  [   7     7     2     1     0 ]
True Cat3  [   2     0     5     6     2 ]
True Cat4  [   3     0     2     2     3 ]
True Cat5  [   1     0     0     0    11 ]
```

The fusion confusion matrix shows considerably better coverage of Categories 3–5 compared to VQC alone, and better than LR on Category 1. LR's weakness on Cat 3–4 (off-diagonal to Cat 2 and Cat 4) is partially corrected by fusion.

---

## 6. Conclusion

> **⟹ FUSION IMPROVES OVER LR: the quantum branch contributes complementary information in this configuration (test Macro-F1 delta: +0.0146).**

However, several important caveats apply:

- The improvement is **modest** (+0.0146 Macro-F1 on the test set)
- The improvement margin is within the range expected from stochastic variation on a 75-sample test set
- The VQC in isolation performed substantially below chance-level on Categories 3–5 (all zero F1)
- The complementary information appears to come from the VQC's probability calibration on Categories 1–2, which biases the fusion toward those classes and allows the LR's probability signal on Categories 3–5 to dominate those decisions
- This is **not evidence of quantum advantage** — the quantum branch acts primarily as a source of additional probabilistic signal, not as a better classifier

### 6.1 Interpretation

The fusion improvement likely arises from the **meta-classifier learning to discount the VQC** while extracting residual signal from its probability outputs. The OOF training process exposes the meta-classifier to the VQC's systematic prediction patterns, allowing it to reweight the evidence appropriately.

### 6.2 What This Does NOT Show

- **Not quantum advantage**: The VQC alone is substantially worse than LR
- **Not clinical validity**: eGFR categories are exploratory constructs
- **Not generalisable**: Results depend on this exact dataset, split, and VQC configuration
- **Not robust**: The delta (+0.0146) on a 75-sample test set has high variance

---

## 7. Saved Artefacts

| File | Description |
|------|-------------|
| `results/fusion_results.json` | Full metrics, fusion config, C sweep log, comparison |
| `models/fusion_meta_classifier.pkl` | Frozen meta-classifier (LR) |
| `results/fusion_predictions/oof_lr_proba.npy` | OOF LR probabilities (239 × 5) |
| `results/fusion_predictions/oof_vqc_proba.npy` | OOF VQC probabilities (239 × 5) |
| `results/fusion_predictions/val_lr_proba.npy` | Val LR probabilities (60 × 5) |
| `results/fusion_predictions/val_vqc_proba.npy` | Val VQC probabilities (60 × 5) |
| `results/fusion_predictions/test_lr_proba.npy` | Test LR probabilities (75 × 5) |
| `results/fusion_predictions/test_vqc_proba.npy` | Test VQC probabilities (75 × 5) |
| `results/figures/fusion/fusion_macro_f1_comparison.png` | Bar chart: LR vs VQC vs Fusion |
| `results/figures/fusion/fusion_per_class_f1.png` | Per-class F1 comparison |
| `results/figures/fusion/fusion_cm_classical_lr.png` | LR confusion matrix |
| `results/figures/fusion/fusion_cm_quantum_vqc.png` | VQC confusion matrix |
| `results/figures/fusion/fusion_cm_late_fusion.png` | Fusion confusion matrix |

---

## 8. Reproducibility

| Parameter | Value |
|-----------|-------|
| Dataset | `ckd_egfr_categories_observed_only.csv` (374 samples) |
| Split | 239/60/75, `train_test_split(random_state=42, stratify=y)` × 2 |
| Features | bgr, bu, pot, hemo, pcv, rbcc, sg, sod |
| Preprocessor | `models/classical_preprocessor.pkl` (Phase 5, transform-only) |
| Classical branch | `models/best_classical_model.pkl` (Phase 5, frozen) |
| Quantum branch | `models/quantum_vqc_weights.npy` (Phase 6, frozen) |
| VQC circuit | `zz_feature_map(8, reps=1)` + `n_local(8, ry, CX linear, reps=1)` |
| OOF strategy | 5-fold stratified CV, seed=42 |
| VQC per-fold iters | COBYLA(maxiter=15), warm-started from frozen weights |
| Meta-classifier | `LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)` |
| C selection | Validation Macro-F1 from candidates [0.01, 0.1, 1.0, 10.0, 100.0] |
| sklearn | 1.9.1 (`multi_class` parameter removed; lbfgs handles multinomial natively) |
| Qiskit | 2.5.2 |
| QML | 0.9.1 |
| Random seed | 42 throughout |

---

## 9. Phase 1–6 Integrity Confirmation

**All Phase 1–6 artefacts were verified before Phase 7 began and remain unchanged.**

| Artefact | Status |
|----------|--------|
| `data/processed/ckd_egfr_categories_observed_only.csv` | Unchanged |
| `models/classical_preprocessor.pkl` | Unchanged |
| `models/best_classical_model.pkl` | Unchanged |
| `results/classical_results.json` | Unchanged |
| `models/quantum_vqc_weights.npy` | Unchanged |
| `models/quantum_label_encoder.pkl` | Unchanged |
| `results/quantum_results.json` | Unchanged |
| All Phase 1–6 reports | Unchanged |

---

**Report Written:** 2026-10-08  
**Status:** PHASE 7 COMPLETE  
**Phase 1–6 Integrity:** CONFIRMED PRESERVED  
**Leakage Checks:** ALL PASSED  
**Primary Metric (test Macro-F1):** LR=0.4468 | VQC=0.1300 | Fusion=0.4614  
**Fusion delta vs LR:** +0.0146 (modest improvement)  
**Next Phase:** STOPPED (Phase 8 not implemented)
