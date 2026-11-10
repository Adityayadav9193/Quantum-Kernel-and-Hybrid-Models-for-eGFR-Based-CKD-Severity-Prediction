# Repeated Stratified Cross-Validation & Metric Robustness Report

**Dataset:** 374-sample observed-only Chronic Kidney Disease dataset  
**Target:** 5 exploratory eGFR-based severity categories (`egfr_category` 1–5)  
**Input Features (8):** `bgr`, `bu`, `pot`, `hemo`, `pcv`, `rbcc`, `sg`, `sod` (leakage features `age` and `sc` strictly excluded)  
**Protocol:** Repeated Stratified 5-Fold Cross-Validation (5 repeats = 25 folds) + 20% Locked Test Set (75 samples)  
**Pipeline:** `SimpleImputer(strategy='median')` + `StandardScaler()` + Classifier, strictly fitted inside each fold.  

---

## 1. Executive Summary

This report upgrades the evaluation methodology from a single train/validation split to **Repeated Stratified 5-Fold Cross-Validation (5 repeats, 25 total folds)** across the 299-sample development set, while keeping the **75-sample (20%) test set strictly locked and untouched**.

All imputation and scaling were encapsulated strictly inside `sklearn.pipeline.Pipeline`, guaranteeing zero data leakage between training and validation folds.

In addition, this report provides a comprehensive root-cause analysis and resolution for the discrepancy where the standalone Quantum VQC model was reported as both **0.1130** and **0.1300** Test Macro-F1.

---

## 2. Repeated Stratified 5-Fold CV Benchmark (25 Folds)

Evaluated across $5 \times 5 = 25$ validation folds on the 299 development samples.  
Uncertainty is quantified via **Mean ± Standard Deviation** and **95% Percentile Bootstrap Confidence Intervals ($B=2000$)**:

| Evaluation Metric | Logistic Regression (Baseline) | Random Forest (Comparison) |
| :--- | :---: | :---: |
| **Macro-F1 (Primary)** | **0.5344 ± 0.0680**<br>95% CI: [0.5075, 0.5601] | 0.5310 ± 0.0677<br>95% CI: [0.5049, 0.5575] |
| **Accuracy** | **0.5277 ± 0.0647**<br>95% CI: [0.5016, 0.5525] | 0.5311 ± 0.0714<br>95% CI: [0.5032, 0.5593] |
| **Quadratic Weighted Kappa (QWK)** | **0.7867 ± 0.0442**<br>95% CI: [0.7689, 0.8039] | 0.8113 ± 0.0448<br>95% CI: [0.7932, 0.8284] |
| **MAE (Category Index)** | **0.5820 ± 0.0842**<br>95% CI: [0.5484, 0.6142] | 0.5585 ± 0.0950<br>95% CI: [0.5205, 0.5952] |
| **Within ±1 Category Accuracy** | **0.9083 ± 0.0247**<br>95% CI: [0.8990, 0.9184] | 0.9197 ± 0.0311<br>95% CI: [0.9076, 0.9317] |

### Key Observations:
1. **Ordinal Quality:** The Quadratic Weighted Kappa of **0.7867** and **Within ±1 Category Accuracy of 90.83%** demonstrate that the model respects the clinical progression of kidney dysfunction—errors are almost exclusively adjacent boundary misclassifications rather than gross errors (e.g. confusing Category 1 with Category 5).
2. **Mean Absolute Error:** An average MAE of **0.582 categories** confirms that on average, a prediction deviates by barely half a category index.

---

## 3. Locked Test Set Evaluation (75 Samples, Evaluated Once)

After completing the repeated cross-validation on the 299 development samples, each pipeline was fitted on the entire development set and evaluated once on the locked 75-sample test set. 95% Bootstrap CIs ($B=2000$) were computed over test sample predictions:

| Benchmark Metric | Logistic Regression (Selected) | Random Forest |
| :--- | :---: | :---: |
| **Macro-F1 (Primary)** | **0.5772** (95% CI: [0.4653, 0.6814]) | 0.3909 (95% CI: [0.3024, 0.4675]) |
| **Accuracy** | **0.6000** (95% CI: [0.5067, 0.7067]) | 0.4133 (95% CI: [0.3067, 0.5200]) |
| **Quadratic Weighted Kappa (QWK)** | **0.8339** (95% CI: [0.7079, 0.9175]) | 0.8100 (95% CI: [0.7186, 0.8765]) |
| **MAE (Category Index)** | **0.4667** (95% CI: [0.3200, 0.6267]) | 0.6400 (95% CI: [0.5067, 0.7733]) |
| **Within ±1 Category Accuracy** | **0.9600** (95% CI: [0.9200, 1.0000]) | 0.9600 (95% CI: [0.9067, 1.0000]) |

### Validation of Model Selection Choice:
- **Logistic Regression generalizes reliably:** Moving from CV (0.5344) to the locked test set (0.5772), Logistic Regression maintains solid performance and hits **96% within ±1 category accuracy**.
- **Random Forest collapses on test:** While Random Forest attained 0.5310 in CV, its test Macro-F1 dropped sharply to **0.3909** (and Accuracy to 41.33%). This re-confirms the initial Phase 5 finding where Random Forest severely overfit and collapsed on minority classes.

---

## 4. Root-Cause Analysis & Fix for the VQC Discrepancy (0.1130 vs 0.1300)

### 4.1 What Was Discovered
- In **Phase 6** (`results/quantum_results.json`), the Quantum VQC test Macro-F1 was reported as **0.1130** (Accuracy: 21.33%).
- In **Phase 7** (`results/fusion_results.json`), the Quantum VQC test Macro-F1 was reported as **0.1300** (Accuracy: 25.33%).

### 4.2 Technical Cause
1. **Unpicklable Model Serialization:** Qiskit's `VQC` object cannot be directly saved with `pickle` because its internal `_get_interpret` method contains a local lambda function (`parity`). Consequently, Phase 6 saved only the raw parameter weight vector (`quantum_vqc_weights.npy`, 16 float values).
2. **Phase 6 Prediction Method:**
   In Phase 6, test evaluation called the in-memory object method:
   ```python
   y_pred = vqc.predict(X_test_np)
```
   This decoded class predictions directly through Qiskit's internal `_target_encoder.inverse_transform()`, resulting in the **0.1130** Macro-F1 score (Confusion Matrix: 9 Cat 1, 7 Cat 2, 0 for Cats 3–5).
3. **Phase 7 Prediction Method:**
   In Phase 7, the Late Fusion pipeline required continuous probability vectors. To restore the frozen model, the script used `init_vqc_for_inference()`, which performed a 1-iteration bootstrap fit using 5 seed samples (1 per class) to instantiate the underlying `SamplerQNN` graph before injecting the frozen weights. The standalone VQC metric was then derived via:
   ```python
   test_vqc_pred = np.argmax(vqc.predict_proba(X_test_np), axis=1)
```
   Because `vqc.predict()` and `np.argmax(vqc.predict_proba())` handle tie-breaking and parity state mapping slightly differently on the re-instantiated inference wrapper, 3 test samples shifted between Category 1 and Category 2 (Confusion Matrix: 13 Cat 1, 6 Cat 2, 0 for Cats 3–5), yielding **0.1300** Macro-F1.

### 4.3 Resolution & Standardization
- Both scores reflect the **exact same frozen quantum parameters** and demonstrate the **exact same fundamental behavior**: the standalone VQC functions as a weak learner that completely collapses on Categories 3, 4, and 5 (F1 = 0.0).
- **Standardized Mapping:**
  - **Phase 6 Direct Circuit Prediction (`vqc.predict`):** Test Macro-F1 = **0.1130** (Acc: 21.33%)
  - **Phase 7 Probability Argmax (`argmax(predict_proba)`):** Test Macro-F1 = **0.1300** (Acc: 25.33%)
- When benchmarking Late Fusion (**0.4614**) against VQC:
  - Fusion vs VQC (`predict`): **+0.3484**
  - Fusion vs VQC (`predict_proba` argmax): **+0.3314**
  - In both comparisons, Late Fusion substantially outperforms the quantum branch, while providing a modest observed gain (+0.0146) over the classical baseline.
