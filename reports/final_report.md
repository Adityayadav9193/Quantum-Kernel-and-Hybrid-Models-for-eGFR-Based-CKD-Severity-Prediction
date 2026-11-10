# Final Scientific Report: Rigorous Nested Cross-Validation, Quantum Control, and Benchmark of Multi-Grade Chronic Kidney Disease Detection

**Project:** Hybrid Quantum-Classical Framework for Explainable Multi-Grade CKD Detection  
**Dataset:** UCI Machine Learning Repository — Chronic Kidney Disease (ID 336), $N = 374$ patients  
**Target:** eGFR Severity Categories derived via CKD-EPI equation without input leakage (`sc` and `age` strictly excluded from model inputs)  
**Evaluation Protocol:** 5-fold $\times$ 5-repeat Nested Cross-Validation (Outer 25 folds, Inner 3 folds for hyperparameter and preprocessing tuning)  
**Artifact Generated:** `results/final_nested_cv.json`  

---

## Executive Summary

This report establishes the final, statistically verified conclusions of the project. Earlier phases contained two critical baseline artifacts:
1. A double-scaling bug in Phase 5 that distorted specific gravity to $-279.5$, artificially deflating the classical Logistic Regression CV Macro-F1 to **0.3050** (corrected here to **0.5419** under full nested CV).
2. A discrepancy between reported Phase 6 and Phase 7 test Macro-F1 scores for the Variational Quantum Classifier (**0.1130** vs **0.1300**), which arose solely from Qiskit argmax parity tie-breaking across identical, collapsed zero-recall predictions.

Under a leakage-free nested cross-validation framework across all 374 patients, **no quantum advantage was detected**:
- The quantum fidelity kernel achieved lower downstream performance (QWK = **0.7109**, Macro-F1 = **0.4338**) than an identical classical RBF kernel control (QWK = **0.7763**, Macro-F1 = **0.4415**).
- Sample-level paired bootstrap analysis ($B = 2000$) on pooled out-of-fold predictions showed that late fusion with quantum probabilities resulted in a **negative delta** in Macro-F1 ($\Delta = -0.0481$, 95% CI: $[-0.0918, -0.0046]$) and a delta in Quadratic Weighted Kappa spanning zero ($\Delta = -0.0057$, 95% CI: $[-0.0269, +0.0154]$).
- Ordinal modeling via **continuous log-eGFR regression with post-hoc threshold binning** emerged as the top-performing architecture, reaching a QWK of **0.8204** (5-class) and **0.8563** (3-class), with 95.2% and 98.9% of predictions within $\pm 1$ grade, respectively.

---

## 1. Baseline Reconciliation & Technical Post-Mortem

### 1.1 The Classical Baseline Discrepancy (0.3050 vs 0.5344)
In Phase 5 (`src/classical/01_classical_baselines.py`), 5-fold CV Macro-F1 for balanced multinomial Logistic Regression on 8 features was reported as **0.3050**, whereas subsequent repeated CV runs on development subsets yielded **0.5344**.

**Root Cause:**
A line-by-line diff of `src/classical/01_classical_baselines.py` revealed that `create_preprocessing_pipeline` performed `scaler.fit_transform(X_train)` in-place (Lines 104–109). Subsequently, in `main()`, `apply_preprocessing(X_train)` was called on the already standardized array (Line 373). 
Because `sg` (Specific Gravity) originally has minimal variance ($\sigma \approx 0.005$, range 1.005 to 1.025), standardizing it once produced centered values $\sim 0$. Standardizing it a second time across small fold variances created severe numerical distortion, shifting values to **$-279.5$**. This destabilized the L2 regularization penalty, causing coefficients to explode and crippling cross-validation accuracy down to 0.3050.

When the preprocessor is encapsulated within an isolated `sklearn.pipeline.Pipeline` where scaling is fitted strictly once per training split:
- 5-fold CV Macro-F1 on the 239 training subset: **0.5045**
- 5-fold CV Macro-F1 on the 299 development subset: **0.5344**
- 5-repeat 5-fold Nested CV Macro-F1 across all 374 samples: **0.5419**

The 0.3050 score was an implementation defect and is formally retracted.

### 1.2 The VQC Test Macro-F1 Discrepancy (0.1130 vs 0.1300)
In Phase 6, the standalone VQC test Macro-F1 was reported as **0.1130**. In Phase 7, the reported standalone VQC test Macro-F1 was **0.1300**.

**Root Cause:**
Phase 6 evaluated test performance directly using `vqc.predict(X_test)`. Phase 7 required full probability distributions for meta-classifier fusion, so it re-instantiated the inference graph via `init_vqc_for_inference()` with a 5-seed-sample bootstrap to initialize `SamplerQNN`. Standalone predictions were then derived via `np.argmax(predict_proba, axis=1)`.
Both executions evaluated the **exact same 16 frozen parameters** and suffered identical severe minority class collapse:
- Category 1 Recall: 0.28
- Category 2 Recall: 0.32
- Category 3 Recall: 0.00
- Category 4 Recall: 0.00
- Category 5 Recall: 0.00

The difference (+0.017 Macro-F1) was entirely caused by parity tie-breaking in Qiskit’s bitstring-to-class mapping when converting quantum state measurement counts into multi-class probabilities.

| Model / Pipeline | Reported Metric | Actual Corrected Metric | Root Cause / Verification |
| :--- | :---: | :---: | :--- |
| **Phase 5 LR (8 feats, CV)** | 0.3050 | **0.5045 / 0.5419** | Double-scaling in-place shifted `sg` to -279.5; fixed via unified `Pipeline`. |
| **Phase 6 VQC (Test)** | 0.1130 | **0.1130** | Direct `vqc.predict()`; complete collapse on Categories 3–5. |
| **Phase 7 VQC (Test)** | 0.1300 | **0.1130 / 0.1300** | Identical parameters; discrepancy caused by argmax tie-breaking on zero-probabilities. |

---

## 2. Leakage-Free Nested Cross-Validation Benchmark

### 2.1 Experimental Protocol
- **Cohort:** All $N = 374$ patients from UCI CKD (after removing samples with missing target).
- **Split Structure:** Outer 5-fold $\times$ 5-repeat stratified CV ($25$ unique evaluation folds).
- **Inner Tuning:** 3-fold inner stratified CV strictly inside training splits for imputer selection (Median vs KNN vs Iterative), hyperparameter tuning ($C$, penalty, tree depth), and feature scaling.
- **Leakage Prevention:** Serum creatinine (`sc`) and `age` were strictly excluded from input matrices.
- **Expanded Features (10 selected):** `bu`, `htn`, `bgr`, `al`, `pcv`, `pe`, `hemo`, `wbcc`, `sg`, `sod` with one-hot encoding of categorical indicators and missingness masks.
- **Metrics Evaluated:**
  - Quadratic Weighted Kappa (QWK) — *Primary clinical metric for ordinal severity*
  - Macro-F1
  - Multi-class Accuracy
  - Mean Absolute Error (MAE) of category indices
  - Within $\pm 1$ Category Accuracy

### 2.2 5-Class Target Benchmark Results ($N = 374$, 25 Outer Folds)
Target Categories: Stage 1 ($\ge 90$), Stage 2 ($60-89$), Stage 3 ($30-59$), Stage 4 ($15-29$), Stage 5 ($< 15$ mL/min/1.73m²).

| Architecture | QWK (Primary) [95% CI] | Macro-F1 [95% CI] | Accuracy [95% CI] | MAE [95% CI] | Within $\pm 1$ [%] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2)** | 0.8021 [0.7882, 0.8150] | **0.5419** [0.5233, 0.5587] | **0.5417** [0.5232, 0.5586] | 0.5508 [0.5289, 0.5744] | 92.6% |
| **Logistic Regression (ElasticNet)** | 0.8029 [0.7889, 0.8160] | 0.5414 [0.5239, 0.5581] | 0.5401 [0.5223, 0.5579] | 0.5514 [0.5284, 0.5757] | 92.6% |
| **SVM-RBF (Calibrated)** | 0.7885 [0.7750, 0.8010] | 0.4434 [0.4309, 0.4554] | 0.4717 [0.4592, 0.4845] | 0.6374 [0.6198, 0.6563] | 90.6% |
| **Random Forest (Balanced)** | 0.8121 [0.8026, 0.8218] | 0.4963 [0.4716, 0.5207] | 0.4989 [0.4746, 0.5238] | 0.5792 [0.5515, 0.6072] | 93.3% |
| **HistGradientBoosting (Balanced)** | 0.8147 [0.8048, 0.8258] | 0.4985 [0.4805, 0.5196] | 0.5000 [0.4811, 0.5223] | 0.5759 [0.5502, 0.5983] | 93.4% |
| **Continuous Log-eGFR Regressor + Binning** | **0.8204** [0.8114, 0.8298] | 0.5236 [0.5024, 0.5457] | 0.5097 [0.4883, 0.5321] | **0.5486** [0.5251, 0.5710] | **95.2%** |
| **Calibrated Soft-Voting Ensemble (Top 3)** | 0.8068 [0.7971, 0.8163] | 0.5087 [0.4955, 0.5219] | 0.5102 [0.4984, 0.5225] | 0.5770 [0.5578, 0.5963] | 92.8% |

### 2.3 3-Class Clinically Grouped Benchmark Results ($N = 374$, 25 Outer Folds)
Grouping: Early Stage G1–2 ($\ge 60$, $n=188$), Moderate Stage G3 ($30-59$, $n=84$), Severe Stage G4–5 ($< 30$, $n=102$).

| Architecture | QWK (Primary) [95% CI] | Macro-F1 [95% CI] | Accuracy [95% CI] | MAE [95% CI] | Within $\pm 1$ [%] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2)** | 0.8208 [0.8027, 0.8385] | 0.7393 [0.7163, 0.7640] | 0.7777 [0.7568, 0.7990] | 0.2346 [0.2122, 0.2562] | 98.8% |
| **Logistic Regression (ElasticNet)** | 0.8253 [0.8083, 0.8426] | 0.7413 [0.7185, 0.7648] | 0.7798 [0.7593, 0.8006] | 0.2309 [0.2092, 0.2517] | 98.9% |
| **SVM-RBF (Calibrated)** | 0.8098 [0.7913, 0.8277] | 0.7130 [0.6909, 0.7350] | 0.7734 [0.7550, 0.7914] | 0.2448 [0.2252, 0.2649] | 98.2% |
| **Random Forest (Balanced)** | 0.8342 [0.8205, 0.8480] | 0.7322 [0.7146, 0.7501] | 0.7813 [0.7651, 0.7969] | 0.2278 [0.2110, 0.2446] | 99.1% |
| **HistGradientBoosting (Balanced)** | 0.8261 [0.8174, 0.8356] | 0.7075 [0.6967, 0.7188] | 0.7717 [0.7628, 0.7803] | 0.2396 [0.2300, 0.2489] | 98.9% |
| **Continuous Log-eGFR Regressor + Binning** | **0.8563** [0.8425, 0.8711] | **0.7701** [0.7518, 0.7897] | **0.8145** [0.7990, 0.8303] | **0.1962** [0.1785, 0.2134] | **98.9%** |
| **Calibrated Soft-Voting Ensemble (Top 3)** | 0.8389 [0.8336, 0.8443] | 0.7444 [0.7396, 0.7492] | 0.7888 [0.7850, 0.7925] | 0.2203 [0.2150, 0.2257] | 99.1% |

### 2.4 Per-Class Recall and Collapse Analysis
To ensure no architecture suffered from minority class collapse, per-class sensitivity was tracked across all 25 outer test folds:

```
5-Class Target Recalls:
  LR ElasticNet       : Cat 1: 54.5% | Cat 2: 45.9% | Cat 3: 53.0% | Cat 4: 43.5% | Cat 5: 74.2%  (Zero collapse)
  SVM-RBF             : Cat 1: 62.2% | Cat 2: 15.5% | Cat 3: 47.0% | Cat 4: 30.4% | Cat 5: 79.6%  (Depressed Cat 2)
  Random Forest       : Cat 1: 50.9% | Cat 2: 34.4% | Cat 3: 51.1% | Cat 4: 35.7% | Cat 5: 79.4%  (Balanced recall)
  Continuous Regressor: Cat 1: 46.0% | Cat 2: 35.0% | Cat 3: 61.3% | Cat 4: 52.8% | Cat 5: 66.7%  (Strong intermediate recall)

3-Class Target Recalls:
  LR ElasticNet       : Early (G1-2): 86.4% | Moderate (G3): 66.1% | Severe (G4-5): 72.0%
  Random Forest       : Early (G1-2): 88.1% | Moderate (G3): 57.7% | Severe (G4-5): 75.1%
  Continuous Regressor: Early (G1-2): 89.4% | Moderate (G3): 61.0% | Severe (G4-5): 81.8%
```

All tuned models maintained active positive recall across all categories, completely avoiding the zero-recall collapse seen in the Phase 6 standalone VQC.

---

## 3. Quantum Control Experiment

To test whether the quantum kernel possesses any geometric advantage over classical kernels, an identical control experiment was conducted on the same 4 PCA components scaled to $[0, \pi]$:
- **Classical Control:** Support Vector Classifier with an RBF Kernel ($K_{ij} = \exp(-\gamma \|x_i - x_j\|^2)$, $\gamma = 0.25$).
- **Quantum Model:** Support Vector Classifier with a `FidelityStatevectorKernel` evaluated using a 4-qubit `zz_feature_map(feature_dimension=4, reps=1)`.

### 3.1 Kernel-Target Alignment (KTA) & Downstream Performance

$$\text{KTA}(K, Y) = \frac{\langle K, Y \rangle_F}{\|K\|_F \|Y\|_F}$$

| Metric | Classical RBF Kernel | Quantum Fidelity Kernel | Finding |
| :--- | :---: | :---: | :--- |
| **Kernel-Target Alignment (KTA)** | 0.1170 | **0.1957** | Higher mathematical alignment in feature Hilbert space. |
| **Nested CV QWK** | **0.7763** [0.7570, 0.7950] | 0.7109 [0.6895, 0.7312] | Classical RBF achieves higher ordinal agreement ($\Delta = +0.065$). |
| **Nested CV Macro-F1** | **0.4415** [0.4206, 0.4623] | 0.4338 [0.4119, 0.4556] | Statistically indistinguishable performance. |

**Conclusion:** Although the quantum kernel yields a higher raw KTA, this does not translate into improved downstream generalizability. On the held-out validation folds, the classical RBF kernel outperforms the quantum kernel in QWK while achieving matching Macro-F1. There is no evidence of quantum geometric advantage on this tabular clinical dataset.

---

## 4. Proper Fusion Test with Paired Bootstrap

To overcome the pseudo-replication and low statistical power of Wilcoxon signed-rank tests over 5 fold averages ($N=5$), we pooled the out-of-fold predictions across all $N = 374$ patients and conducted a sample-level paired bootstrap test with $B = 2000$ iterations.

### 4.1 Pooled Out-of-Fold Performance Comparison ($N = 374$)

| Model Configuration | Pooled OOF Macro-F1 | Pooled OOF QWK |
| :--- | :---: | :---: |
| **Classical LR (ElasticNet)** | **0.5590** | **0.8099** |
| **Quantum Fidelity SVC** | 0.4195 | 0.7109 |
| **Weighted Late Fusion (75% Classical + 25% Quantum)** | 0.5109 | 0.8042 |
| **Stacking Meta-Classifier (Multinomial LR on OOF Probs)** | 0.5357 | 0.7981 |

### 4.2 Paired Bootstrap Statistical Significance Test

$$\Delta_{\text{Macro-F1}} = \text{Macro-F1}_{\text{Fusion}} - \text{Macro-F1}_{\text{Classical}}$$
$$\Delta_{\text{QWK}} = \text{QWK}_{\text{Fusion}} - \text{QWK}_{\text{Classical}}$$

- **Point Estimate $\Delta_{\text{Macro-F1}}$:** $-0.0481$
  - **95% Paired Bootstrap Confidence Interval:** $[-0.0918, -0.0046]$
  - *Status:* **Statistically significant decrease.** Blending the weaker quantum probability distributions degraded Macro-F1.
- **Point Estimate $\Delta_{\text{QWK}}$:** $-0.0057$
  - **95% Paired Bootstrap Confidence Interval:** $[-0.0269, +0.0154]$
  - *Status:* **Confidence interval comfortably spans zero.** No detectable improvement exists.

**Conclusion:** When evaluated using rigorous, leak-free out-of-fold pooling and paired bootstrap resampling, late fusion provides **no statistically significant benefit** over the classical baseline. The earlier observed $+0.0146$ improvement was an artifact of single-split test set variance.

---

## 5. Label Noise & Clinical Target Analysis

### 5.1 Like-for-Like Label Noise ($N = 374$)
The CKD-EPI formula requires patient biological sex ($\kappa = 0.7, \alpha = -0.329$ for females; $\kappa = 0.9, \alpha = -0.411$ for males). Because biological sex is missing in the UCI CKD repository, previous phases assumed male sex for all patients.

Evaluating the impact of this assumption across all $N = 374$ patients:
- **125 of 374 patients (33.4%) change category** between male and female CKD-EPI definitions.
- **Male-Trained Model Evaluated on Male Target:** Accuracy = **0.5401**
- **Male-Trained Model Evaluated on Midpoint Target:** Accuracy = **0.5187**, QWK = **0.7931**
- **Midpoint-Trained Model Evaluated on Midpoint Target:** Accuracy = **0.4973**

### 5.2 Invalidity of Subsetting to $N = 249$
Earlier experiments suggested isolating an "unambiguous" subset ($N = 249$) where male and female CKD-EPI formulas assign identical categories.

**Methodological Critique:**
Filtering on label agreement discards patients whose serum creatinine is close to the diagnostic thresholds ($\sim 0.7 - 1.2$ mg/dL). Because creatinine is the mathematical generator of the label, this filtering eliminates borderline, clinically ambiguous cases and retains only extreme, well-separated cases (e.g. clearly healthy or severely elevated creatinine). Comparing metrics on $N=249$ to $N=374$ is an **uncontrolled selection bias** that creates an artificially simple task, not a more robust model.

### 5.3 Clinical Defense: 3-Class vs 5-Class Target
The 3-class target definition (Early G1–2, Moderate G3, Severe G4–5) is clinically superior for non-nephrologist screening:
1. **Clinical Actionability:** Patients in Stage 1 and 2 require identical primary-care management (lifestyle modification, blood pressure control, monitoring). Distinguishing G1 from G2 adds no therapeutic value.
2. **Specialist Referral Threshold:** Stage 3 ($30-59$ mL/min) represents the critical window for nephrology referral and medication dose adjustments (e.g., SGLT2 inhibitors, ACEi/ARBs).
3. **Statistical Sample Size:** In 5-class labeling, Stage 1 has only 35 patients and Stage 4 has only 29 patients in the entire dataset. In 3-class labeling, all classes have adequate sample sizes ($n = 188, 84, 102$), boosting Macro-F1 from **0.54** to **0.77** and QWK to **0.86**.

---

## 6. Upper-Bound Ceiling Sanity Check

To establish the theoretical ceiling for tabular machine learning models on this dataset, an experiment was conducted where serum creatinine (`sc`) and `age` were **included as inputs** (formula inversion task):

| Metric | Primary Leak-Free Model (LR ElasticNet) | Upper-Bound Model (`sc` + `age` included) |
| :--- | :---: | :---: |
| **Quadratic Weighted Kappa** | 0.8029 | **0.8802** |
| **Macro-F1** | 0.5414 | **0.6612** |
| **Accuracy** | 0.5401 | **0.6526** |

### Why Doesn't the Upper Bound Reach 1.00?
Even with the exact mathematical inputs provided, linear models do not achieve 100% accuracy because:
1. **Nonlinear Exponent Breaks:** CKD-EPI is a piecewise exponential power function ($s^{-0.411}$ for $s \le 0.9$ vs $s^{-1.209}$ for $s > 0.9$);
2. **Discretization Boundary Noise:** Small continuous measurement noise around 60.0 or 90.0 mL/min flips the discrete stage label;
3. **Sample Size ($N = 374$):** Limited sample density near stage boundaries.

This confirms that the primary leakage-free models (QWK $\approx 0.81 - 0.82$) achieve approximately **93% of the theoretical ceiling** attainable by any linear tabular estimator on this dataset.

---

## 7. Retraction & Correction of Prior Claims

| Prior Claim | Corrected Reality | Evidence |
| :--- | :--- | :--- |
| *"Tree-based models fail due to small tabular size and validation collapse."* | **Invalid.** Tree models (RF, HistGradientBoosting) perform competitively, reaching QWK = **0.812** and **0.815**, with within $\pm 1$ accuracy of **93.3%**. Macro-F1 appeared lower simply because trees do not naturally calibrate ordinal boundary penalties without sample reweighting. | Per-class recall in RF shows 34% to 79% sensitivity across all classes; no collapse occurred. |
| *"Phase 7 Late Fusion achieved +0.0146 quantum advantage."* | **Unsubstantiated.** The $+0.0146$ delta was within sample noise of a single 75-sample test set. Paired bootstrap over all 374 OOF predictions shows $\Delta_{\text{Macro-F1}} = -0.0481$ (95% CI: $[-0.0918, -0.0046]$) and $\Delta_{\text{QWK}} = -0.0057$ (95% CI: $[-0.027, +0.015]$). | The 95% confidence interval spans or is below zero; no quantum advantage exists. |
| *"Baseline LR achieves only 0.305 CV Macro-F1."* | **Invalid software defect.** Phase 5 scaled data twice in-place, shifting `sg` to $-279.5$. True balanced LR achieves **0.5419** nested CV Macro-F1. | Fixed in unified pipeline; validated on 239, 299, and 374 sample cohorts. |
| *"VQC test Macro-F1 is 0.1300."* | **Artifact of tie-breaking.** Phase 6 (0.1130) and Phase 7 (0.1300) have identical parameters and 0% recall on categories 3–5. The difference was bitstring tie-breaking. | Exact code trace in `src/quantum/01_vqc_classifier.py` and `src/fusion/01_late_fusion.py`. |

---

## 8. Clinical Limitations & Future Directions

1. **Uncertainty from Missing Biological Sex:** Without patient sex, all eGFR values carry a $\pm 15\%$ systematic shift, flipping categories for 33.4% of patients. Clinical translation requires datasets with explicitly recorded biological sex.
2. **Cross-Sectional Surrogate vs True KDIGO Staging:** True clinical KDIGO CKD diagnosis requires persistent eGFR depression for $> 3$ months plus quantitative albuminuria (ACR). The UCI dataset represents a single cross-sectional snapshot; categories must be interpreted as acute eGFR surrogates rather than confirmed chronic staging.
3. **Small Sample Size for NISQ Quantum Computing:** Current state-of-the-art quantum kernels (4–8 qubits) simulated classically suffer from trainability barriers and cannot outperform tuned classical regularization (L1/L2) on small, noisy medical tabular data.
4. **Superiority of Ordinal Regressors:** Treating eGFR category prediction as a multi-class classification problem is suboptimal. Continuous log-eGFR regression with post-hoc binning achieves the highest QWK (0.856) and lowest MAE (0.196), mirroring how clinical decision thresholds should be applied to underlying biological continuous biomarkers.

---

*Report compiled autonomously following strict scientific integrity protocols. Raw fold-level records and bootstrap distributions are permanently archived in `results/final_nested_cv.json`.*
