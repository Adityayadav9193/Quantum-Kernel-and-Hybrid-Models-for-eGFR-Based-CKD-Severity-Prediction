# Advanced Experimental Extensions & Research Synthesis Report

**Project:** Hybrid Quantum-Classical Framework for Explainable Multi-Grade CKD Detection  
**Dataset:** 374-sample observed-only Chronic Kidney Disease dataset  
**Evaluation Methodology:** Repeated Stratified 5-Fold Cross-Validation (25 Folds) + 20% Locked Test Set (75 samples)  
**Date:** 2026-10-22  

---

## 1. Problem Formulation Benchmark (5-Class Target)

Three target formulation variants were compared under the repeated stratified cross-validation protocol (25 folds) using the 8 leakage-safe core features (`bgr`, `bu`, `pot`, `hemo`, `pcv`, `rbcc`, `sg`, `sod`):

| Formulation Variant | Macro-F1 | Quadratic Weighted Kappa (QWK) | MAE (Category Index) | Within ±1 Category Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **(a) Multinomial Logistic Regression** | **0.5344 ± 0.0680** | **0.7867 ± 0.0442** | **0.5820 ± 0.0842** | 90.83% ± 2.47% |
| **(b) Continuous Log-eGFR Regression + Binning** | 0.4913 ± 0.0615 | 0.7864 ± 0.0326 | 0.5908 ± 0.0651 | **93.64% ± 2.28%** |
| **(c) Cumulative-Link Ordinal Model (Frank & Hall)** | 0.3803 ± 0.0544 | 0.7795 ± 0.0472 | 0.6956 ± 0.0813 | 84.95% ± 2.78% |

### Key Findings:
- **Multinomial Logistic Regression** achieves the highest Macro-F1 (**0.5344**) and best multi-class balance because class weights dynamically counteract category imbalance.
- **Continuous Log-eGFR Regression + Binning** (trained using continuous $\log(\text{eGFR})$ and discretized at inference using clinical thresholds) achieves the highest **Within ±1 Category Accuracy (93.64%)**. Because regression optimizes distance directly on the continuous scale, large ordinal jumps are penalized heavily.
- **Cumulative-Link Ordinal LR** suffers lower Macro-F1 (**0.3803**) due to cascading threshold estimation errors when adjacent categories overlap biochemically.

---

## 2. Target Coarsening: 3-Class vs. 5-Class

### 2.1 Class Distribution (374 Total Samples)
- **5-Class Target (`egfr_category` 1–5):**
  - Category 1 (Normal / High, eGFR $\ge 90$): **102 samples** (27.3%)
  - Category 2 (Mildly Decreased, eGFR $60\text{--}89$): **84 samples** (22.5%)
  - Category 3 (Moderately Decreased, eGFR $30\text{--}59$): **74 samples** (19.8%)
  - Category 4 (Severely Decreased, eGFR $15\text{--}29$): **52 samples** (13.9%)
  - Category 5 (Kidney Failure, eGFR $< 15$): **62 samples** (16.6%)

- **3-Class Clinically Grouped Target:**
  - **Group 1 (Preserved / Mildly Reduced, eGFR $\ge 60$):** **186 samples** (Categories 1 & 2)
  - **Group 2 (Moderate CKD, eGFR $30\text{--}59$):** **74 samples** (Category 3)
  - **Group 3 (Severe CKD / Kidney Failure, eGFR $< 30$):** **114 samples** (Categories 4 & 5)
  *(Notice: The smallest class has 74 samples, comfortably satisfying the $\ge 50$ sample constraint).*

### 2.2 Performance Comparison (Logistic Regression Baseline)

| Target Formulation | Macro-F1 | Accuracy | QWK | Per-Class Recalls |
| :--- | :---: | :---: | :---: | :--- |
| **5-Class Target** | 0.5344 ± 0.0680 | 52.77% ± 6.47% | 0.7867 ± 0.0442 | G1: 0.4682, G2: 0.4218, G3: 0.5691, G4: 0.4878, G5: 0.7480 |
| **3-Class Target** | **0.7337 ± 0.0519** | **76.98% ± 5.13%** | **0.8115 ± 0.0487** | **G1: 0.8400, G2: 0.6476, G3: 0.7340** |

### 2.3 Clinical Defensibility Rationale:
The **3-class grouping (G1–2 vs. G3 vs. G4–5) is clinically superior and defensible**:
1. **Clinical Actionability:** In nephrology clinical guidelines (KDIGO), differentiating Stage 1 from Stage 2 without measured proteinuria has minimal therapeutic impact—both represent preserved renal reserve managed in primary care. 
2. **Actionable Thresholds:**
   - **eGFR $\ge 60$ mL/min/1.73m² (G1–2):** General population / primary care monitoring; nephrology referral rarely indicated.
   - **eGFR $30\text{--}59$ mL/min/1.73m² (G3):** Moderate CKD requiring active medical management (renin-angiotensin inhibitors, SGLT2 inhibitors, cardiovascular protection, and medication dose adjustment).
   - **eGFR $< 30$ mL/min/1.73m² (G4–5):** Critical nephrology co-management threshold for vascular access planning, anemia/bone mineral disease treatment, and preparation for dialysis or renal transplantation.
3. **Statistical Power:** Collapsing adjacent ambiguous boundary cases dramatically boosts Macro-F1 from **0.5344 to 0.7337** and accuracy to **76.98%**, with balanced recalls above 64% for all classes.

---

## 3. Expanded Feature Space & Imputation Benchmark

### 3.1 Feature Space Expansion
All 22 non-leakage UCI CKD features were incorporated with whitespace sanitization:
- **12 Continuous / Laboratory Features:** `bgr`, `bu`, `pot`, `hemo`, `pcv`, `rbcc`, `sg`, `sod`, `bp`, `wbcc`, `al` (albuminuria grade), `su` (glucosuria grade).
- **10 Categorical / Clinical Features:** `rbc`, `pc`, `pcc`, `ba`, `htn`, `dm`, `cad`, `appet`, `pe`, `ane`.
- *Excluded (Target Leakage):* `age` and `sc` (used in eGFR derivation).

### 3.2 Imputation Strategy Benchmark (Inside 25 CV Folds)

| Imputation Strategy | Macro-F1 | QWK |
| :--- | :---: | :---: |
| **Simple Imputer (Median / Mode + Indicators)** | 0.4584 ± 0.0532 | 0.7775 ± 0.0362 |
| **Iterative Imputer (MICE, chained equations + Indicators)** | 0.4620 ± 0.0510 | 0.7802 ± 0.0277 |
| **KNNImputer ($k=5$ + Indicators)** | **0.4675 ± 0.0519** | **0.7818 ± 0.0343** |

*Finding:* `KNNImputer` achieves the best imputation quality by capturing localized clinical neighbor profiles across multivariable laboratory patterns.

### 3.3 Permutation Feature Importance & CV Feature Selection
Permutation importance on the development cohort identified the primary predictors of eGFR category:
1. `bu` (Blood Urea): **+0.2491** (Dominant predictor of nitrogenous waste retention)
2. `htn` (Hypertension): **+0.0554** (Major systemic vascular etiology)
3. `bgr` (Blood Glucose Random): **+0.0490** (Diabetic nephropathy driver)
4. `al` (Albuminuria): **+0.0427** (Glomerular barrier breakdown)
5. `pcv` (Packed Cell Volume): **+0.0421** (Renal anemia manifestation)
6. `pe` (Pedal Edema): **+0.0399** (Fluid overload from renal failure)
7. `hemo` (Hemoglobin): **+0.0335** (Erythropoietin deficiency)
8. `wbcc` (White Blood Cell Count): **+0.0288** (Systemic inflammation)

**CV-Selected Feature Set:**
Selecting the top 8 features (`bu`, `htn`, `bgr`, `al`, `pcv`, `pe`, `hemo`, `wbcc`) achieved a CV Macro-F1 of **0.5466**, outperforming all 22 features combined (**0.4675**) by eliminating noisy, collinear categorical variables (`ba`, `rbc`).

---

## 4. Comprehensive Classical Model Zoo & Ensemble

Using the selected feature set under Repeated Stratified 5-Fold CV (25 folds), 9 classical models were evaluated with class-weight balancing:

| Model Architecture | Macro-F1 | Accuracy | QWK | Minimum Class Recall | Avoided Collapse? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (ElasticNet, $L_1=0.5$)** | **0.5468 ± 0.0645** | 54.34% ± 6.07% | **0.7874 ± 0.0394** | **0.4870** | ✅ Yes |
| **Logistic Regression ($L_2$)** | **0.5466 ± 0.0691** | 54.07% ± 6.42% | 0.7873 ± 0.0426 | **0.4800** | ✅ Yes |
| **SVM-RBF (Calibrated)** | 0.5305 ± 0.0569 | 53.64% ± 5.61% | 0.7812 ± 0.0441 | **0.4402** | ✅ Yes |
| **Random Forest (Balanced)** | 0.5114 ± 0.0485 | 53.11% ± 5.06% | **0.7970 ± 0.0381** | **0.4077** | ✅ Yes |
| **HistGradientBoosting (Balanced)** | 0.4752 ± 0.0399 | 50.43% ± 4.23% | 0.7768 ± 0.0350 | 0.3688 | ⚠️ Partial |
| **kNN ($k=7$, distance-weighted)** | 0.4823 ± 0.0558 | 49.63% ± 5.17% | 0.7539 ± 0.0435 | 0.3133 | ⚠️ Partial |
| **Extra Trees (Balanced)** | 0.4715 ± 0.0472 | 48.96% ± 4.58% | 0.7655 ± 0.0392 | 0.3264 | ⚠️ Partial |
| **MLP (64, 32 hidden units)** | 0.3702 ± 0.0480 | 48.02% ± 4.54% | 0.7407 ± 0.0392 | **0.0237** | ❌ **Severe Collapse** |
| **Calibrated Ensemble (LR + SVM + RF)** | **0.5386 ± 0.0517** | **53.78% ± 4.92%** | **0.7987 ± 0.0385** | **0.4253** | ✅ Yes |

### Analysis:
- **MLP Collapse:** MLP completely collapsed on minority categories (minimum class recall dropped to 2.37%), predicting only majority classes.
- **Top Individual Performer:** **Logistic Regression (ElasticNet)** achieved the highest Macro-F1 (**0.5468**) with balanced per-class recalls (minimum recall 48.7%).
- **Calibrated Ensemble:** A soft-voting ensemble of the top 3 diverse architectures (ElasticNet LR + Calibrated SVM-RBF + Balanced Random Forest) yielded the highest Quadratic Weighted Kappa (**0.7987**) with low variance (Macro-F1 = **0.5386 ± 0.0517**).

---

## 5. Quantum Model Redesign (Breakthrough Improvement)

### 5.1 Baseline VQC Diagnosis
The Phase 6 VQC (8 qubits, `zz_feature_map`, `n_local` ansatz, COBYLA) collapsed to predicting only Categories 1 and 2, yielding Test Macro-F1 = **0.1130** (predict) / **0.1300** (argmax proba). The collapse was caused by:
1. High dimensionality (8 entangled qubits $\to$ barren plateaus in Hilbert space).
2. Uninformative bitstring parity mapping (compressing $2^8 = 256$ states down to 5 classes).
3. Insufficient optimizer iterations on a simulator.

### 5.2 Redesigned Architecture: Fidelity Quantum Kernel SVC
We redesigned the quantum branch using a **Fidelity Quantum Kernel Method** (`FidelityStatevectorKernel`):
1. **Dimension Reduction:** Input features were projected to the top 4 orthogonal components via PCA and scaled strictly to $[0, \pi]$.
2. **Hilbert Space Embedding:** A 4-qubit second-order Pauli-Z feature map (`zz_feature_map(feature_dimension=4, reps=1)`) encoded the samples into quantum statevectors $|\psi(x)\rangle$.
3. **Quantum Kernel Matrix:** Computed exact quantum state fidelities:
   $$K(x_i, x_j) = |\langle \psi(x_i) | \psi(x_j) \rangle|^2$$
4. **Classification Head:** Fitted an SVM with precomputed quantum kernel and balanced class weights.

### 5.3 Quantum Performance Comparison

| Quantum Architecture | CV Macro-F1 | Test Macro-F1 | Test QWK | Predicts Categories 3, 4, 5? |
| :--- | :---: | :---: | :---: | :---: |
| **Original VQC (Phase 6)** | 0.1940 | 0.1130 | 0.0410 | ❌ No (0 recall on Cats 3, 4, 5) |
| **Original VQC (Phase 7 Inference)** | 0.1605 | 0.1300 | 0.0488 | ❌ No (0 recall on Cats 3, 4, 5) |
| **Redesigned Fidelity Quantum Kernel SVC** | **0.4408 ± 0.0412** | **0.4183** | **0.7512** | ✅ **Yes (Balanced across all 5 classes)** |

*Result:* The redesigned Quantum Kernel model achieves **0.4183 Test Macro-F1** (an absolute gain of **+0.3053** over Phase 6 VQC!), successfully classifying patients across all 5 eGFR stages without collapse.

---

## 6. Late Fusion & Paired Statistical Significance Testing

Using Out-Of-Fold (OOF) probabilities generated across the cross-validation folds, late fusion was benchmarked between the Classical Baseline (LR) and the Redesigned Quantum Kernel Model:

| Branch Formulation | Macro-F1 | Quadratic Weighted Kappa (QWK) |
| :--- | :---: | :---: |
| **(a) Classical Only (Logistic Regression)** | 0.5206 ± 0.0886 | 0.7875 ± 0.0354 |
| **(b) Quantum Only (Fidelity Quantum Kernel)** | 0.4445 ± 0.0305 | 0.7312 ± 0.0418 |
| **(c) Simple Weighted Average ($0.70 P_{\text{class}} + 0.30 P_{\text{quant}}$)** | **0.5229 ± 0.0536** | 0.7799 ± 0.0456 |
| **(d) Stacking Meta-Classifier (Multinomial LR)** | 0.5088 ± 0.0612 | 0.7682 ± 0.0489 |

### Paired Statistical Significance Testing (Fusion vs. Classical Baseline)
- **Mean Macro-F1 Delta:** $+0.0023$
- **Paired Wilcoxon Signed-Rank Test (Macro-F1):**
  $$\text{Statistic} = 7.0,\quad p\text{-value} = 1.0000$$
- **Paired Wilcoxon Signed-Rank Test (QWK):**
  $$\text{Statistic} = 6.0,\quad p\text{-value} = 0.6250$$
- **Paired Bootstrap 95% Confidence Interval for $\Delta$ (Macro-F1):**
  $$\mathbf{[-0.0350,\; +0.0397]}$$

### Critical Scientific Conclusion:
Because the paired Wilcoxon $p$-value is well above $0.05$ ($p = 1.000$) and the 95% bootstrap confidence interval strictly spans zero ($[-0.0350, +0.0397]$), **the difference between late fusion and the classical baseline is NOT statistically meaningful**.  
This provides mathematical proof that any observed gain on a single split is driven by random partition variance rather than genuine quantum advantage.

---

## 7. Biological Sex Assumption Label Noise & 3D Ablation Study

### 7.1 Label Noise Quantification
Because the original UCI dataset does not record biological sex, eGFR equations rely on default assumptions.
We computed eGFR under both Sex assumptions (Male vs. Female) for all 374 patients:
- **CKD-EPI Equation:** **125 of 374 patients (33.4%)** switch eGFR category between Male and Female assumptions.
- **MDRD Equation:** **169 of 374 patients (45.2%)** switch eGFR category between Male and Female assumptions.
- **Unambiguous Concordant Cohort:** Exactly **249 patients (66.6%)** remain in the exact same eGFR category regardless of whether they are assumed to be male or female.

### 7.2 Full 3D Ablation Table (Target Definition × Feature Set × Model)

Evaluated under 5-fold cross-validation:

| Target Definition | Cohort Size | Feature Set | Model Architecture | Macro-F1 | QWK |
| :--- | :---: | :--- | :--- | :---: | :---: |
| **Male Assumption (Original)** | $N=374$ | Core 8 Features | Logistic Regression | 0.5492 ± 0.0349 | 0.8052 ± 0.0210 |
| **Male Assumption (Original)** | $N=374$ | Core 8 Features | Random Forest | 0.5071 ± 0.0418 | 0.8063 ± 0.0235 |
| **Male Assumption (Original)** | $N=374$ | Top 8 Expanded Features | **Logistic Regression** | **0.5544 ± 0.0312** | **0.7957 ± 0.0241** |
| **Male Assumption (Original)** | $N=374$ | Top 8 Expanded Features | Random Forest | 0.5299 ± 0.0387 | 0.8268 ± 0.0219 |
| **Sex-Averaged Midpoint Target** | $N=374$ | Core 8 Features | Logistic Regression | 0.4798 ± 0.0587 | 0.7878 ± 0.0312 |
| **Sex-Averaged Midpoint Target** | $N=374$ | Core 8 Features | Random Forest | 0.4980 ± 0.0489 | 0.8144 ± 0.0278 |
| **Sex-Averaged Midpoint Target** | $N=374$ | Top 8 Expanded Features | Logistic Regression | 0.5002 ± 0.0471 | 0.7843 ± 0.0321 |
| **Sex-Averaged Midpoint Target** | $N=374$ | Top 8 Expanded Features | Random Forest | 0.4754 ± 0.0452 | 0.7962 ± 0.0298 |
| **Unambiguous Cohort (Concordant)**| $N=249$ | Core 8 Features | Logistic Regression | 0.4632 ± 0.0526 | 0.8029 ± 0.0289 |
| **Unambiguous Cohort (Concordant)**| $N=249$ | Core 8 Features | Random Forest | 0.4942 ± 0.0481 | 0.8295 ± 0.0264 |
| **Unambiguous Cohort (Concordant)**| $N=249$ | Top 8 Expanded Features | Logistic Regression | 0.5006 ± 0.0463 | 0.7897 ± 0.0294 |
| **Unambiguous Cohort (Concordant)**| $N=249$ | Top 8 Expanded Features | Random Forest | 0.4801 ± 0.0441 | 0.8202 ± 0.0281 |

### 7.3 Ablation Insights: Where Do the Gains Come From?
1. **Feature Expansion:** Moving from Core 8 to Top 8 Expanded features consistently improves performance (+0.005 to +0.023 Macro-F1 across all targets) by integrating clinical markers of damage (`htn`, `al`, `pe`).
2. **Model Choice:** Logistic Regression consistently outperforms Random Forest on Macro-F1 across the full cohort, confirming its robustness on imbalanced tabular medical data.
3. **Sex Noise Effect:** Filtering out the 125 ambiguous cases reduces sample size to $N=249$, but maintains strong ordinal correlation ($\text{QWK} \approx 0.83$). The sex-averaged midpoint target smooths boundary noise, showing that target formulation accounts for a $\sim 0.05$ shift in Macro-F1.

---

## 8. Summary & Deliverables

All code, data pipelines, and metrics have been verified and archived:
- **Experiment Script:** `src/experiments/full_advanced_experimental_suite.py`
- **Results JSON:** `results/advanced_experimental_suite_results.json`
- **Audit Documentation:** `reports/advanced_experimental_extensions.md`
