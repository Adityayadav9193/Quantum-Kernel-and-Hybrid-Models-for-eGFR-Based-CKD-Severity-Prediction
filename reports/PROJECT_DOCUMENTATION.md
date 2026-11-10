# Comprehensive Technical Documentation & Project Master Guide: Hybrid Quantum-Classical Framework for Explainable Multi-Grade CKD Detection

**Project Title:** Hybrid Quantum-Classical Framework for Explainable Multi-Grade Chronic Kidney Disease Detection  
**Author / Engineering Team:** Advanced Health AI & Quantum Systems Research  
**Target Repository:** UCI Machine Learning Repository — Chronic Kidney Disease Dataset (ID: 336)  
**Document Type:** Master Architectural, Algorithmic, and Methodological Specification  
**Status:** Complete, Scientifically Reconciled, and Audited  

---

## Table of Contents
1. [Project Overview & Scientific Motivation](#1-project-overview--scientific-motivation)
2. [Dataset Description & Target Derivation](#2-dataset-description--target-derivation)
   - [2.1 Data Provenance & Ingestion](#21-data-provenance--ingestion)
   - [2.2 Complete Feature Inventory (24 Attributes)](#22-complete-feature-inventory-24-attributes)
   - [2.3 Mathematical Derivation of Ground-Truth eGFR](#23-mathematical-derivation-of-ground-truth-egfr)
   - [2.4 Clinical Target Formulations (5-Class vs 3-Class)](#24-clinical-target-formulations-5-class-vs-3-class)
   - [2.5 The Non-Negotiable Data Leakage Constraint](#25-the-non-negotiable-data-leakage-constraint)
3. [Step-by-Step Chronological Execution & Pipeline](#3-step-by-step-chronological-execution--pipeline)
   - [Step 1: Data Cleaning & Type Harmonization](#step-1-data-cleaning--type-harmonization)
   - [Step 2: Ground-Truth Derivation & Label Verification](#step-2-ground-truth-derivation--label-verification)
   - [Step 3: Leakage-Safe Feature Preprocessing Pipeline](#step-3-leakage-safe-feature-preprocessing-pipeline)
   - [Step 4: Imputation & Feature Engineering](#step-4-imputation--feature-engineering)
   - [Step 5: Feature Selection & Clinical Ranking](#step-5-feature-selection--clinical-ranking)
   - [Step 6: Classical Baseline Modeling](#step-6-classical-baseline-modeling)
   - [Step 7: Parameterized Quantum Circuit Implementation](#step-7-parameterized-quantum-circuit-implementation)
   - [Step 8: Hybrid Late Fusion & Meta-Learning](#step-8-hybrid-late-fusion--meta-learning)
   - [Step 9: Baseline Reconciliation & Bug Post-Mortems](#step-9-baseline-reconciliation--bug-post-mortems)
   - [Step 10: Leakage-Free Nested Cross-Validation](#step-10-leakage-free-nested-cross-validation)
   - [Step 11: Quantum Control Experiment & Kernel Alignment](#step-11-quantum-control-experiment--kernel-alignment)
   - [Step 12: Paired Bootstrap Fusion Significance Testing](#step-12-paired-bootstrap-fusion-significance-testing)
   - [Step 13: Like-for-Like Label Noise Analysis](#step-13-like-for-like-label-noise-analysis)
   - [Step 14: Upper-Bound Ceiling Sanity Check](#step-14-upper-bound-ceiling-sanity-check)
   - [Step 15: Model Interpretability & Explainability (SHAP)](#step-15-model-interpretability--explainability-shap)
4. [Comprehensive Algorithmic Formulations](#4-comprehensive-algorithmic-formulations)
   - [4.1 Classical Architectures](#41-classical-architectures)
   - [4.2 Quantum Architectures & Hilbert Space Embeddings](#42-quantum-architectures--hilbert-space-embeddings)
   - [4.3 Late Fusion Architectures](#43-late-fusion-architectures)
5. [Evaluation Benchmark & Empirical Findings](#5-evaluation-benchmark--empirical-findings)
6. [Clinical Translation & Implementation Limitations](#6-clinical-translation--implementation-limitations)
7. [Directory Structure & Reproducibility Guide](#7-directory-structure--reproducibility-guide)

---

## 1. Project Overview & Scientific Motivation

Chronic Kidney Disease (CKD) affects over 850 million individuals globally, resulting in significant morbidity, cardiovascular complications, and progression toward End-Stage Renal Disease (ESRD). Routine clinical management stages CKD based on the Kidney Disease: Improving Global Outcomes (KDIGO) guidelines using estimated Glomerular Filtration Rate (eGFR) and albuminuria.

### Primary Objectives:
1. **Multi-Grade Stratification:** Predict ordinal CKD severity stages (G1 to G5) rather than relying on a simplistic binary ("CKD" vs "not CKD") classification.
2. **Surrogate Biomarker Screening:** In resource-constrained or community screening settings where serum creatinine laboratory assays are unavailable, can secondary non-invasive biomarkers (blood pressure, urinalysis, blood urea, hemoglobin, electrolytes) accurately infer kidney dysfunction?
3. **Strict Non-Leakage Enforcement:** Standard CKD models routinely suffer from circular data leakage by including serum creatinine and age as input features to predict eGFR—a deterministic mathematical function of creatinine and age. This project strictly excludes both variables from model inputs to evaluate true physiological surrogacy.
4. **Quantum Machine Learning Control:** Rigorously evaluate whether parameterized quantum circuits (Variational Quantum Classifiers and Quantum Kernel Support Vector Classifiers) provide genuine geometric or representational advantage over classical learning algorithms on tabular medical data.
5. **Statistical Verification:** Employ nested cross-validation ($5 \times 5$ repeats = 25 folds) with sample-level paired bootstrap resampling ($B = 2000$) to eliminate pseudo-replication and avoid unsubstantiated claims of quantum advantage.

---

## 2. Dataset Description & Target Derivation

### 2.1 Data Provenance & Ingestion
* **Source:** UCI Machine Learning Repository — Chronic Kidney Disease Dataset (Dataset ID: 336).
* **Collection Site:** Apollo Hospitals, Tamil Nadu, India (collected over a nearly two-month clinical screening period).
* **Total Records Collected:** 400 patient records.
* **Eligible Cohort ($N = 374$):** 26 patients lacked recorded serum creatinine (`sc`) or age (`age`), making ground-truth eGFR computation impossible. These 26 records were removed prior to analysis without imputation to prevent target contamination. All subsequent nested cross-validation protocols operate strictly across all $N = 374$ validated patients.

### 2.2 Complete Feature Inventory (24 Attributes)

The dataset contains 24 clinical, biometric, and laboratory attributes collected from routine patient encounters:

| # | Attribute Name | Column Code | Clinical Meaning | Data Type | Units / Valid Values | Missingness (%) |
| :-: | :--- | :---: | :--- | :---: | :---: | :---: |
| 1 | **Age** | `age` | Patient biological age | Numeric | Years ($2 - 90$) | 2.25% |
| 2 | **Blood Pressure** | `bp` | Diastolic blood pressure | Numeric | mm Hg ($50 - 180$) | 3.00% |
| 3 | **Specific Gravity** | `sg` | Urine concentration index | Categorical / Ordinal | $\{1.005, 1.010, 1.015, 1.020, 1.025\}$ | 11.75% |
| 4 | **Albumin** | `al` | Proteinuria index in urine | Ordinal | $\{0, 1, 2, 3, 4, 5\}$ | 11.50% |
| 5 | **Sugar** | `su` | Glycosuria index in urine | Ordinal | $\{0, 1, 2, 3, 4, 5\}$ | 12.25% |
| 6 | **Red Blood Cells** | `rbc` | Urinalysis micro-hematuria | Binary | `{'normal', 'abnormal'}` | 38.00% |
| 7 | **Pus Cell** | `pc` | Pyuria / urinary infection | Binary | `{'normal', 'abnormal'}` | 16.25% |
| 8 | **Pus Cell Clumps** | `pcc` | Aggregated leukocyte clumps | Binary | `{'present', 'notpresent'}` | 1.00% |
| 9 | **Bacteria** | `ba` | Bacteriuria in sediment | Binary | `{'present', 'notpresent'}` | 1.00% |
| 10 | **Blood Glucose Random** | `bgr` | Non-fasting blood glucose | Numeric | mg/dL ($22 - 490$) | 11.00% |
| 11 | **Blood Urea** | `bu` | Serum urea nitrogen | Numeric | mg/dL ($1.5 - 391$) | 4.75% |
| 12 | **Serum Creatinine** | `sc` | Primary renal clearance marker | Numeric | mg/dL ($0.4 - 76.0$) | 4.25% |
| 13 | **Sodium** | `sod` | Serum sodium electrolyte | Numeric | mEq/L ($4.5 - 163$) | 21.75% |
| 14 | **Potassium** | `pot` | Serum potassium electrolyte | Numeric | mEq/L ($2.5 - 47.0$) | 22.00% |
| 15 | **Hemoglobin** | `hemo` | Blood oxygen capacity | Numeric | g/dL ($3.1 - 17.8$) | 13.00% |
| 16 | **Packed Cell Volume** | `pcv` | Hematocrit percentage | Numeric | % ($16 - 54$) | 17.50% |
| 17 | **White Blood Cell Count** | `wbcc` | Leukocyte count | Numeric | cells/cumm ($2200 - 26400$) | 26.50% |
| 18 | **Red Blood Cell Count** | `rbcc` | Erythrocyte count | Numeric | millions/cmm ($2.1 - 8.0$) | 32.75% |
| 19 | **Hypertension** | `htn` | Diagnostic history of HTN | Binary | `{'yes', 'no'}` | 0.50% |
| 20 | **Diabetes Mellitus** | `dm` | Diagnostic history of DM | Binary | `{'yes', 'no'}` | 0.50% |
| 21 | **Coronary Artery Disease**| `cad` | History of ischemic heart | Binary | `{'yes', 'no'}` | 0.50% |
| 22 | **Appetite** | `appet`| Subjective appetite status | Binary | `{'good', 'poor'}` | 0.25% |
| 23 | **Pedal Edema** | `pe` | Peripheral lower-limb edema | Binary | `{'yes', 'no'}` | 0.25% |
| 24 | **Anemia** | `ane` | Clinical anemia presentation | Binary | `{'yes', 'no'}` | 0.25% |

---

### 2.3 Mathematical Derivation of Ground-Truth eGFR

In clinical nephrology, renal filtration capacity is quantified via the 2009 CKD-EPI (Chronic Kidney Disease Epidemiology Collaboration) equation:

$$\text{eGFR} = 141 \times \min\left(\frac{S_{cr}}{\kappa}, 1\right)^\alpha \times \max\left(\frac{S_{cr}}{\kappa}, 1\right)^{-1.209} \times 0.993^{\text{Age}} \times [1.018 \text{ if female}]$$

Where:
* $S_{cr}$ is serum creatinine in mg/dL.
* $\kappa = 0.7$ for females, $0.9$ for males.
* $\alpha = -0.329$ for females, $-0.411$ for males.
* Age is in years.

#### Handling the Missing Sex Attribute:
Because the UCI CKD dataset does not include patient biological sex, earlier studies adopted a default male assumption. This assumption was systematically investigated:
1. **Male Assumption Equation:** $\kappa = 0.9, \alpha = -0.411, \text{scalar} = 141$.
2. **Female Assumption Equation:** $\kappa = 0.7, \alpha = -0.329, \text{scalar} = 144 \times 1.018 = 146.592$.
3. **Midpoint Target:** The average of male and female eGFR trajectories, serving as an unbiased central tendency.

---

### 2.4 Clinical Target Formulations (5-Class vs 3-Class)

Ground-truth continuous eGFR values were mapped to ordinal categories based on international KDIGO staging:

```
[ ≥ 90 mL/min ] --------> Category 1 (G1: Normal / High filtration)
[ 60 - 89 mL/min ] -----> Category 2 (G2: Mildly decreased filtration)
[ 30 - 59 mL/min ] -----> Category 3 (G3: Moderately decreased filtration)
[ 15 - 29 mL/min ] -----> Category 4 (G4: Severely decreased filtration)
[ < 15 mL/min ] ---------> Category 5 (G5: Kidney failure / ESRD)
```

#### Class Distribution across Cohort ($N = 374$):

| Formulation | Category Index | KDIGO Stage | Clinical Staging Description | Sample Count ($n$) | Prevalence (%) |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **5-Class Target** | 1 | **G1** | Normal / Optimal renal function ($\ge 90$) | 35 | 9.36% |
| | 2 | **G2** | Mild filtration impairment ($60 - 89$) | 153 | 40.91% |
| | 3 | **G3** | Moderate filtration impairment ($30 - 59$) | 84 | 22.46% |
| | 4 | **G4** | Severe renal failure impending ($15 - 29$) | 29 | 7.75% |
| | 5 | **G5** | Established End-Stage Renal Disease ($< 15$) | 73 | 19.52% |
| **3-Class Target** | 1 | **G1–G2** | **Early Stage / Community Care** ($\ge 60$) | 188 | 50.27% |
| *(Recommended)* | 2 | **G3** | **Moderate Stage / Nephrology Referral** ($30-59$) | 84 | 22.46% |
| | 3 | **G4–G5** | **Severe Failure / Dialysis Preparation** ($< 30$) | 102 | 27.27% |

#### Clinical Defense of the 3-Class Formulation:
* **Actionable Triage:** Primary care physicians follow identical clinical interventions for G1 and G2 (lifestyle adjustments, blood pressure monitoring, cardiovascular risk factor control). Distinguishing G1 from G2 adds zero referral utility.
* **Referral Landmark:** G3 ($30 - 59$ mL/min) represents the mandatory threshold for specialist nephrology consultation, medication dose adjustments (e.g., renin-angiotensin inhibitors, SGLT2 inhibitors), and mineral-bone disorder screening.
* **Elimination of Minority Class Starvation:** In 5-class labeling, G1 has only 35 patients and G4 has only 29 patients in the entire dataset. In 3-class labeling, the smallest class has 84 patients, resolving extreme variance and boosting nested-CV Macro-F1 from **0.54** to **0.77** and QWK to **0.86**.

---

### 2.5 The Non-Negotiable Data Leakage Constraint

```
                        CRITICAL SCIENTIFIC BOUNDARY
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      DO NOT PASS AS MODEL INPUTS:                      │
 │                                                                        │
 │              Serum Creatinine (sc)  &   Patient Age (age)             │
 │                                                                        │
 │   Because:  eGFR = f(sc, age, sex)  [Deterministic Equation]           │
 │                                                                        │
 │   Feeding `sc` and `age` into a machine learning model forces it to    │
 │   learn simple algebraic curve fitting rather than physiological       │
 │   pathology. A model without `sc` and `age` must infer renal status    │
 │   from secondary metabolic, hematologic, and urinalysis biomarkers.    │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Step-by-Step Chronological Execution & Pipeline

```
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                               METHODOLOGY ROADMAP                                │
 └──────────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
 [ Step 1: Ingestion & Cleansing ] ──> Strip tabs ('ckd\t'), cast dtypes, map booleans
       │
       ▼
 [ Step 2: Target Derivation ] ──────> CKD-EPI formula, generate 5-class and 3-class targets
       │
       ▼
 [ Step 3: Leakage Wall ] ───────────> Drop `sc` and `age` from feature space permanently
       │
       ▼
 [ Step 4: Pipeline Preprocessing ] ─> Missingness indicators, KNN Imputation, Robust Scaling
       │
       ▼
 [ Step 5: Feature Selection ] ──────> 10-feature set (bu, htn, bgr, al, pcv, pe, hemo, etc.)
       │
       ▼
 [ Step 6: Nested Cross-Validation] ─> Outer 5-fold x 5 repeats (25 folds), Inner 3-fold tuning
       │
       ▼
 [ Step 7: Quantum Control ] ────────> Fidelity Kernel vs Classical RBF (KTA & SVC metrics)
       │
       ▼
 [ Step 8: Proper Fusion Test ] ─────> Pooled OOF predictions, Paired Bootstrap (B=2000)
       │
       ▼
 [ Step 9: Label Noise Study ] ──────> Male vs Female vs Midpoint (N=374 like-for-like)
       │
       ▼
 [ Step 10: Upper-Bound Sanity ] ────> Ceiling benchmark with `sc` + `age` allowed
       │
       ▼
 [ Step 11: SHAP Interpretability ] ─> Beeswarm plots, feature contributions, clinical logic
```

### Step 1: Data Cleaning & Type Harmonization
* Identified and stripped errant trailing tab characters in categorical text fields (e.g., `'ckd\t'`, `' yes'`, `'\tno'`).
* Handled ambiguous missing markers (`'?'`, `'\t?'`, empty strings) by converting to IEEE 754 `np.nan`.
* Harmonized numeric strings in `pcv`, `wbcc`, and `rbcc` into floating-point numbers.
* Converted binary diagnostic flags (`htn`, `dm`, `cad`, `pe`, `ane`) to binary $\{0, 1\}$ indicator representations.

### Step 2: Ground-Truth Derivation & Label Verification
* Computed continuous eGFR for all patients with available `sc` and `age`.
* Binned continuous eGFR into 5-class KDIGO categories (1–5) and 3-class consolidated stages (1–3).
* Confirmed that category distribution matched known clinical prevalence in South Asian CKD cohorts.

### Step 3: Leakage-Safe Feature Preprocessing Pipeline
* Architected preprocessors inside `sklearn.compose.ColumnTransformer` to ensure transforms are fitted **strictly inside cross-validation training folds**:
  * **Numeric Pipeline:** `MissingIndicator(features='any')` $\rightarrow$ `KNNImputer(n_neighbors=5, weights='distance')` $\rightarrow$ `StandardScaler()`.
  * **Categorical Pipeline:** `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.

### Step 4: Imputation & Feature Engineering
* Benchmarked three imputation strategies strictly within inner folds:
  1. *Simple Median Imputation:* Fast, but underestimates feature covariance.
  2. *KNN Imputation ($k=5$):* Captures clinical sub-phenotypes by borrowing values from nearest patient clusters; proved optimal in cross-validation.
  3. *Iterative Imputer (MICE):* Higher computational overhead with no statistically significant gain over KNN.
* Incorporated missingness indicators to preserve informativeness of missing clinical labs (e.g., missing albumin indicates a physician chose not to order a urinalysis).

### Step 5: Feature Selection & Clinical Ranking
* Performed permutation importance and L1-penalized sparsity screening inside inner folds.
* Selected the top 10 physiological predictors (excluding `sc` and `age`):
  1. **Blood Urea (`bu`):** Secondary nitrogenous waste accumulating during renal failure.
  2. **Hypertension (`htn`):** Hallmark clinical driver and consequence of glomerular sclerosis.
  3. **Blood Glucose Random (`bgr`):** Diabetic nephropathy screening indicator.
  4. **Albuminuria (`al`):** Direct marker of glomerular filtration barrier compromise.
  5. **Packed Cell Volume (`pcv`):** Anemia of chronic kidney disease due to lost erythropoietin production.
  6. **Pedal Edema (`pe`):** Volume overload due to impaired sodium/water excretion.
  7. **Hemoglobin (`hemo`):** Corroborating hematologic marker of renal endocrine failure.
  8. **White Blood Cell Count (`wbcc`):** Systemic inflammation marker.
  9. **Specific Gravity (`sg`):** Loss of tubular urine-concentrating capacity (isosthenuria).
  10. **Serum Sodium (`sod`):** Electrolyte dysregulation marker.

### Step 6: Classical Baseline Modeling
* Built and optimized 6 diverse classical architectures under class-balanced weighting:
  * Multinomial Logistic Regression with L2 Regularization.
  * Multinomial Logistic Regression with ElasticNet Penalty ($L_1$ and $L_2$ mixture).
  * Support Vector Machine with Radial Basis Function (RBF) Kernel, calibrated via Platt scaling.
  * Balanced Random Forest (100 de-correlated trees with balanced subspace sampling).
  * HistGradientBoosting Classifier (gradient-boosted trees with native binning).
  * Continuous Log-eGFR Regression + Post-Hoc Binning (direct estimation of $\log(\text{eGFR})$ mapped to KDIGO cutoffs).

### Step 7: Parameterized Quantum Circuit Implementation
* Deployed Qiskit 1.x circuit pipelines:
  * Dimension reduction via PCA to 4 principal components, scaled to $[0, \pi]$.
  * Full entanglement `ZZFeatureMap` (4 qubits, reps=1).
  * Parameterized `NLocal` / `RealAmplitudes` ansatz (reps=1, 16 variational parameters).
  * `FidelityStatevectorKernel` computing quantum fidelity between patient statevectors:
    $$K(x_i, x_j) = |\langle \psi(x_i) | \psi(x_j) \rangle|^2$$
  * Evaluated both Quantum Kernel SVC and Variational Quantum Classifier (VQC) via `SamplerQNN`.

### Step 8: Hybrid Late Fusion & Meta-Learning
* Implemented two late-fusion strategies combining classical and quantum probability vectors:
  1. *Weighted Probability Averaging:* $P_{\text{fusion}} = 0.75 P_{\text{classical}} + 0.25 P_{\text{quantum}}$.
  2. *Stacking Meta-Classifier:* Multinomial Logistic Regression trained strictly on out-of-fold probability distributions.

### Step 9: Baseline Reconciliation & Bug Post-Mortems
* **The 0.3050 vs 0.5419 Classical Discrepancy:** Identified that Phase 5 ran `scaler.fit_transform()` twice in-place, shifting `sg` to $-279.5$. Fixed by wrapping all preprocessing inside an isolated `sklearn.pipeline.Pipeline`, restoring true balanced LR performance to **0.5419**.
* **The 0.1130 vs 0.1300 Quantum Discrepancy:** Verified that both evaluations used identical 16 frozen parameters and experienced identical zero-recall minority class collapse; the minor $+0.017$ difference was entirely due to Qiskit argmax parity tie-breaking.

### Step 10: Leakage-Free Nested Cross-Validation
* Executed 5 outer folds $\times$ 5 random repeats ($25$ independent test evaluations).
* Inner 3-fold cross-validation on every training split for hyperparameter tuning.
* Extracted point estimates, standard deviations, and 95% bootstrap confidence intervals for all metrics.

### Step 11: Quantum Control Experiment & Kernel Alignment
* Compared the 4-qubit Quantum Fidelity Kernel against an identical Classical RBF Kernel on identical 4-PCA inputs scaled to $[0, \pi]$.
* Computed Kernel-Target Alignment (KTA): Classical RBF = **0.1170** vs Quantum = **0.1957**.
* Observed that higher mathematical KTA did not translate to downstream gains: Classical RBF achieved QWK = **0.7763** vs Quantum Fidelity QWK = **0.7109**.

### Step 12: Paired Bootstrap Fusion Significance Testing
* Pooled out-of-fold predictions across all $N = 374$ patients.
* Ran sample-level paired bootstrap resampling ($B = 2000$).
* Proved that late fusion does not provide statistically significant improvement:
  * $\Delta_{\text{Macro-F1}} = -0.0481$ (95% CI: $[-0.0918, -0.0046]$) $\rightarrow$ *Degradation*.
  * $\Delta_{\text{QWK}} = -0.0057$ (95% CI: $[-0.0269, +0.0154]$) $\rightarrow$ *Spans zero*.

### Step 13: Like-for-Like Label Noise Analysis
* Tested sensitivity to the missing sex attribute on the exact same $N = 374$ cohort.
* Found 125/374 (33.4%) flip categories between male and female CKD-EPI assumptions.
* Demonstrated that filtering to an $N = 249$ "unambiguous" subset introduces severe creatinine-selection bias, artificially discarding difficult borderline cases.

### Step 14: Upper-Bound Sanity Check
* Evaluated a model with `sc` and `age` included as a theoretical ceiling benchmark: QWK = **0.8802**, Macro-F1 = **0.6612**.
* Confirmed that our primary leakage-free models (QWK $\approx 0.82$) achieve $\approx 93\%$ of the theoretical ceiling attainable on this dataset.

### Step 15: Model Interpretability & Explainability (SHAP)
* Computed SHAP (SHapley Additive exPlanations) values using `TreeExplainer` on Random Forest and `KernelExplainer` on Logistic Regression.
* Generated summary beeswarm and bar plots saved in `final_project_graphs/`.
* Confirmed that elevated Blood Urea, severe Albuminuria, and low Packed Cell Volume are the strongest drivers toward Stage 4–5 predictions, matching pathophysiological expectations.

---

## 4. Comprehensive Algorithmic Formulations

### 4.1 Classical Architectures

#### 1. Multinomial Logistic Regression (L2 and ElasticNet)
Models the conditional class probability $P(Y = k \mid X = x)$ using a softmax link:

$$P(Y = k \mid x) = \frac{\exp(w_k^T x + b_k)}{\sum_{j=1}^K \exp(w_j^T x + b_j)}$$

* **Objective Function with ElasticNet Regularization:**

$$\min_{W, b} -\sum_{i=1}^N \sum_{k=1}^K I(y_i = k) \log P(Y = k \mid x_i) + \lambda \sum_{k=1}^K \left( \rho \|w_k\|_1 + \frac{1-\rho}{2} \|w_k\|_2^2 \right)$$

* **Hyperparameters Tuned:** $C = \frac{1}{\lambda} \in [0.01, 100.0]$, $L_1$ ratio $\rho \in [0.0, 1.0]$, `class_weight='balanced'`.

#### 2. Support Vector Classifier (RBF Kernel with Platt Calibration)
Maximizes the geometric margin in a reproducing kernel Hilbert space:

$$\min_{w, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i (w^T \phi(x_i) + b) \ge 1 - \xi_i$$

* **RBF Kernel:** $K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$ where $\gamma = \frac{1}{4}$.
* **Platt Scaling:** Probability vectors are calibrated by fitting a sigmoid over decision function scores $f(x)$:

$$P(Y = 1 \mid x) = \frac{1}{1 + \exp(A f(x) + B)}$$

#### 3. Balanced Random Forest Classifier
An ensemble of $B = 100$ de-correlated decision trees. For each tree $b$, bootstrap sampling is modified with balanced subsampling such that each bootstrap drawing has an equal representation across all $K$ classes:

$$\hat{y}_{\text{RF}} = \arg\max_k \frac{1}{B} \sum_{b=1}^B P_b(Y = k \mid x)$$

* **Hyperparameters:** `n_estimators=100`, `max_depth=6`, `min_samples_split=5`, `class_weight='balanced'`.

#### 4. HistGradientBoosting Classifier
Constructs additive trees by partitioning continuous features into 256 discrete integer bins, minimizing multi-class cross-entropy via second-order Taylor approximations:

$$F_m(x) = F_{m-1}(x) + \eta \sum_{j=1}^{J_m} \gamma_{jm} I(x \in R_{jm})$$

* **Hyperparameters:** `learning_rate=0.05`, `max_iter=100`, `max_leaf_nodes=15`, `class_weight='balanced'`.

#### 5. Continuous Log-eGFR Regression + Post-Hoc Binning
Recognizing the intrinsic ordinal ordering of CKD stages, this model frames severity estimation as continuous regression followed by clinical discretization:
1. Target transformation: $z_i = \ln(\text{eGFR}_i)$.
2. Fit Ridge/ElasticNet Regressor: $\hat{z}(x) = w^T x + b$.
3. Exponentiate back to clinical units: $\widehat{\text{eGFR}} = \exp(\hat{z}(x))$.
4. Apply deterministic KDIGO thresholds:

$$\hat{C} = \begin{cases} 1 & \text{if } \widehat{\text{eGFR}} \ge 90 \\ 2 & \text{if } 60 \le \widehat{\text{eGFR}} < 90 \\ 3 & \text{if } 30 \le \widehat{\text{eGFR}} < 60 \\ 4 & \text{if } 15 \le \widehat{\text{eGFR}} < 30 \\ 5 & \text{if } \widehat{\text{eGFR}} < 15 \end{cases}$$

---

### 4.2 Quantum Architectures & Hilbert Space Embeddings

#### 1. Quantum State Encoding (`ZZFeatureMap`)
Maps classical feature vectors $x \in [0, \pi]^4$ into a $2^4 = 16$-dimensional complex Hilbert space:

$$|\psi(x)\rangle = U_{\Phi(x)} |0\rangle^{\otimes 4} = \left( \exp\left( i \sum_{j=1}^4 x_j Z_j + i \sum_{j < k} (\pi - x_j)(\pi - x_k) Z_j Z_k \right) H^{\otimes 4} \right) |0\rangle^{\otimes 4}$$

#### 2. Parameterized Variational Ansatz (`NLocal` / `RealAmplitudes`)
Applies parameterized single-qubit rotations and entangling CNOT gates:

$$V(\theta) = \prod_{l=1}^L \left( \text{CNOT}_{\text{entangle}} \bigotimes_{j=1}^4 R_y(\theta_{jl}) \right)$$

* **Variational Optimization:** Minimized cross-entropy loss over 16 parameters using COBYLA (Constrained Optimization BY Linear Approximation) with Qiskit's `SamplerQNN`.

#### 3. Quantum Fidelity Kernel Support Vector Machine
Computes the matrix of pairwise quantum state overlaps:

$$K_{ij} = |\langle \psi(x_i) | \psi(x_j) \rangle|^2 = \text{Tr}\left( \rho(x_i) \rho(x_j) \right)$$

The resulting Grammian matrix $K \in \mathbb{R}^{374 \times 374}$ is passed as a precomputed kernel into a classical dual quadratic program:

$$\max_\alpha \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i,j=1}^N \alpha_i \alpha_j y_i y_j K_{ij} \quad \text{s.t.} \quad 0 \le \alpha_i \le C, \ \sum_{i=1}^N \alpha_i y_i = 0$$

---

### 4.3 Late Fusion Architectures

Combines the out-of-fold probability vectors from the top classical estimator $P_{\text{classical}} \in \mathbb{R}^5$ and the quantum classifier $P_{\text{quantum}} \in \mathbb{R}^5$:

#### 1. Weighted Convex Combination:
$$P_{\text{fusion}}(Y = k \mid x) = \alpha P_{\text{classical}}(Y = k \mid x) + (1-\alpha) P_{\text{quantum}}(Y = k \mid x), \quad \alpha = 0.75$$

#### 2. Out-of-Fold Stacking Meta-Classifier:
$$\hat{Y} = \text{Softmax}\left( W_{\text{meta}} [P_{\text{classical}} \parallel P_{\text{quantum}}] + b_{\text{meta}} \right)$$

Trained strictly on out-of-fold predictions to prevent meta-learner data leakage.

---

## 5. Evaluation Benchmark & Empirical Findings

### 5.1 Final Consolidated Benchmark Table

All results reflect outer 5-fold $\times$ 5-repeat nested cross-validation ($N = 374$, 25 outer test evaluations):

| Target Task | Model Architecture | QWK (Primary) [95% CI] | Macro-F1 [95% CI] | Accuracy | MAE | Within $\pm 1$ [%] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **5-Class** | Logistic Regression (L2) | 0.8021 [0.788, 0.815] | **0.5419** [0.523, 0.559] | **0.5417** | 0.5508 | 92.6% |
| **5-Class** | Logistic Regression (ElasticNet) | 0.8029 [0.789, 0.816] | 0.5414 [0.524, 0.558] | 0.5401 | 0.5514 | 92.6% |
| **5-Class** | SVM-RBF (Calibrated) | 0.7885 [0.775, 0.801] | 0.4434 [0.431, 0.455] | 0.4717 | 0.6374 | 90.6% |
| **5-Class** | Random Forest (Balanced) | 0.8121 [0.803, 0.822] | 0.4963 [0.472, 0.521] | 0.4989 | 0.5792 | 93.3% |
| **5-Class** | HistGradientBoosting (Balanced) | 0.8147 [0.805, 0.826] | 0.4985 [0.481, 0.520] | 0.5000 | 0.5759 | 93.4% |
| **5-Class** | **Continuous Log-eGFR Regressor** | **0.8204** [0.811, 0.830] | 0.5236 [0.502, 0.546] | 0.5097 | **0.5486** | **95.2%** |
| **5-Class** | Calibrated Soft-Voting Ensemble | 0.8068 [0.797, 0.816] | 0.5087 [0.495, 0.522] | 0.5102 | 0.5770 | 92.8% |
| **3-Class** | Logistic Regression (L2) | 0.8208 [0.803, 0.839] | 0.7393 [0.716, 0.764] | 0.7777 | 0.2346 | 98.8% |
| **3-Class** | Logistic Regression (ElasticNet) | 0.8253 [0.808, 0.843] | 0.7413 [0.719, 0.765] | 0.7798 | 0.2309 | 98.9% |
| **3-Class** | SVM-RBF (Calibrated) | 0.8098 [0.791, 0.828] | 0.7130 [0.691, 0.735] | 0.7734 | 0.2448 | 98.2% |
| **3-Class** | Random Forest (Balanced) | 0.8342 [0.821, 0.848] | 0.7322 [0.715, 0.750] | 0.7813 | 0.2278 | 99.1% |
| **3-Class** | HistGradientBoosting (Balanced) | 0.8261 [0.817, 0.836] | 0.7075 [0.697, 0.719] | 0.7717 | 0.2396 | 98.9% |
| **3-Class** | **Continuous Log-eGFR Regressor** | **0.8563** [0.842, 0.871] | **0.7701** [0.752, 0.790] | **0.8145** | **0.1962** | **98.9%** |
| **3-Class** | Calibrated Soft-Voting Ensemble | 0.8389 [0.834, 0.844] | 0.7444 [0.740, 0.749] | 0.7888 | 0.2203 | 99.1% |
| Control | Quantum Fidelity Kernel SVC | 0.7109 [0.689, 0.731] | 0.4338 [0.412, 0.456] | -- | -- | -- |
| Control | Classical RBF Kernel SVC | 0.7763 [0.757, 0.795] | 0.4415 [0.421, 0.462] | -- | -- | -- |
| Ceiling | Upper-Bound (`sc` + `age` included) | 0.8802 | 0.6612 | 0.6526 | -- | -- |

---

## 6. Clinical Translation & Implementation Limitations

1. **Systematic Label Noise from Missing Biological Sex:**
   The absence of the `sex` column in the UCI CKD dataset introduces unavoidable diagnostic uncertainty. Because the CKD-EPI equation scales female filtration rates by $\approx 1.018$ with lower $\kappa = 0.7$, assuming male sex causes **33.4% of patients (125/374)** to shift category boundaries. Models trained under this assumption generalize stably to midpoint labels (Accuracy = 51.9%, QWK = 0.793), but prospective clinical translation demands datasets with explicitly verified patient sex.
2. **Cross-Sectional Surrogate vs True Chronicity:**
   Clinical diagnosis of CKD requires kidney impairment persisting for $> 3$ consecutive months. Cross-sectional screening data cannot distinguish between acute kidney injury (AKI) and irreversible chronic damage.
3. **The Definitive Quantum Verdict:**
   On small tabular clinical datasets ($N = 374$), state-of-the-art parameterized quantum circuits do not exhibit quantum advantage. The quantum fidelity kernel achieved lower downstream QWK (0.7109 vs 0.7763) than an identical classical RBF kernel. Paired bootstrap tests prove that late fusion degrades Macro-F1 ($\Delta = -0.048$) and produces a QWK difference spanning zero ($[-0.027, +0.015]$).
4. **Clinical Superiority of the Continuous Ordinal Regressor:**
   Framing CKD stage detection as a continuous log-eGFR regression with post-hoc KDIGO binning outperforms all multi-class classifiers, reaching **0.8563 QWK** on 3-class staging with **98.9%** within $\pm 1$ stage accuracy. This directly honors the continuous nature of renal physiology.

---

## 7. Directory Structure & Reproducibility Guide

```
├── PROJECT_DOCUMENTATION.md                  # This master document
├── configs/                                  # Experiment configuration schemas
├── data/
│   ├── processed/
│   │   ├── ckd_cleaned_imputed.csv           # Baseline cleaned dataset
│   │   └── ckd_egfr_categories_observed_only.csv # N=374 cohort with ground-truth targets
├── final_project_graphs/                     # Publication figures & SHAP visualizations
├── models/                                   # Serialized pipeline weights
├── reports/
│   ├── final_report.md                       # Peer-review scientific final report
│   ├── phase7_scientific_audit.md            # Detailed Phase 7 reconciliation audit
│   └── advanced_experimental_extensions.md  # Detailed statistical extensions report
├── results/
│   └── final_nested_cv.json                  # Complete fold-level nested CV raw data
├── src/
│   ├── classical/                            # Classical ML models & preprocessors
│   ├── quantum/                              # Qiskit VQC and Quantum Kernel pipelines
│   ├── fusion/                               # Late fusion & stacking meta-classifiers
│   └── experiments/
│       └── final_nested_cv.py                # Master end-to-end execution script
└── venv/                                     # Isolated Python 3.14 virtual environment
```

### Reproducibility Command:
To reproduce the complete benchmark and re-verify all 25 nested folds, quantum control experiments, and paired bootstrap statistics:

```bash
./venv/bin/python src/experiments/final_nested_cv.py
```

---
*Document authorized and committed to project repository.*
