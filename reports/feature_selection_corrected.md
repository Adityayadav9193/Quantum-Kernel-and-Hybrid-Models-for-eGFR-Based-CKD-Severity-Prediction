# Feature Selection Report (CORRECTED)
## eGFR-Based Kidney Function Categorization Project

**Date:** 2026-09-14  
**Status:** PHASE 4 COMPLETE (CORRECTED)  
**Primary Experiment:** Experiment B (Leakage-Aware Exploratory Configuration)

---

## 1. FEATURE SELECTION LEAKAGE CONTROL

### ✅ CONFIRMED: Leakage-Free Feature Selection (CORRECTED)

**Leakage Control Measures:**
1. **Training Data Only:** All feature selection performed on 239 training samples (from established Phase 3 split)
2. **No Test Data:** Test data (75 samples) never used for feature selection
3. **No Validation Data:** Validation data (60 samples) not used for final feature selection
4. **Cross-Validation:** Feature stability analysis used independent training folds (within training data only)
5. **Consistent Split Protocol:** Uses established Phase 3 split (239/60/75 samples)

**Corrected Data Split for Feature Selection:**
- Training data: 239 samples (used for feature selection - ESTABLISHED Phase 3 split)
- Validation data: 60 samples (not used for feature selection)
- Test data: 75 samples (never used for feature selection)

**Experiment B Configuration:**
- **Excluded Features:** sc (serum creatinine), age (direct eGFR variables)
- **Included Features:** 22 leakage-aware features
- **Target:** eGFR-based severity category (1-5)
- **Configuration:** Leakage-aware exploratory configuration

---

## 2. FEATURE ANALYSIS RESULTS (CORRECTED)

### 2.1 Correlation Analysis

**Top Features by Correlation with eGFR Category:**
1. **bu (blood urea):** 0.665
2. **htn (hypertension):** -0.590
3. **dm (diabetes mellitus):** 0.501
4. **al (albumin):** 0.429
5. **pcv (packed cell volume):** -0.426

**Interpretation:** Blood urea shows the strongest positive correlation with eGFR category, while hypertension and diabetes show negative correlations.

### 2.2 Random Forest Feature Importance

**Top Features by Random Forest Importance:**
1. **bu (blood urea):** 0.1709
2. **hemo (hemoglobin):** 0.1017
3. **bgr (blood glucose random):** 0.0873
4. **pcv (packed cell volume):** 0.0865
5. **rbcc (red blood cell count):** 0.0767

**Interpretation:** Blood urea is the most important feature, followed by hemoglobin-related markers. This aligns with clinical expectations.

### 2.3 Mutual Information Analysis

**Top Features by Mutual Information:**
1. **bu (blood urea):** 0.5156
2. **pcv (packed cell volume):** 0.3386
3. **hemo (hemoglobin):** 0.3257
4. **rbcc (red blood cell count):** 0.3126
5. **htn (hypertension):** 0.3072

**Interpretation:** Mutual information confirms the importance of kidney function markers (blood urea, hemoglobin, cell counts).

### 2.4 Method Comparison

**Consistent Top Features Across Methods:**
- **bu (blood urea):** #1 in all methods
- **hemo (hemoglobin):** #2 in RF and MI, #7 in correlation
- **pcv (packed cell volume):** #4 in RF, #1 in MI, #5 in correlation
- **rbcc (red blood cell count):** #5 in RF, #4 in MI, #8 in correlation

**Method Differences:**
- **Correlation:** Emphasizes binary features (htn, dm)
- **Random Forest:** Emphasizes continuous numeric features
- **Mutual Information:** Balanced between numeric and categorical features

---

## 3. FEATURE STABILITY ANALYSIS (CORRECTED)

### 3.1 Selection Frequency (Top 8 Features)

**Features Selected in 100% of CV Folds:**
1. **bgr (blood glucose):** 1.00 (100%)
2. **bu (blood urea):** 1.00 (100%)
3. **pot (potassium):** 1.00 (100%)
4. **hemo (hemoglobin):** 1.00 (100%)
5. **pcv (packed cell volume):** 1.00 (100%)
6. **rbcc (red blood cell count):** 1.00 (100%)

**High Frequency Features:**
7. **sod (sodium):** 0.80 (80%)
8. **sg (specific gravity):** 0.60 (60%)

**Interpretation:** The 6 features with 100% selection frequency show excellent stability. Additional features (sod, sg) show high stability.

### 3.2 Rank Stability

**Most Stable Features (Lowest Rank Standard Deviation):**
1. **bu (blood urea):** 0.000 (perfectly stable)
2. **hemo (hemoglobin):** 0.000 (perfectly stable)
3. **appet (appetite):** 0.548 (highly stable)
4. **pcv (packed cell volume):** 0.548 (highly stable)
5. **al (albumin):** 0.548 (highly stable)

**Interpretation:** Blood urea and hemoglobin show perfect rank stability, confirming their importance.

---

## 4. CONSISTENT SELECTION RULE

### 4.1 Selection Methodology

**Consistent Selection Rule:**
1. **Primary:** Stability frequency (features selected in 100% of CV folds)
2. **Secondary:** RF + MI agreement (top features in both methods)
3. **Tertiary:** Correlation for tie-breaking

**Application:**
- Step 1: Identify features with 100% stability (6 features: bgr, bu, pot, hemo, pcv, rbcc)
- Step 2: Since 6 < 8 (target), add RF + MI agreement features (sg)
- Step 3: Add additional features from top RF importance (sod)
- Step 4: Final 8 features: bgr, bu, pot, hemo, pcv, rbcc, sg, sod

### 4.2 Final Selected Features (8 features)

| Feature | Clinical Name | RF Importance | Mutual Information | Correlation | Stability | Selected? |
|---------|---------------|---------------|-------------------|------------|-----------|-----------|
| bu | Blood urea | 0.1709 | 0.5156 | 0.665 | 1.00 | ✅ YES |
| hemo | Hemoglobin | 0.1017 | 0.3257 | -0.370 | 1.00 | ✅ YES |
| pcv | Packed cell volume | 0.0865 | 0.3386 | -0.426 | 1.00 | ✅ YES |
| rbcc | Red blood cell count | 0.0767 | 0.3126 | -0.359 | 1.00 | ✅ YES |
| bgr | Blood glucose random | 0.0873 | 0.1396 | 0.278 | 1.00 | ✅ YES |
| pot | Potassium | 0.0630 | 0.0964 | 0.133 | 1.00 | ✅ YES |
| sg | Specific gravity | 0.0534 | 0.2399 | -0.322 | 0.60 | ✅ YES |
| sod | Sodium | 0.0567 | 0.1617 | -0.084 | 0.80 | ✅ YES |

**Selection Consistency:** Final features are consistent with stability analysis (6/8 with 100% stability, 2/8 with high stability).

---

## 5. FEATURE COUNT COMPARISON

### 5.1 Alternative Feature Counts

**4 Features:** bgr, bu, pot, hemo (Combined score: 0.3751)  
**6 Features:** bgr, bu, pot, hemo, pcv, rbcc (Combined score: 0.3858)  
**8 Features:** bgr, bu, pot, hemo, pcv, rbcc, sg, sod (Combined score: 0.3533)  
**10 Features:** bgr, bu, pot, hemo, pcv, rbcc, sg, sod, wbcc, bp (Combined score: 0.3058)

**Recommendation:** 8 features provides good balance between clinical coverage and quantum feasibility.

---

## 6. PCA ANALYSIS (CORRECTED)

### 6.1 Variance Explained by Component Count
- **4 components:** 50.7% variance explained
- **6 components:** 62.5% variance explained
- **8 components:** 71.1% variance explained
- **10 components:** 78.0% variance explained

### 6.2 PCA vs Direct Feature Selection
**PCA Findings:**
- 8 components explain 71.1% of variance
- However, PCA components are not interpretable as clinical features
- PCA is useful for dimensionality reduction but not for explainability

**Direct Feature Selection Advantages:**
- Maintains clinical interpretability
- Allows explanation of quantum model decisions
- Better suited for medical applications

**Recommendation:** Use direct feature selection for quantum branch, PCA as baseline comparison only.

---

## 7. EXPERIMENT A/B/C COMPARISON

### 7.1 Experiment A: Full Features (with Leakage)
**Features:** 24 (includes sc, age)  
**Configuration:** Upper-bound/reference experiment only  
**Leakage Impact:** sc (serum creatinine) is #1 feature (expected due to target derivation)

### 7.2 Experiment B: No Leakage (Leakage-Aware) - PRIMARY
**Features:** 22 (excludes sc, age)  
**Configuration:** Leakage-aware exploratory configuration  
**Leakage Control:** Excludes direct eGFR variables (sc, age)

### 7.3 Experiment C: Partial Exclusion
**Features:** 23 (excludes sc only)  
**Configuration:** Leakage-aware (partial)  
**Leakage Control:** Excludes sc (primary leakage variable) but includes age

### 7.4 Key Comparison Insights
**Leakage Impact:** 
- Experiment A: sc (serum creatinine) dominates as #1 feature
- Experiment B: bu (blood urea) becomes #1 after excluding sc
- This confirms the leakage analysis predictions

**Consistent Features:**
- bu (blood urea) appears in top 5 across all experiments
- hemo (hemoglobin) appears in top 5 across all experiments
- pcv (packed cell volume) appears in top 5 across all experiments

---

## 8. FEATURE SELECTION INTEGRITY VERIFICATION

### ✅ Data Split Integrity (CORRECTED)
- **Training data used for feature selection:** YES (239 samples - ESTABLISHED Phase 3 split)
- **Validation data used for model selection:** YES (will be used in Phase 5)
- **Test data remains untouched:** YES (75 samples)
- **Preprocessing remains training-only:** YES
- **Consistent split protocol:** YES (Phase 3 split used throughout)

### ✅ Experiment B Configuration
- **Excludes age:** YES
- **Excludes sc (serum creatinine):** YES
- **No leakage variables included:** YES
- **Configuration:** Leakage-aware exploratory configuration

### ✅ Target Generation Integrity
- **Labels generated from observed data only:** YES (374 samples)
- **No imputed values used for target:** YES
- **eGFR-based category terminology:** YES

---

## 9. CLASS DISTRIBUTION

### 9.1 Category Distribution (Moderately Imbalanced)
| Category | Count | Percentage | Description |
|----------|-------|------------|-------------|
| 1 | 102 | 27.27% | Normal or high kidney function (eGFR ≥ 90) |
| 2 | 84 | 22.46% | Mildly decreased kidney function (eGFR 60-89) |
| 3 | 74 | 19.79% | Moderately decreased kidney function (eGFR 30-59) |
| 4 | 52 | 13.90% | Severely decreased kidney function (eGFR 15-29) |
| 5 | 62 | 16.58% | Kidney failure (eGFR < 15) |

**Distribution Characterization:** Moderately imbalanced (range: 13.90% - 27.27%)

---

## 10. LIMITATIONS

### 10.1 Sample Size
**Limitation:** 374 samples for feature selection  
**Impact:** Feature stability analysis limited by small sample size  
**Mitigation:** Cross-validation used, but results should be interpreted with caution

### 10.2 Missing Values in Features
**Limitation:** Selected features have missing values in original dataset  
**Impact:** Requires imputation during preprocessing  
**Mitigation:** Imputation will be done in preprocessing pipeline (train-only)

### 10.3 Categorical Feature Encoding
**Limitation:** Categorical features encoded as integers for feature selection  
**Impact:** May not capture all categorical information  
**Mitigation:** Will use proper encoding in preprocessing pipeline

### 10.4 Sex Information Missing
**Limitation:** Sex variable not available in dataset  
**Impact:** eGFR calculation uses male-default MDRD equation  
**Systematic Error:** ~25% overestimation of eGFR for females  
**Sex Distribution:** Unknown (cannot quantify bias magnitude/direction)

---

## 11. RECOMMENDATIONS FOR PHASE 5

### 11.1 Classical Baselines
**Use Selected Features:** bgr, bu, pot, hemo, pcv, rbcc, sg, sod  
**Preprocessing:** Use corrected preprocessor (train-only imputation)  
**Experiment:** Primary focus on Experiment B (leakage-aware exploratory configuration)

### 11.2 Quantum Model
**Initial Qubit Count:** 8 qubits  
**Feature Map:** ZZFeatureMap (to be tested)  
**Feature Count:** 8 features as selected above  
**Interpretability:** Maintain clinical feature names for explainability

### 11.3 Validation Protocol
**Continue Using:** Train/validation/test split with stratification (239/60/75)
**Feature Selection:** Use selected features only (no re-selection on validation/test)
**Model Selection:** Use validation set for hyperparameter tuning
**Final Evaluation:** Use test set only once

---

## 12. CORRECTIONS MADE

### 12.1 Training Split Consistency
**Issue:** Phase 4 created a new 80/20 split (299 samples) instead of using established Phase 3 split (239 samples)  
**Fix:** Re-ran feature selection using established Phase 3 training data (239 samples)  
**Result:** Consistent split protocol throughout project

### 12.2 Feature Selection Consistency
**Issue:** Final selected features (bu, hemo, pcv, rbcc, bgr, al, htn, sod) were inconsistent with stability analysis (bgr, bu, pot, hemo, pcv, rbcc, wbcc, sod)  
**Fix:** Applied consistent selection rule (stability frequency primary, RF+MI secondary)  
**Result:** Final features consistent with stability (bgr, bu, pot, hemo, pcv, rbcc, sg, sod)

### 12.3 Terminology
**Issue:** Original report used inappropriate terminology for exploratory analysis  
**Fix:** Changed to "Leakage-aware exploratory configuration"  
**Issue:** Classes described as "balanced"  
**Fix:** Changed to "moderately imbalanced" with exact distribution

### 12.4 PCA Implementation
**Issue:** PCA was fitted on new split (299 samples) instead of established training data (239 samples)  
**Fix:** Re-ran PCA using established training data (239 samples)  
**Result:** Consistent methodology throughout

---

## 13. CONCLUSION

### 13.1 Feature Selection Status
**✅ Feature Selection:** COMPLETE (CORRECTED)  
**✅ No Feature-Selection Leakage:** CONFIRMED (CORRECTED)  
**✅ Experiment B Correctly Configured:** CONFIRMED  
**✅ Feature Stability Analysis:** COMPLETE  
**✅ PCA Analysis:** COMPLETE (CORRECTED)  
**✅ Consistent Selection Rule:** APPLIED

### 13.2 Final Selected Features
**Count:** 8 features  
**List:** bgr, bu, pot, hemo, pcv, rbcc, sg, sod  
**Justification:** Consistent with stability analysis (6/8 with 100% stability), high predictive relevance, clinical interpretability, quantum feasibility

### 13.3 Readiness for Phase 5
**Status:** ✅ READY (CORRECTED)  
**Confidence:** High - leakage-free methodology, consistent protocol, stable features  
**Next Step:** Phase 5 (Classical Baselines) using selected features and established split protocol

---

**Report Completed:** 2026-09-14  
**Status:** PHASE 4 COMPLETE (CORRECTED)  
**Next Phase:** Phase 5 (Classical Baselines)
