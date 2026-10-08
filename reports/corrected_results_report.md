# Corrected Results Report
## eGFR-Based Kidney Function Categorization Project

**Date:** 2026-09-14  
**Status:** CORRECTED METHODOLOGY - PHASES 1-3 COMPLETE  
**Validation:** ALL CHECKS PASSED

---

## 1. DATASET STATISTICS

### 1.1 Sample Information
**Original Dataset:** 400 samples (UCI Chronic Kidney Disease dataset)  
**Usable Samples (Observed-Only):** 374 samples  
**Samples Removed:** 26 samples (6.5%) due to missing required variables

### 1.2 Missing Variable Analysis
**Samples Removed Breakdown:**
- Missing serum creatinine (sc): 17 samples
- Missing age: 9 samples
- Total unique samples with missing sc OR age: 26 samples

**Rationale:** Target labels derived ONLY from observed serum creatinine and age to avoid circular dependency with imputed values.

---

## 2. eGFR CATEGORY DISTRIBUTION

### 2.1 Category Distribution (Observed-Only Data)
| Category | Count | Percentage | Description |
|----------|-------|------------|-------------|
| 1 | 102 | 27.27% | Normal or high kidney function (eGFR ≥ 90) |
| 2 | 84 | 22.46% | Mildly decreased kidney function (eGFR 60-89) |
| 3 | 74 | 19.79% | Moderately decreased kidney function (eGFR 30-59) |
| 4 | 52 | 13.90% | Severely decreased kidney function (eGFR 15-29) |
| 5 | 62 | 16.58% | Kidney failure (eGFR < 15) |

**Total:** 374 samples

### 2.2 Category Statistics
**Minimum Samples per Category:** 52 (Category 4)  
**Maximum Samples per Category:** 102 (Category 1)  
**Sparse Categories:** None (all categories have ≥10 samples)

**Conclusion:** All categories have sufficient samples for stratified train/validation/test evaluation.

---

## 3. ORIGINAL UCI CKD DISTRIBUTION

### 3.1 Original Binary Label Distribution (Observed-Only Data)
| Class | Count | Percentage |
|-------|-------|------------|
| ckd | 228 | 60.96% |
| ckd\t (malformed) | 2 | 0.53% |
| notckd | 144 | 38.50% |

**Total:** 374 samples

**Note:** The 2 malformed 'ckd\t' labels were cleaned to 'ckd' during preprocessing.

---

## 4. CROSS-TABULATION: ORIGINAL CKD LABEL vs eGFR CATEGORY

### 4.1 Cross-Tabulation Matrix
| Original Label | Category 1 | Category 2 | Category 3 | Category 4 | Category 5 | Total |
|----------------|------------|------------|------------|------------|------------|-------|
| ckd | 20 | 23 | 72 | 51 | 62 | 228 |
| ckd\t (malformed) | 0 | 0 | 1 | 1 | 0 | 2 |
| notckd | 82 | 61 | 1 | 0 | 0 | 144 |
| **Total** | **102** | **84** | **74** | **52** | **62** | **374** |

### 4.2 Key Findings
1. **High eGFR ≠ No CKD:** 143 "notckd" samples in Categories 1-2 (82 + 61)
2. **CKD Spans All Categories:** "ckd" samples appear in all 5 categories (20 in Category 1)
3. **Low eGFR ≈ CKD:** 0 "notckd" samples in Categories 4-5
4. **Different Constructs:** The two targets measure different clinical constructs

**Clinical Interpretation:** eGFR categories measure kidney function, not clinical CKD diagnosis. Clinical CKD requires evidence of kidney damage in addition to eGFR values.

---

## 5. TRAIN/VALIDATION/TEST SPLIT

### 5.1 Split Configuration
**Random State:** 42  
**Stratification:** Yes (by eGFR category)  
**Test Size:** 20.1% (75 samples)  
**Validation Size:** 16.0% (60 samples)  
**Train Size:** 63.9% (239 samples)

### 5.2 Split Results
| Split | Samples | Percentage | Category Distribution |
|-------|---------|------------|---------------------|
| Train | 239 | 63.9% | {1: 65, 2: 53, 3: 47, 4: 34, 5: 40} |
| Validation | 60 | 16.0% | {1: 16, 2: 14, 3: 12, 4: 8, 5: 10} |
| Test | 75 | 20.1% | {1: 21, 2: 17, 3: 15, 4: 10, 5: 12} |
| **Total** | **374** | **100%** | - |

### 5.3 Stratification Verification
**Maximum Category Distribution Difference:** < 10% across all splits  
**Conclusion:** Stratification successfully maintains class distribution.

---

## 6. EXPERIMENT CONFIGURATIONS

### 6.1 Experiment A: Full Features (with Leakage)
**Purpose:** Upper-bound/reference experiment only  
**Clinical Validity:** LOW - not clinically realistic  
**Leakage-Aware:** NO

**Features (24 total):**
- age ⚠️ (direct eGFR variable)
- bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc ⚠️ (direct eGFR variable)
- sod, pot, hemo, pcv, wbcc, rbcc, htn, dm, cad, appet, pe, ane

**Excluded Features:** None  
**Included Leakage Features:** sc, age

---

### 6.2 Experiment B: No Leakage (Leakage-Aware) - PRIMARY
**Purpose:** Clinically realistic scenario  
**Clinical Validity:** HIGH - clinically realistic  
**Leakage-Aware:** YES

**Features (22 total):**
- bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu
- sod, pot, hemo, pcv, wbcc, rbcc, htn, dm, cad, appet, pe, ane

**Excluded Features:** sc, age  
**Included Leakage Features:** None

**Status:** PRIMARY EXPERIMENT for all scientific claims

---

### 6.3 Experiment C: Partial Exclusion (Intermediate)
**Purpose:** Intermediate performance assessment  
**Clinical Validity:** MEDIUM - partially realistic  
**Leakage-Aware:** YES

**Features (23 total):**
- age ⚠️ (secondary eGFR variable)
- bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu
- sod, pot, hemo, pcv, wbcc, rbcc, htn, dm, cad, appet, pe, ane

**Excluded Features:** sc  
**Included Leakage Features:** age

---

## 7. PREPROCESSING LEAKAGE CONFIRMATION

### 7.1 Preprocessing Pipeline Integrity
**✅ CONFIRMED:** Preprocessing is leakage-free

**Evidence:**
1. **Imputation:** Fit on training data only (median for numeric, mode for categorical)
2. **Scaling:** Fit on training data only (StandardScaler)
3. **Encoding:** Fit on training data only (LabelEncoder)
4. **Split Order:** Train/validation/test split occurs BEFORE any preprocessing
5. **Test Data:** Never used for fitting any transformation parameters

### 7.2 Preprocessing Artifacts
**Saved Preprocessor:** `models/preprocessor_corrected.pkl`  
**Validation:** All preprocessing objects are fitted on training data only  
**Test Results:** Preprocessing test passed with no NaN values in processed data

---

## 8. METHODOLOGY VALIDATION

### 8.1 Automated Validation Checks
**Label Generation:** ✅ PASS  
**Preprocessing Artifacts:** ✅ PASS  
**Leakage Experiments:** ✅ PASS  
**Target Distribution:** ✅ PASS  
**Terminology:** ✅ PASS  
**Preprocessing Leakage-Free:** ✅ PASS

### 8.2 Data Quality Checks
**Observed Data:** 374 samples with no missing sc or age  
**Target Labels:** All valid (1-5), no NaN values  
**Class Distribution:** All categories ≥10 samples  
**Cross-tabulation:** Expected clinical patterns confirmed

### 8.3 Target Leakage Control
**Experiment A:** Includes leakage variables (sc, age) - for reference only  
**Experiment B:** Excludes leakage variables (sc, age) - PRIMARY  
**Experiment C:** Partial exclusion (excludes sc only) - intermediate

---

## 9. SCIENTIFIC LIMITATIONS

### 9.1 Sex Information Missing
**Limitation:** Sex variable not available in dataset  
**Impact:** eGFR calculation uses male-default MDRD equation  
**Systematic Error:** ~25% overestimation of eGFR for females  
**Sex Distribution:** Unknown (cannot quantify bias magnitude/direction)

**Mitigation:** 
- Clearly documented in all reports
- Results treated as exploratory only
- No clinical validity claimed

### 9.2 Small Sample Size
**Limitation:** 374 samples for 5-class classification  
**Impact:** Limited statistical power, especially for complex models  
**Mitigation:**
- Appropriate for exploratory quantum ML investigation
- Stratified sampling ensures class representation
- Results interpreted with appropriate caution

### 9.3 eGFR-Only Target
**Limitation:** Target based solely on eGFR, not clinical CKD diagnosis  
**Impact:** eGFR categories measure kidney function, not clinical CKD status  
**Mitigation:**
- Correct terminology used throughout
- Clinical interpretation disclaimers added
- Cross-tabulation with original labels provided

---

## 10. RECOMMENDATIONS FOR PHASE 4

### 10.1 Feature Selection Approach
**Primary Dataset:** `data/processed/ckd_egfr_categories_observed_only.csv` (374 samples)  
**Primary Experiment:** Experiment B (leakage-aware, 22 features)  
**Validation:** Use corrected preprocessor (`models/preprocessor_corrected.pkl`)

### 10.2 Feature Selection Strategy
**Target Qubit Count:** 8 features/qubits initially  
**Feature Count:** Select 8 features from 22 leakage-aware features  
**Methods:** 
- Correlation analysis
- Random Forest feature importance
- Mutual information
- Optional PCA

### 10.3 Experimental Protocol
**Primary Analysis:** Experiment B (leakage-aware)  
**Comparison:** Include Experiments A and C for leakage quantification  
**Reporting:** Clearly distinguish leakage vs non-leakage results

### 10.4 Documentation Standards
**Terminology:** Use "eGFR-based severity category" consistently  
**Limitations:** Document sex limitation in all reports  
**Scope:** Frame as exploratory quantum ML investigation

---

## 11. CONCLUSION

### 11.1 Corrected Methodology Status
**All Critical Issues:** ✅ RESOLVED  
**All Validation Checks:** ✅ PASSED  
**Preprocessing:** ✅ LEAKAGE-FREE  
**Data Integrity:** ✅ OBSERVED-ONLY  
**Terminology:** ✅ CORRECTED

### 11.2 Scientific Validity
**Status:** ✅ SCIENTIFICALLY DEFENSIBLE (EXPLORATORY)  
**Scope:** eGFR-based kidney function categorization  
**Clinical Validity:** Limited (eGFR categories ≠ clinical CKD stages)  
**Quantum ML Suitability:** Appropriate for exploratory investigation

### 11.3 Readiness for Phase 4
**Status:** ✅ READY  
**Confidence:** High - all validation checks passed  
**Data Quality:** Good - 374 samples, balanced categories  
**Methodology:** Sound - leakage-free preprocessing, correct terminology

**Recommendation:** Proceed to Phase 4 (Feature Selection) using the corrected methodology.

---

**Report Completed:** 2026-09-14  
**Status:** CORRECTED RESULTS READY  
**Next Step:** Phase 4 (Feature Selection)
