# Dataset Audit Report
## UCI Chronic Kidney Disease Dataset

**Date:** 2026-09-14  
**UCI ID:** 336  
**Source:** https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease

---

## 1. Dataset Overview

### Basic Information
- **Total Samples:** 400
- **Total Features:** 25 (24 input features + 1 target)
- **Dataset Type:** Clinical/medical
- **Target Variable:** `class` (binary: ckd/notckd)

### Feature List
| Feature | Type | Description |
|---------|------|-------------|
| age | numeric | Age in years |
| bp | numeric | Blood pressure in mm/Hg |
| sg | numeric | Specific gravity |
| al | numeric | Albumin |
| su | numeric | Sugar |
| rbc | categorical | Red blood cells (normal/abnormal) |
| pc | categorical | Pus cell (normal/abnormal) |
| pcc | categorical | Pus cell clumps (present/notpresent) |
| ba | categorical | Bacteria (present/notpresent) |
| bgr | numeric | Blood glucose random |
| bu | numeric | Blood urea |
| sc | numeric | Serum creatinine |
| sod | numeric | Sodium |
| pot | numeric | Potassium |
| hemo | numeric | Hemoglobin |
| pcv | numeric | Packed cell volume |
| wbcc | numeric | White blood cell count |
| rbcc | numeric | Red blood cell count |
| htn | categorical | Hypertension (yes/no) |
| dm | categorical | Diabetes mellitus (yes/no) |
| cad | categorical | Coronary artery disease (yes/no) |
| appet | categorical | Appetite (good/poor) |
| pe | categorical | Pedal edema (yes/no) |
| ane | categorical | Anemia (yes/no) |
| class | categorical | Target: CKD status (ckd/notckd) |

---

## 2. Data Quality Assessment

### Missing Values Analysis

| Feature | Missing Count | Missing % | Severity |
|---------|---------------|-----------|----------|
| rbc | 152 | 38.00% | **HIGH** |
| rbcc | 131 | 32.75% | **HIGH** |
| wbcc | 106 | 26.50% | **MEDIUM** |
| pot | 88 | 22.00% | **MEDIUM** |
| sod | 87 | 21.75% | **MEDIUM** |
| pcv | 71 | 17.75% | **MEDIUM** |
| pc | 65 | 16.25% | **MEDIUM** |
| hemo | 52 | 13.00% | **LOW** |
| su | 49 | 12.25% | **LOW** |
| sg | 47 | 11.75% | **LOW** |
| al | 46 | 11.50% | **LOW** |
| bgr | 44 | 11.00% | **LOW** |
| bu | 19 | 4.75% | **LOW** |
| sc | 17 | 4.25% | **LOW** |
| bp | 12 | 3.00% | **LOW** |
| age | 9 | 2.25% | **LOW** |
| htn | 2 | 0.50% | **LOW** |
| dm | 2 | 0.50% | **LOW** |
| cad | 2 | 0.50% | **LOW** |
| appet | 1 | 0.25% | **LOW** |
| pe | 1 | 0.25% | **LOW** |
| ane | 1 | 0.25% | **LOW** |
| class | 0 | 0.00% | - |

**Summary:** 21 out of 25 features have missing values. Red blood cell features (rbc, rbcc) have the highest missing rates (>30%).

### Data Type Distribution
- **Numeric features:** 14 (age, bp, sg, al, su, bgr, bu, sc, sod, pot, hemo, pcv, wbcc, rbcc)
- **Categorical features:** 11 (rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane, class)

### Data Quality Issues

#### 1. Whitespace Inconsistencies
- **Target variable `class`:** Contains 'ckd\t' (tab character) instead of 'ckd' (2 instances)
- **Diabetes `dm`:** Contains '\tno' (tab character) instead of 'no'

#### 2. Duplicate Rows
- **Duplicate count:** 0
- **Duplicate percentage:** 0.00%

---

## 3. Class Distribution

### Target Variable: `class`
| Class | Count | Percentage |
|-------|-------|------------|
| ckd | 248 | 62.00% |
| notckd | 150 | 37.50% |
| ckd\t (malformed) | 2 | 0.50% |

**Issues:**
- Class imbalance: 62% CKD vs 38% non-CKD
- Malformed label 'ckd\t' needs cleaning (should be 'ckd')

---

## 4. Statistical Summary (Numeric Features)

| Feature | Mean | Std | Min | 25% | 50% | 75% | Max |
|---------|------|-----|-----|-----|-----|-----|-----|
| age | 51.48 | 17.17 | 2.0 | 42.0 | 55.0 | 64.5 | 90.0 |
| bp | 76.47 | 13.68 | 50.0 | 70.0 | 80.0 | 80.0 | 180.0 |
| sg | 1.017 | 0.006 | 1.005 | 1.010 | 1.020 | 1.020 | 1.025 |
| al | 0.43 | 1.18 | 0.0 | 0.0 | 0.0 | 1.0 | 5.0 |
| su | 0.49 | 0.91 | 0.0 | 0.0 | 0.0 | 0.0 | 5.0 |
| bgr | 148.03 | 79.96 | 22.0 | 85.0 | 119.0 | 192.0 | 490.0 |
| bu | 57.43 | 23.50 | 8.0 | 42.0 | 54.0 | 66.0 | 180.0 |
| sc | 3.07 | 3.59 | 0.4 | 1.1 | 1.6 | 2.9 | 24.0 |
| sod | 137.53 | 10.23 | 4.5 | 131.0 | 138.0 | 143.0 | 163.0 |
| pot | 4.64 | 1.60 | 2.5 | 4.0 | 4.6 | 5.4 | 9.0 |
| hemo | 12.53 | 2.83 | 3.1 | 10.3 | 12.6 | 14.4 | 17.8 |
| pcv | 38.88 | 8.99 | 9.0 | 32.0 | 40.0 | 45.0 | 54.0 |
| wbcc | 8406.12 | 2944.47 | 2200.0 | 6500.0 | 8000.0 | 9800.0 | 26400.0 |
| rbcc | 4.71 | 1.03 | 2.1 | 3.9 | 4.8 | 5.4 | 8.0 |

**Potential Issues:**
- `age` has a minimum of 2.0 years (possibly pediatric cases)
- `sod` has a suspicious minimum of 4.5 (likely data error)
- `sc` (serum creatinine) has a wide range (0.4 to 24.0), indicating severe CKD cases

---

## 5. Categorical Feature Analysis

### Binary/Categorical Features
| Feature | Unique Values | Values |
|---------|---------------|--------|
| rbc | 2 | normal, abnormal |
| pc | 2 | normal, abnormal |
| pcc | 2 | present, notpresent |
| ba | 2 | present, notpresent |
| htn | 2 | yes, no |
| dm | 2 | yes, no (plus '\tno' whitespace issue) |
| cad | 2 | yes, no |
| appet | 2 | good, poor |
| pe | 2 | yes, no |
| ane | 2 | yes, no |

---

## 6. Critical Findings for Multi-Stage Label Derivation

### Available Variables for eGFR Calculation
✅ **serum creatinine (sc):** Present  
✅ **age:** Present  
❌ **sex:** **NOT PRESENT**

### Implications
The standard CKD-EPI (Chronic Kidney Disease Epidemiology Collaboration) equation requires sex as a parameter. The absence of sex information is a **major limitation** for accurate eGFR calculation.

**Recommended Action:**
- Investigate sex-independent eGFR equations (e.g., MDRD equation without sex modification, or pediatric equations if applicable)
- Clearly document this limitation in all reports
- Consider alternative staging approaches if eGFR calculation is not feasible
- Do not invent or impute sex values

---

## 7. Recommendations for Data Cleaning

### Immediate Actions Required
1. **Fix whitespace issues:**
   - Replace 'ckd\t' with 'ckd' in target variable
   - Replace '\tno' with 'no' in dm variable

2. **Handle missing values:**
   - Implement appropriate imputation strategies for numeric features (median imputation)
   - Implement appropriate imputation for categorical features (mode imputation or 'missing' category)

3. **Handle potential outliers:**
   - Investigate `sod = 4.5` as potential data error
   - Consider capping extreme values after clinical validation

4. **Investigate pediatric cases:**
   - Age range includes very young patients (2 years)
   - Pediatric eGFR equations differ from adult equations
   - May need to segment data by age group

---

## 8. Dataset Limitations

1. **Small sample size:** 400 samples is small for multi-class ML and quantum ML
2. **High missingness:** Several features have >20% missing values
3. **Missing sex:** Critical limitation for eGFR calculation
4. **Class imbalance:** 62% CKD vs 38% non-CKD
5. **Potential data quality issues:** Outliers in sodium, whitespace inconsistencies
6. **Age range:** Includes pediatric cases (age 2), which may require different clinical protocols

---

## 9. Conclusions

The UCI Chronic Kidney Disease dataset is suitable for initial research but has significant limitations:

**Strengths:**
- Comprehensive clinical features
- Includes key renal function markers (creatinine, urea, hemoglobin)
- Official UCI dataset with clear documentation

**Weaknesses:**
- Small sample size for multi-stage classification
- High missingness in several features
- **Missing sex variable - critical for eGFR calculation**
- Data quality issues (whitespace, potential outliers)

**Next Steps:**
1. Clean whitespace issues
2. Implement robust missing value imputation
3. Investigate sex-independent eGFR calculation methods
4. Consider reduced-stage classification if multi-stage proves infeasible
5. Document all limitations in research reports

---

**Report Generated:** 2026-09-14  
**Phase 1 Status:** ✅ COMPLETE
