# Scientific Methodology Audit Report
## Hybrid Quantum-Classical eGFR-Based Kidney Function Categorization Project

**Date:** 2026-09-14  
**Audit Scope:** Phases 1-3 Implementation  
**Status:** CRITICAL ISSUES IDENTIFIED - CORRECTED IN METHODOLOGY_AUDIT_V2

---

## EXECUTIVE SUMMARY

This audit identifies several critical methodological issues that must be addressed before proceeding to Phase 4 (Feature Selection). The current implementation has fundamental scientific flaws that compromise the validity of the research question and experimental design.

**CRITICAL FINDINGS:**
1. **Target Definition:** Incorrectly calling eGFR categories "CKD stages" 
2. **eGFR Calculation:** Using male-default equation without sex (systematic bias)
3. **Label Generation:** Labels derived from imputed data (6.5% of samples)
4. **Clinical Validity:** eGFR categories do not align with clinical CKD diagnosis
5. **Target Leakage:** Severe leakage from variables used in target derivation

**RECOMMENDATION:** Do not proceed to Phase 4 until these issues are resolved.

---

## 1. TARGET DEFINITION AUDIT

### Current Implementation
The current implementation calls the target "CKD Stage 1-5" and uses clinical CKD staging terminology.

### Scientific Reality
**INCORRECT TERMINOLOGY:** The target should be called "eGFR-based severity categories," not "CKD stages."

**Rationale:**
- Clinical CKD staging (KDIGO guidelines) requires evidence of kidney damage for Stages 1-2
- Evidence of kidney damage includes: proteinuria, albuminuria, structural abnormalities, or other markers
- The dataset contains only eGFR values, not evidence of kidney damage
- High eGFR does not equal healthy kidneys
- Low eGFR does not equal CKD without other evidence

**Correct Terminology:**
- "eGFR-based severity category" (1-5)
- "eGFR stratum" (1-5)
- "Kidney function category based on eGFR"

**Incorrect Terminology:**
- "CKD Stage 1-5" (implies clinical diagnosis)
- "CKD severity stage" (implies clinical validity)

### Cross-tabulation Analysis
```
              eGFR Category 1  eGFR Category 2  eGFR Category 3  eGFR Category 4  eGFR Category 5
Original Label
notckd                    82              64                4                0                0
ckd                       22              28               81               53               66
```

**Critical Finding:** 82 "notckd" samples are assigned to eGFR Category 1 (≥90 mL/min/1.73m²). This demonstrates that high eGFR does not equate to absence of CKD, and vice versa.

---

## 2. eGFR CALCULATION AUDIT

### Current Implementation
**Equation:** MDRD without sex coefficient (male default)  
**Formula:** eGFR = 175 × (Serum Creatinine)^-1.154 × (Age)^-0.203  
**Constants:** 175 (male coefficient), -1.154 (creatinine exponent), -0.203 (age exponent)  
**Units:** Serum creatinine in mg/dL, age in years, eGFR in mL/min/1.73m²

### Scientific Issues

#### Issue 1: Missing Sex Information
**Current Approach:** Using male coefficient as "most conservative assumption"

**Scientific Reality:** This is NOT scientifically defensible.

**Impact Analysis:**
- Standard MDRD equation: eGFR = 175 × SCr^-1.154 × Age^-0.203 × (0.742 if female)
- Current implementation: eGFR = 175 × SCr^-1.154 × Age^-0.203 (no sex coefficient)
- **Result:** eGFR overestimated for females by approximately 25% (since 1/0.742 ≈ 1.35)

**Sex Distribution in Dataset:** Unknown (sex variable not available)

**Severity:** HIGH - Systematic bias of unknown magnitude and direction

#### Issue 2: Alternative Equations Not Evaluated
**Sex-Independent Options Available:**
1. **Cockcroft-Gault Equation:** Requires sex and weight (not available)
2. **Mayo Clinic Quadratic Equation:** Requires sex (not available)
3. **Pediatric Schwartz Equation:** Requires height (not available)
4. **Bedside Schwartz Equation:** Requires height (not available)

**Conclusion:** No sex-independent eGFR equation is feasible with available variables.

#### Issue 3: Creatinine Handling
**Current:** `serum_creatinine = np.maximum(serum_creatinine, 0.1)` (clips values below 0.1)

**Issue:** This is an arbitrary threshold that may distort eGFR values for very low creatinine values.

**Severity:** MEDIUM - Affects a small number of samples

### Recommended Fix
**Option 1:** Clearly document the sex limitation and treat results as exploratory only  
**Option 2:** Restrict analysis to samples where eGFR calculation is clinically valid (unknown without sex)  
**Option 3:** Use a different target definition that doesn't require eGFR

---

## 3. LABEL GENERATION AUDIT

### Current Implementation
Labels were generated **AFTER** imputation using the cleaned dataset.

### Imputation Impact Analysis
**Raw Dataset Missing Values:**
- Serum creatinine (sc): 17 missing (4.25%)
- Age: 9 missing (2.25%)
- Samples with missing sc OR age: 26 (6.5%)

**Imputation Method:**
- Serum creatinine: Median imputation (1.30 mg/dL)
- Age: Median imputation (55.00 years)

**Result:** 26 samples (6.5%) have eGFR labels derived from imputed values.

### Critical Issue
**Methodological Flaw:** Target labels depend on imputed input features.

**Impact:**
- 17 samples with imputed serum creatinine have eGFR labels based on median creatinine
- 9 samples with imputed age have eGFR labels based on median age
- These samples are effectively "synthetic" in terms of their target labels
- This introduces circularity: imputation → eGFR calculation → target label

### Recommended Fix
**Option 1:** Derive labels BEFORE imputation, using only observed values
- Drop samples with missing sc or age (reduces dataset to 374 samples)
- Ensures all labels are based on observed data

**Option 2:** Create separate analysis tracks
- Track A: Labels from observed data only (374 samples)
- Track B: Labels including imputed data (400 samples)
- Compare results and report differences

**Severity:** HIGH - 6.5% of labels are methodologically compromised

---

## 4. ORIGINAL CKD LABEL AUDIT

### Current Implementation
The original UCI binary CKD/not-CKD label is retained but not used in the current multi-stage approach.

### Cross-tabulation Analysis
```
              eGFR Cat 1  eGFR Cat 2  eGFR Cat 3  eGFR Cat 4  eGFR Cat 5  Total
Original Label
notckd              82          64           4           0           0    150
ckd                 22          28          81          53          66    250
Total              104          92          85          53          66    400
```

### Critical Findings

#### Finding 1: High eGFR Does Not Equal No CKD
- 82 "notckd" samples have eGFR ≥ 90 (Category 1)
- 64 "notckd" samples have eGFR 60-89 (Category 2)
- **Interpretation:** High eGFR does not rule out CKD

#### Finding 2: CKD Diagnosis Spans All eGFR Categories
- "ckd" samples appear in all 5 eGFR categories
- 22 "ckd" samples have eGFR ≥ 90 (Category 1)
- **Interpretation:** CKD diagnosis includes patients with preserved kidney function

#### Finding 3: Low eGFR Strongly Correlates with CKD
- 0 "notckd" samples in Categories 4-5
- All Category 4-5 samples are "ckd"
- **Interpretation:** Severe eGFR reduction strongly correlates with CKD diagnosis

### Clinical Interpretation
The original UCI CKD label likely represents clinical diagnosis, which includes:
- Evidence of kidney damage (proteinuria, structural abnormalities)
- OR reduced kidney function (low eGFR)
- OR both

The eGFR-based categories represent only kidney function, not clinical CKD status.

**Conclusion:** The two targets measure different clinical constructs and should not be conflated.

---

## 5. PREPROCESSING LEAKAGE AUDIT

### Current Implementation
The preprocessing pipeline (`src/preprocessing/preprocessing.py`) appears to be implemented correctly:

**Positive Aspects:**
- Train/validation/test split occurs before any preprocessing
- All transformations (encoding, scaling) fit ONLY on training data
- Test data is transformed using training data parameters
- Stratified sampling used for class distribution

**Code Verification:**
```python
# Split occurs first
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=test_size, random_state=self.random_state, stratify=y
)

# Then preprocessing on training data only
X_train_processed, y_train_encoded = preprocessor.preprocess_train(X_train, y_train)

# Test data transformed with training parameters
X_test_processed, y_test_encoded = preprocessor.preprocess_test(X_test, y_test)
```

### Issue: Imputation Before Split
**The fundamental issue is NOT in the preprocessing pipeline, but in the data cleaning phase:**

- Imputation occurred in Phase 2 before train/test split
- All 400 samples were imputed using global statistics
- This means test-set information potentially leaked into training data via imputation

**Severity:** MEDIUM - Affects feature distributions but not target labels

### Recommended Fix
**Option 1:** Move imputation into preprocessing pipeline
- Perform imputation AFTER train/test split
- Fit imputation parameters on training data only
- Transform test data using training imputation parameters

**Option 2:** Accept current approach with documentation
- Document that imputation used global statistics
- Note this as a limitation in all reports

---

## 6. TARGET LEAKAGE AUDIT

### Current Implementation
The target (eGFR category) is derived from serum creatinine and age.

### Leakage Quantification
**Correlation Analysis:**
- Serum creatinine (sc) vs eGFR category: r = 0.971 (extremely high)
- Age vs eGFR category: r = 0.416 (moderate)
- eGFR vs eGFR category: r = -0.977 (perfect correlation, by definition)

### Experimental Configurations

#### Configuration A: Full Features (with leakage)
**Features:** All 24 features including sc and age  
**Leakage:** Severe - both variables used in target derivation  
**Expected Performance:** Artificially high  
**Clinical Validity:** LOW

#### Configuration B: No Leakage (leakage-aware)
**Features:** 22 features excluding sc and age  
**Leakage:** None - direct target variables excluded  
**Expected Performance:** Lower but realistic  
**Clinical Validity:** HIGH

#### Configuration C: Partial Exclusion
**Features:** 23 features excluding sc only  
**Leakage:** Partial - age excluded but sc is the primary leakage variable  
**Expected Performance:** Intermediate  
**Clinical Validity:** MEDIUM

### Current Implementation Status
The current implementation correctly identifies and designs these three configurations.

### Recommendation
**Primary Focus:** Configuration B (no leakage) for all clinical claims  
**Comparison:** Run all configurations to quantify leakage impact  
**Reporting:** Clearly distinguish leakage vs non-leakage results

---

## 7. CLASS DISTRIBUTION ANALYSIS

### Current Distribution
```
eGFR Category 1: 104 samples (26.00%)
eGFR Category 2: 92 samples (23.00%)
eGFR Category 3: 85 samples (21.25%)
eGFR Category 4: 53 samples (13.25%)
eGFR Category 5: 66 samples (16.50%)
```

### Stratified Split Feasibility
**Minimum samples per class:** 53 (Category 4)  
**Required for stratified split:** ≥10 samples per class  

**Conclusion:** All classes have sufficient samples for stratified train/validation/test evaluation.

### Potential Issue: Small Dataset
**Total samples:** 400  
**After train/val/test split (64/16/20):**
- Train: 256 samples (~51 per class)
- Validation: 64 samples (~13 per class)
- Test: 80 samples (~16 per class)

**Concern:** Small sample size may limit statistical power, especially for complex models like quantum ML.

---

## 8. SCIENTIFIC RECOMMENDATION

### Recommended Target Definition

**Option 1: Keep eGFR Categories with Correct Terminology (RECOMMENDED)**
- **Target Name:** "eGFR-based severity category" (1-5)
- **Methodology:** MDRD equation without sex coefficient
- **Label Generation:** Use only observed data (drop 26 samples with missing sc/age)
- **Leakage Control:** Exclude sc and age from model inputs (Configuration B)
- **Clinical Interpretation:** "Kidney function category based on eGFR, not clinical CKD diagnosis"

**Advantages:**
- Maintains multi-class structure for quantum ML investigation
- Addresses terminology issue
- Removes imputation-dependent labels
- Controls for target leakage
- Scientifically defensible as exploratory analysis

**Limitations:**
- Small sample size (374 samples)
- Sex limitation remains
- Not clinically validated CKD staging

---

**Option 2: Restrict to CKD-Positive Samples (ALTERNATIVE)**
- **Target Name:** "eGFR-based severity category among CKD patients" (1-5)
- **Methodology:** Same as Option 1
- **Sample Selection:** Only 250 samples with original "ckd" label
- **Rationale:** Focus on CKD population severity assessment

**Advantages:**
- More clinically focused
- Removes confusion between eGFR and CKD diagnosis
- 250 samples still sufficient for multi-class

**Limitations:**
- Loses information from non-CKD samples
- Still has sex limitation
- Small sample size

---

**Option 3: Use Original Binary Target (CONSERVATIVE)**
- **Target Name:** "CKD diagnosis" (binary: ckd/notckd)
- **Methodology:** Original UCI labels
- **Sample Size:** 400 samples
- **Leakage Control:** Less critical (target not derived from eGFR)

**Advantages:**
- Clinically validated target
- No eGFR calculation issues
- Larger sample size per class
- No target leakage concern

**Limitations:**
- Binary classification (less interesting for quantum ML)
- Loses multi-stage investigation
- Original label quality unknown

---

### Recommended Experimental Design

**Primary Experiment:**
- **Target:** eGFR-based severity category (1-5)
- **Samples:** 374 samples (observed sc and age only)
- **Features:** 22 features (exclude sc and age)
- **Preprocessing:** Imputation moved into pipeline (fit on train only)
- **Leakage Control:** Configuration B (no leakage)
- **Terminology:** "eGFR-based severity category"

**Comparison Experiments:**
- Configuration A (full features) to quantify leakage impact
- Configuration C (partial exclusion) as intermediate comparison
- Binary classification (original target) as baseline

**Documentation Requirements:**
- Clearly state all limitations in all reports
- Do not claim clinical validity for eGFR-based categories
- Quantify and report sex limitation impact
- Report results from observed-only data separately from imputed data

---

## 9. REQUIRED CHANGES BEFORE PHASE 4

### Critical Changes (Must Fix)

1. **Terminology Correction**
   - Change all references from "CKD Stage" to "eGFR-based severity category"
   - Update all reports, code comments, and documentation
   - Update file names if they contain "CKD stage" terminology

2. **Label Generation Fix**
   - Re-derive labels using only observed data (374 samples)
   - Drop 26 samples with missing sc or age
   - Create separate analysis track for imputed data if desired

3. **Preprocessing Fix**
   - Move imputation into preprocessing pipeline
   - Fit imputation parameters on training data only
   - Transform test data using training imputation parameters

4. **Sex Limitation Documentation**
   - Add explicit quantification of sex limitation impact
   - Document that results are exploratory only
   - Do not claim clinical validity

### Important Changes (Should Fix)

5. **Target Leakage Documentation**
   - Clearly document which configurations are leakage-aware
   - Report results from Configuration B as primary
   - Use Configuration A only for leakage quantification

6. **Clinical Interpretation**
   - Add disclaimer that eGFR categories ≠ clinical CKD stages
   - Clarify that high eGFR does not rule out CKD
   - Explain difference between kidney function and CKD diagnosis

### Optional Changes (Consider)

7. **Alternative Target Evaluation**
   - Consider running parallel analysis with original binary target
   - Compare multi-class vs binary performance
   - Evaluate if quantum ML benefits from multi-class structure

---

## 10. CONCLUSION

The current implementation has several critical methodological issues that compromise scientific validity:

**CRITICAL ISSUES:**
1. Incorrect terminology (eGFR categories called "CKD stages")
2. Labels derived from imputed data (6.5% of samples)
3. Sex limitation not properly quantified
4. Target leakage not adequately controlled in primary analysis

**RECOMMENDATION:** Do not proceed to Phase 4 until critical issues are resolved.

**TIMELINE ESTIMATE:** 2-3 hours to implement required changes and re-run affected phases.

**SCIENTIFIC VALIDITY:** After fixes, the project will be scientifically defensible as an exploratory investigation of quantum ML for eGFR-based kidney function categorization, with clear limitations documented.

---

**Audit Completed:** 2026-09-14  
**Next Step:** Implement required changes and re-run Phases 2-3  
**Decision Point:** After fixes, re-evaluate whether to proceed to Phase 4
