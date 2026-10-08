# Phase 2 Summary: Multi-Stage CKD Label Derivation

## Completed Tasks

### 1. eGFR Calculation Module
- ✅ Created `src/clinical/egfr.py` with multiple eGFR equation implementations
- ✅ Implemented MDRD equation without sex coefficient (male default)
- ✅ Documented critical limitation: missing sex information
- ✅ Added clear clinical warnings about eGFR-only staging

### 2. Multi-Stage Label Derivation
- ✅ Derived 5-stage CKD labels from eGFR calculations
- ✅ Applied KDIGO staging guidelines:
  - Stage 1: eGFR ≥ 90 mL/min/1.73m²
  - Stage 2: eGFR 60-89 mL/min/1.73m²
  - Stage 3: eGFR 30-59 mL/min/1.73m²
  - Stage 4: eGFR 15-29 mL/min/1.73m²
  - Stage 5: eGFR < 15 mL/min/1.73m²

### 3. Stage Distribution Analysis
- ✅ Analyzed distribution across all 5 stages
- ✅ Created visualization of stage distribution
- ✅ Cross-tabulated with original binary labels

## Key Findings

### Stage Distribution (All Stages Have Sufficient Samples)
| Stage | Count | Percentage | Clinical Description |
|-------|-------|------------|---------------------|
| 1 | 104 | 26.00% | Normal/high eGFR |
| 2 | 92 | 23.00% | Mildly decreased eGFR |
| 3 | 85 | 21.25% | Moderately decreased eGFR |
| 4 | 53 | 13.25% | Severely decreased eGFR |
| 5 | 66 | 16.50% | Kidney failure |

**Result:** All stages have ≥10 samples, making multi-stage classification feasible.

### Cross-tabulation with Original Binary Label
| Class | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 | Total |
|-------|---------|---------|---------|---------|---------|-------|
| ckd | 22 | 28 | 81 | 53 | 66 | 250 |
| notckd | 82 | 64 | 4 | 0 | 0 | 150 |

**Key Insight:** 
- Stages 1-2 contain many "notckd" samples (high eGFR without evidence of kidney damage)
- Stages 3-5 are almost exclusively "ckd" samples (clinically significant CKD)
- This aligns with clinical understanding: eGFR alone is not sufficient for early-stage CKD diagnosis

### Critical Limitation Documented
**Missing Sex Information:**
- Standard CKD-EPI equation requires sex as a parameter
- Using MDRD equation without sex coefficient (male default)
- This overestimates eGFR for females by approximately 25%
- This is a significant limitation that must be documented in all reports

## Methodology Used

### eGFR Calculation
- **Equation:** MDRD without sex coefficient
- **Variables:** Serum creatinine (sc), Age
- **Formula:** eGFR = 175 × (Serum Creatinine)^-1.154 × (Age)^-0.203

### Clinical Note
eGFR alone is NOT sufficient to diagnose CKD. Clinical diagnosis requires evidence of kidney damage (proteinuria, structural abnormalities, etc.) for Stages 1-2. The derived stages should be interpreted as "eGFR-based stages" rather than clinical CKD diagnoses.

## Files Created
- `src/clinical/egfr.py` - eGFR calculation module
- `src/data/03_label_derivation.py` - Label derivation script
- `data/processed/ckd_multistage.csv` - Multi-stage dataset
- `reports/label_derivation.md` - Detailed derivation report
- `results/figures/stage_distribution.png` - Stage distribution visualization

## Decision Point: Proceed with Multi-Stage?

**Recommendation:** ✅ **YES - Proceed with multi-stage classification**

**Rationale:**
1. All 5 stages have sufficient samples (≥10 each)
2. Reasonable distribution across stages (not heavily imbalanced)
3. Clinically meaningful progression across stages
4. Clear separation between early stages (1-2) and advanced stages (3-5)
5. The sex limitation is documented and can be addressed in experimental design

**Caveats:**
- Must document the sex limitation in all reports
- Must implement leakage-aware experiments (exclude sc and age from some models)
- Must not overclaim clinical validity from eGFR-only staging

## Next Steps (Phase 3)

**Leakage Analysis and Preprocessing:**
1. Investigate target leakage (eGFR derived from sc and age)
2. Design leakage-aware experiments
3. Implement preprocessing pipeline
4. Set up train/validation/test splits with stratification
5. Ensure no test-set information leaks into preprocessing

## Phase 2 Status: ✅ COMPLETE
