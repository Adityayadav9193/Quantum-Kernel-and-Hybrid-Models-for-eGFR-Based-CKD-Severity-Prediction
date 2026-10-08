# Corrected eGFR-Based Category Label Derivation Report

**Date:** 2026-09-14
**Status:** CORRECTED - Uses only observed data

## Methodology

### eGFR Calculation
- **Equation:** MDRD without sex coefficient (male default)
- **Variables Used:** Serum creatinine (sc), Age
- **Data Source:** Raw dataset (NO imputation)
- **Sample Selection:** Only samples with observed sc and age
- **Samples Excluded:** 26 samples with missing required variables
- **Final Sample Size:** 374 samples

### CRITICAL LIMITATIONS

1. **Missing Sex Information:** Sex variable not available in dataset
2. **Equation Bias:** Using male-default MDRD equation
3. **Systematic Error:** eGFR overestimated for females by ~25%
4. **Unknown Sex Distribution:** Cannot quantify bias magnitude/direction
5. **Target Definition:** eGFR-based categories, NOT clinical CKD stages
6. **Clinical Validity:** eGFR alone insufficient for CKD diagnosis

### eGFR-Based Category Definitions
- **Category 1:** eGFR ≥ 90 mL/min/1.73m² (normal/high kidney function)
- **Category 2:** eGFR 60-89 mL/min/1.73m² (mildly decreased kidney function)
- **Category 3:** eGFR 30-59 mL/min/1.73m² (moderately decreased kidney function)
- **Category 4:** eGFR 15-29 mL/min/1.73m² (severely decreased kidney function)
- **Category 5:** eGFR < 15 mL/min/1.73m² (kidney failure)

### Clinical Note
These are eGFR-based severity categories, NOT clinical CKD stages.
Clinical CKD diagnosis requires evidence of kidney damage (proteinuria,
structural abnormalities, etc.) in addition to eGFR values.
High eGFR does not rule out CKD, and low eGFR does not equal CKD
without other evidence.

## Category Distribution

| Category | Count | Percentage | Description |
|----------|-------|-------------|-------------|
| 1 | 102 | 27.27% | Category 1: Normal or high kidney function (eGFR ≥ 90) |
| 2 | 84 | 22.46% | Category 2: Mildly decreased kidney function (eGFR 60-89) |
| 3 | 74 | 19.79% | Category 3: Moderately decreased kidney function (eGFR 30-59) |
| 4 | 52 | 13.90% | Category 4: Severely decreased kidney function (eGFR 15-29) |
| 5 | 62 | 16.58% | Category 5: Kidney failure (eGFR < 15) |

## Sparse Categories

No sparse categories detected (all categories have ≥10 samples).

## Cross-tabulation with Original Binary Label

This cross-tabulation shows that eGFR categories and clinical CKD diagnosis
measure different clinical constructs:

egfr_category    1   2   3   4   5  All
class                                  
ckd             20  23  72  51  62  228
ckd\t            0   0   1   1   0    2
notckd          82  61   1   0   0  144
All            102  84  74  52  62  374
## Key Findings

1. High eGFR does not equal absence of CKD (many 'notckd' in Categories 1-2)
2. CKD diagnosis spans all eGFR categories (some 'ckd' in Category 1)
3. Low eGFR strongly correlates with CKD (no 'notckd' in Categories 4-5)
4. The two targets measure different clinical constructs

## Recommendations

4. Clearly document the sex limitation in all reports
5. Use correct terminology (eGFR-based categories, not CKD stages)
6. Do not claim clinical validity for eGFR-based categories
