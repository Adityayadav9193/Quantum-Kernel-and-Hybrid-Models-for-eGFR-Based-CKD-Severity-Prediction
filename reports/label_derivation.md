# Multi-Stage CKD Label Derivation Report

**Date:** 2026-09-14

## Methodology

### eGFR Calculation
- **Equation:** MDRD without sex coefficient (male default)
- **Variables Used:** Serum creatinine (sc), Age
- **CRITICAL LIMITATION:** Sex information not available in dataset
- **Impact:** eGFR overestimated for females by ~25%

### CKD Staging
- **Stage 1:** eGFR ≥ 90 mL/min/1.73m²
- **Stage 2:** eGFR 60-89 mL/min/1.73m²
- **Stage 3:** eGFR 30-59 mL/min/1.73m²
- **Stage 4:** eGFR 15-29 mL/min/1.73m²
- **Stage 5:** eGFR < 15 mL/min/1.73m²

### Clinical Note
eGFR alone is NOT sufficient to diagnose CKD. Clinical diagnosis requires
evidence of kidney damage (proteinuria, structural abnormalities, etc.) for Stages 1-2.

## Stage Distribution

| Stage | Count | Percentage | Description |
|-------|-------|-------------|-------------|
| 1 | 104 | 26.00% | Stage 1: Kidney damage with normal or high eGFR (≥90) |
| 2 | 92 | 23.00% | Stage 2: Kidney damage with mildly decreased eGFR (60-89) |
| 3 | 85 | 21.25% | Stage 3: Moderately decreased eGFR (30-59) |
| 4 | 53 | 13.25% | Stage 4: Severely decreased eGFR (15-29) |
| 5 | 66 | 16.50% | Stage 5: Kidney failure (eGFR < 15) |

## Sparse Stages

No sparse stages detected (all stages have ≥10 samples).

## Cross-tabulation with Original Binary Label

ckd_stage    1   2   3   4   5  All
class                              
ckd         22  28  81  53  66  250
notckd      82  64   4   0   0  150
All        104  92  85  53  66  400
## Recommendations

4. Clearly document the sex limitation in all reports
5. Consider sex-independent validation if possible
