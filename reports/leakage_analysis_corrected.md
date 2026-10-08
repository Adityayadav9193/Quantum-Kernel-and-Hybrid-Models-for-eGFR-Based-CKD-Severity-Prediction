# Corrected Target Leakage Analysis Report

**Date:** 2026-09-14
**Status:** CORRECTED - Uses observed-only data

## Leakage Mechanism

The target variable (eGFR-based severity category) is derived from eGFR, which is calculated using:
- **Serum creatinine (sc)**
- **Age**
- **Sex** (not available - using male default)

This creates a direct mathematical relationship between these features and the target.
Including them in model inputs constitutes target leakage.

## Leakage Analysis Results

### Correlation with eGFR Category
- Serum creatinine (sc): Very high correlation (expected - directly used in eGFR)
- Age: Moderate correlation (used in eGFR calculation)
- eGFR: Perfect correlation (derived from eGFR)

### Statistical Impact
Models that include 'sc' and 'age' will learn the eGFR formula directly,
leading to artificially high performance that does not reflect clinical reality.

## Experimental Design

### Experiment A: Full Features (with leakage)
- **Features:** All available features including sc and age
- **Purpose:** Establish upper bound performance
- **Expected:** High performance due to leakage
- **Clinical validity:** LOW - not clinically realistic

### Experiment B: No Leakage (leakage-aware)
- **Features:** All features except sc and age
- **Purpose:** Clinically realistic scenario
- **Expected:** Lower but more realistic performance
- **Clinical validity:** HIGH - represents real clinical scenario
- **Status:** PRIMARY EXPERIMENT for scientific claims

### Experiment C: Partial Exclusion
- **Features:** All features except sc (includes age)
- **Purpose:** Intermediate scenario
- **Expected:** Intermediate performance
- **Clinical validity:** MEDIUM - age is clinically relevant but less direct

## Recommendations

1. **Primary focus:** Experiment B (leakage-aware) for all clinical claims
2. **Comparison:** Run all experiments to quantify leakage impact
3. **Reporting:** Clearly distinguish leakage vs non-leakage results
4. **Clinical interpretation:** Only leakage-aware results for clinical claims
5. **Documentation:** Explicitly state which features were excluded and why

## Additional Considerations

### Missing Sex Information
The absence of sex information is a separate limitation that affects
the accuracy of eGFR calculation itself, not just the leakage issue.
This must be documented in all reports.

### Data Quality
Analysis uses observed-only data (374 samples).
26 samples (6.5%) excluded due to missing sc or age.
No imputed values used for target label generation.

### Target Definition
The target represents eGFR-based severity categories, NOT clinical CKD stages.
Clinical CKD diagnosis requires evidence of kidney damage in addition to eGFR.
