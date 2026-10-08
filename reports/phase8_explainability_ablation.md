# Phase 8: Explainability, Ablation, and Error Analysis

## 1. Classical Explainability (SHAP)
**Objective:** Explain the classical Logistic Regression model's predictions.
- **Method:** `shap.LinearExplainer` applied to the frozen Phase 5 LR model.
- **Scope:** SHAP values explain *only* the classical LR branch, not the quantum circuit.
- **Findings:** The features most influential globally (such as `sg`, `pcv`, `hemo`, and `bgr`) drive the classical predictions. Class-specific feature importance reveals how extreme values (e.g., low hemoglobin) push the LR toward predicting more severe eGFR categories.
- **Artifact:** `results/figures/phase8/shap_summary.png`

## 2. Quantum Feature Sensitivity Analysis
**Objective:** Understand how input perturbations affect the frozen VQC's predictions.
- **Method:** For each of the 8 features, a +1 standard deviation perturbation was applied to the test set, keeping other features fixed. The mean absolute change in the output probability vector was recorded.
- **Scope:** This is a sensitivity/perturbation analysis, *not* causal "quantum SHAP".
- **Findings:** The VQC exhibits varied sensitivity across features. Due to the ZZFeatureMap entanglement, perturbations propagate non-linearly, but the overall probability shifts remain constrained, reflecting the model's convergence state.
- **Artifact:** `results/figures/phase8/quantum_sensitivity.png`

## 3. Ablation Study
**Objective:** Evaluate the contribution of the quantum branch.

| Branch | Test Macro-F1 |
|--------|---------------|
| A. Classical LR | 0.4468 |
| B. Quantum VQC (Original) | 0.1130 |
| C. Late Fusion | 0.4614 |

**Analysis:** The fusion of LR + VQC yields a modest improvement (+0.0146) over LR alone. The quantum branch, while performing poorly on its own, acts as a weak learner that provides slightly complementary probability signals (particularly aiding Categories 1 and 3). 

## 4. Error Analysis
**Objective:** Analyze test samples misclassified by the Late Fusion model.
- **Total Test Errors:** 39 out of 75 (Accuracy = 0.4800).
- **Confusion Patterns:** Errors are heavily concentrated in neighboring severity categories (e.g., predicting Category 2 instead of 1, or 4 instead of 3). 
- **Hardest Categories:** Category 4 is the most difficult to predict accurately (F1=0.211), likely due to lower sample count and overlapping biochemical signatures with Categories 3 and 5.
- **Conclusion:** The model struggles to perfectly delineate exact eGFR boundaries but generally identifies the correct broader severity neighborhood.
