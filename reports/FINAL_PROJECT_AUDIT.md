# Final Project Audit

## 1. Project Summary
**Project:** Hybrid Quantum-Classical Framework for Explainable Multi-Grade CKD Detection
**Target:** Exploratory eGFR-based severity categories (1-5). NOT clinically confirmed CKD stages.
**Splits:** 239 Train / 60 Validation / 75 Test.
**Selected Features:** 8 features (`bgr`, `bu`, `pot`, `hemo`, `pcv`, `rbcc`, `sg`, `sod`). Leakage features (`age`, `sc`) successfully excluded.

## 2. Final Models and Metrics
| Model | Validation Macro-F1 | Test Macro-F1 | Test Accuracy | Test Weighted-F1 |
|-------|---------------------|---------------|---------------|------------------|
| Classical LR (Phase 5) | 0.4109 | 0.4468 | 0.4800 | 0.4460 |
| Quantum VQC (Phase 6 Original) | 0.1940 | 0.1130 | 0.2133 | 0.1435 |
| Late Fusion (Phase 7) | 0.4372 | 0.4614 | 0.4800 | 0.4668 |

## 3. Leakage Controls Verified
- Dataset maintained strict 239/60/75 disjoint split.
- Preprocessor (imputation + scaling) fit purely on the 239 training samples.
- OOF predictions strictly isolated folds during fusion training.
- Hyperparameter tuning and model selection used exclusively validation data.
- The 75-sample test set was evaluated only once per frozen model.

## 4. Explainability & Ablation Findings
- **SHAP:** Confirmed classical LR leverages clinically relevant features like `hemo` and `sg` without target leakage.
- **Quantum Sensitivity:** Demonstrated non-linear responsiveness of the VQC to input perturbations.
- **Ablation:** Fusion (+0.0146 Test Macro-F1) indicates the quantum branch offers limited complementary information to the strong classical baseline.

## 5. Limitations & Scientific Conclusion
- **Classical Superiority:** Classical Logistic Regression is substantially stronger than standalone VQC on this tabular task.
- **Modest Fusion Gain:** Late fusion shows a modest observed improvement over LR (+0.0146). 
- **No Quantum Advantage:** This does NOT establish quantum advantage. The VQC acted as a weak complementary signal provider.
- **Exploratory Status:** Results are exploratory. The categories are NOT clinical CKD diagnoses.

**Conclusion:** The project successfully implemented and rigorously evaluated a leakage-free hybrid quantum-classical pipeline. Phases 1-8 artifacts were perfectly preserved. The final evaluation is scientifically sound, reproducible, and appropriately conservative.

**PROJECT COMPLETE.**
