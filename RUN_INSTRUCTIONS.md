# VS Code Local Execution Instructions

**Project:** Leakage-Aware Evaluation of Classical, Quantum-Kernel and Hybrid Models for eGFR-Based CKD Severity Prediction  
**Platform:** macOS (Apple Silicon / Intel) with VS Code terminal (zsh)  
**Python Runtime:** Python 3.14.2 (using the project's pre-configured `./venv`)  

---

## 1. Environment Activation & Configuration

Open your VS Code terminal (`Terminal` -> `New Terminal` or ``Ctrl+` ``) in the project workspace directory:

```bash
cd "/Users/adidoo/Documents/COLLEGE/7th Sem/Healthcare/project"
```

Activate the project virtual environment:

```bash
source venv/bin/activate
```

Configure your environment variables:
```bash
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/matplotlib
```

*Note: `export MPLCONFIGDIR=/tmp/matplotlib` prevents macOS directory permission warnings with Matplotlib font caching.*

Verify the Python interpreter and key dependencies:
```bash
python --version
# Expected: Python 3.14.2

python -c "import qiskit, qiskit_machine_learning, sklearn, pandas, shap; print('Environment dependencies verified successfully!')"
```

---

## 2. Complete Pipeline Execution (Step-by-Step)

Follow the intended pipeline order from raw data ingestion to final nested cross-validation and explainability.

### Step 1: Run Automated Integrity Tests
Verify data integrity, strict non-leakage constraints, train/validation/test isolation, and corrected nomenclature:

```bash
python tests/validate_data_only.py
python tests/validate_corrected_implementation.py
python tests/verify_phase4_corrected.py
python tests/test_phase5.py
```
*All tests should report `PASS`.*

---

### Step 2: Data Cleaning & Ground-Truth Label Derivation
Clean raw whitespace, handle missing values, and calculate ground-truth eGFR categories using the 2009 CKD-EPI equation on observed-only samples ($N = 374$):

```bash
python src/data/02_clean_dataset.py
python src/data/03_label_derivation_corrected.py
python src/data/04_leakage_analysis_corrected.py
```

*Outputs produced in `data/processed/`:*
* `ckd_egfr_categories_observed_only.csv` ($N = 374$, zero missing `sc` or `age`).

---

### Step 3: Leakage-Safe Preprocessing Pipeline
Verify and test the leakage-safe preprocessing pipeline (imputation, scaling, and categorical encoding fitted strictly on training data):

```bash
python src/preprocessing/05_test_preprocessing.py
```

*Artifact generated in `models/`:*
* `preprocessor_corrected.pkl`

---

### Step 4: Feature Selection & Clinical Ranking
Perform feature selection using Random Forest importance, Mutual Information, and 5-fold stability analysis on training data only (with `sc` and `age` excluded):

```bash
python src/feature_selection/01_feature_selection_corrected.py
python src/feature_selection/02_create_additional_visualizations.py
```

*Artifacts generated in `results/figures/feature_selection/`:*
* `selected_features_corrected.json` (top 8 stable features: `bgr`, `bu`, `pot`, `hemo`, `pcv`, `rbcc`, `sg`, `sod`)
* Importance, mutual information, and PCA variance plots.

---

### Step 5: Classical Baseline Models
Train and evaluate classical baselines (Logistic Regression, Random Forest, MLP) under class-weighted evaluation:

```bash
python src/classical/01_classical_baselines.py
python src/classical/02_create_visualizations.py
```

*Artifacts generated:*
* `models/best_classical_model.pkl`
* `models/classical_preprocessor.pkl`
* `results/classical_results.json`
* Confusion matrices and ROC curves in `results/figures/classical/`.

---

### Step 6: Parameterized Quantum Circuit (VQC) Implementation
Train and evaluate the 8-qubit Variational Quantum Classifier (`ZZFeatureMap` + `n_local` RY-CX ansatz, COBYLA optimizer, `StatevectorSampler`):

```bash
python src/quantum/01_quantum_model.py
```

*Artifacts generated:*
* `models/quantum_vqc_weights.npy`
* `models/quantum_label_encoder.pkl`
* `results/quantum_results.json`
* Loss convergence and confusion matrices in `results/figures/quantum/`.

---

### Step 7: Late Fusion & Out-of-Fold Stacking
Generate 5-fold out-of-fold probability distributions and evaluate late fusion (weighted probability combination and meta-classifier stacking):

```bash
python src/fusion/01_late_fusion.py
```

*Artifacts generated:*
* `models/fusion_meta_classifier.pkl`
* `results/fusion_results.json`
* Out-of-fold probability matrices in `results/fusion_predictions/`.

---

### Step 8: Model Explainability & SHAP Analysis
Generate SHAP beeswarm and summary feature attribution plots:

```bash
python src/analysis/01_phase8_analysis.py
```

*Artifacts generated in `results/figures/phase8/`:*
* SHAP summary and importance plots illustrating how Blood Urea, Albumin, and Hemoglobin drive ordinal risk.

---

### Step 9: Leakage-Free Nested Cross-Validation & Quantum Control Benchmark (Master Suite)
Execute the comprehensive, leak-free benchmark across all $N = 374$ patients:
* 5-fold $\times$ 5-repeat nested cross-validation (25 independent test folds).
* 5-Class and 3-Class targets (evaluating LR, Calibrated SVM, Balanced RF, HistGradientBoosting, and Continuous Log-eGFR Regressor).
* Quantum Control: Classical RBF Kernel vs Fidelity Statevector Quantum Kernel (KTA and SVC metrics).
* Sample-level paired bootstrap statistical significance test ($B = 2000$).
* Like-for-like label noise analysis ($N = 374$).
* Upper-bound ceiling sanity check (`sc` and `age` allowed).

```bash
python src/experiments/final_nested_cv.py
```

*Artifacts generated:*
* Complete raw fold data: `results/final_nested_cv.json`
* Full scientific report: `reports/final_report.md`
* Master documentation: `PROJECT_DOCUMENTATION.md`

---

## 3. Interactive Web Application (Streamlit)

To launch the local web interface for interactive multi-grade CKD risk prediction:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 4. Single-Command Full Workflow Run

To run the entire pipeline end-to-end in one automated command:

```bash
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/matplotlib
python tests/validate_data_only.py && \
python tests/validate_corrected_implementation.py && \
python tests/verify_phase4_corrected.py && \
python tests/test_phase5.py && \
python src/data/02_clean_dataset.py && \
python src/data/03_label_derivation_corrected.py && \
python src/preprocessing/05_test_preprocessing.py && \
python src/feature_selection/01_feature_selection_corrected.py && \
python src/classical/01_classical_baselines.py && \
python src/analysis/01_phase8_analysis.py && \
python src/experiments/final_nested_cv.py
```
