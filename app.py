
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression
from qiskit.circuit.library import zz_feature_map, n_local
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA, OptimizerResult
from qiskit.primitives import StatevectorSampler


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

FEATURES = ["bgr", "bu", "pot", "hemo", "pcv", "rbcc", "sg", "sod"]
NUM_FEATURES = 8
NUM_CLASSES = 5
RANDOM_SEED = 42


# ============================================================
# LOAD CLASSICAL MODEL
# ============================================================

@st.cache_resource
def load_classical_model():
    path = os.path.join(PROJECT_ROOT, "models", "best_classical_model.pkl")

    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_preprocessor():
    path = os.path.join(
        PROJECT_ROOT,
        "models",
        "classical_preprocessor.pkl"
    )

    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_input(values, pp):
    X = pd.DataFrame([values], columns=FEATURES)

    numeric_cols = pp["numeric_cols"]

    X[numeric_cols] = pp["numeric_imputer"].transform(
        X[numeric_cols]
    )

    X[numeric_cols] = pp["scaler"].transform(
        X[numeric_cols]
    )

    return X.values.astype(np.float64)


# ============================================================
# VQC RECONSTRUCTION
# ============================================================

@st.cache_resource
def load_vqc():

    weights_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "quantum_vqc_weights.npy"
    )

    frozen_weights = np.load(weights_path)

    fm = zz_feature_map(
        feature_dimension=NUM_FEATURES,
        reps=1
    )

    ansatz = n_local(
        num_qubits=NUM_FEATURES,
        rotation_blocks=["ry"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=1
    )

    vqc = VQC(
        feature_map=fm,
        ansatz=ansatz,
        loss="cross_entropy",
        optimizer=COBYLA(maxiter=1),
        sampler=StatevectorSampler(),
        initial_point=frozen_weights
    )

    # Load training data only to bootstrap the 5-class output head.
    data_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "ckd_egfr_categories_observed_only.csv"
    )

    df = pd.read_csv(data_path)

    X = df[FEATURES].copy()
    y = df["egfr_category"].astype(int)

    pp = load_preprocessor()
    numeric_cols = pp["numeric_cols"]

    X[numeric_cols] = pp["numeric_imputer"].transform(
        X[numeric_cols]
    )

    X[numeric_cols] = pp["scaler"].transform(X[numeric_cols])

    X_np = X.values.astype(np.float64)

    # Encode categories 1-5 -> 0-4
    y_np = y.values - 1

    # One representative sample from each class
    seed_indices = []

    for cls in range(NUM_CLASSES):
        indices = np.where(y_np == cls)[0]

        if len(indices) == 0:
            raise RuntimeError(
                f"Missing class {cls + 1} in bootstrap dataset."
            )

        seed_indices.append(indices[0])

    seed_indices = np.array(seed_indices)

    np.random.seed(RANDOM_SEED)

    # Minimal bootstrap fit.
    vqc.fit(
        X_np[seed_indices],
        y_np[seed_indices]
    )

    # Inject the actual frozen Phase 6 weights.
    result = OptimizerResult()
    result.x = frozen_weights
    result.fun = 0.0

    vqc._fit_result = result

    return vqc


# ============================================================
# PREDICTION
# ============================================================

def _run_prediction_isolated(values):
    """
    Executes the full preprocessing, classical inference, VQC simulation,
    and late-fusion meta-classification.
    
    Running this in an isolated worker process ensures Qiskit's native
    C/Rust mimalloc routines execute on a MainThread, preventing fatal
    segmentation faults caused by thread-local storage mismatches in
    Python 3.14 on macOS when called from Streamlit ScriptRunner threads.
    """
    pp = load_preprocessor()
    lr = load_classical_model()
    vqc = load_vqc()

    X_np = preprocess_input(values, pp)

    # Classical probabilities
    lr_proba = lr.predict_proba(X_np)

    # Quantum probabilities
    vqc_proba = vqc.predict_proba(X_np)

    # Late-fusion meta features
    meta_features = np.hstack([
        lr_proba,
        vqc_proba
    ])

    # Load frozen fusion classifier
    fusion_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "fusion_meta_classifier.pkl"
    )

    with open(fusion_path, "rb") as f:
        fusion_model = pickle.load(f)

    fusion_proba = fusion_model.predict_proba(
        meta_features
    )

    prediction = fusion_model.predict(
        meta_features
    )[0]

    return (
        int(prediction) + 1,
        np.array(lr_proba[0]),
        np.array(vqc_proba[0]),
        np.array(fusion_proba[0])
    )


def predict(values):
    """
    Dispatches prediction to an isolated process to guarantee thread safety
    and prevent native mimalloc segfaults in Qiskit on macOS.
    """
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_prediction_isolated, values)
        return future.result(timeout=60)


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Hybrid CKD Severity Predictor",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Hybrid Quantum-Classical CKD Predictor")

st.markdown(
    """
### eGFR-Based Severity Category Prediction

This demo uses the **frozen research models** from the project:

- Classical Logistic Regression
- 8-qubit Variational Quantum Classifier
- Late-fusion meta-classifier
- Train-only preprocessing

The prediction target is an **exploratory eGFR-based severity category (1–5)**,
not a clinically confirmed CKD stage.
"""
)

st.divider()

st.subheader("Enter Patient Features")

st.caption(
    "Only the 8 leakage-controlled Experiment B features are used."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    bgr = st.number_input(
        "Blood Glucose (bgr)",
        min_value=0.0,
        value=120.0
    )

    bu = st.number_input(
        "Blood Urea (bu)",
        min_value=0.0,
        value=40.0
    )

with col2:
    pot = st.number_input(
        "Potassium (pot)",
        min_value=0.0,
        value=4.5
    )

    hemo = st.number_input(
        "Hemoglobin (hemo)",
        min_value=0.0,
        value=13.0
    )

with col3:
    pcv = st.number_input(
        "Packed Cell Volume (pcv)",
        min_value=0.0,
        value=40.0
    )

    rbcc = st.number_input(
        "Red Blood Cell Count (rbcc)",
        min_value=0.0,
        value=4.5
    )

with col4:
    sg = st.number_input(
        "Specific Gravity (sg)",
        min_value=1.0,
        max_value=1.1,
        value=1.020,
        step=0.005,
        format="%.3f"
    )

    sod = st.number_input(
        "Sodium (sod)",
        min_value=0.0,
        value=140.0
    )

st.divider()

if st.button(
    "🔬 Predict Severity Category",
    type="primary",
    use_container_width=True
):

    values = {
        "bgr": bgr,
        "bu": bu,
        "pot": pot,
        "hemo": hemo,
        "pcv": pcv,
        "rbcc": rbcc,
        "sg": sg,
        "sod": sod
    }

    with st.spinner("Running classical + quantum inference..."):
        try:

            prediction, lr_proba, vqc_proba, fusion_proba = predict(
                values
            )

            st.success(
                f"Predicted eGFR-Based Severity Category: **Category {prediction}**"
            )

            st.info(
                "This is an exploratory machine-learning prediction and "
                "is not a medical diagnosis."
            )

            st.subheader("Model Probabilities")

            result_df = pd.DataFrame({
                "Category": [
                    "Category 1",
                    "Category 2",
                    "Category 3",
                    "Category 4",
                    "Category 5"
                ],
                "Classical LR": lr_proba,
                "Quantum VQC": vqc_proba,
                "Late Fusion": fusion_proba
            })

            st.dataframe(
                result_df.style.format({
                    "Classical LR": "{:.3f}",
                    "Quantum VQC": "{:.3f}",
                    "Late Fusion": "{:.3f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Prediction Confidence")

            st.bar_chart(
                result_df.set_index("Category")["Late Fusion"]
            )

        except Exception as e:

            st.error("Prediction failed.")

            st.exception(e)


st.divider()

st.caption(
    "Research prototype • Qiskit 2.5.2 • "
    "Qiskit Machine Learning 0.9.1 • Statevector simulation • "
    "No quantum advantage claimed"
)
