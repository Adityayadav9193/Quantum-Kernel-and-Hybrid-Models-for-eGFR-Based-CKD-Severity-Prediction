import os, sys, json, pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

try:
    import shap
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    import shap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from qiskit.circuit.library import zz_feature_map, n_local
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA, OptimizerResult
from qiskit.primitives import StatevectorSampler

os.makedirs(os.path.join(PROJECT_ROOT, "results", "phase8"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "results", "figures", "phase8"), exist_ok=True)

# 1. Load Data
df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "ckd_egfr_categories_observed_only.csv"))
FEATURES = ['bgr','bu','pot','hemo','pcv','rbcc','sg','sod']
X = df[FEATURES]
y = df['egfr_category']

X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.20, random_state=42, stratify=y_tv)

with open(os.path.join(PROJECT_ROOT, 'models', 'classical_preprocessor.pkl'), 'rb') as f:
    pp = pickle.load(f)

nc = pp['numeric_cols']
def transform(Xd):
    Xc = Xd.copy()
    Xc[nc] = pp['numeric_imputer'].transform(Xc[nc])
    Xc[nc] = pp['scaler'].transform(Xc[nc])
    return Xc.values.astype(float)

X_train_np = transform(X_train)
X_test_np = transform(X_test)

le = pickle.load(open(os.path.join(PROJECT_ROOT, 'models', 'quantum_label_encoder.pkl'), 'rb'))
y_test_enc = le.transform(y_test)
y_train_enc = le.transform(y_train)

# 2. SHAP for LR
with open(os.path.join(PROJECT_ROOT, 'models', 'best_classical_model.pkl'), 'rb') as f:
    lr = pickle.load(f)

explainer = shap.LinearExplainer(lr, X_train_np)
shap_values = explainer.shap_values(X_train_np)

plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_train_np, feature_names=FEATURES, show=False)
plt.title("SHAP Summary - Classical Logistic Regression")
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "results", "figures", "phase8", "shap_summary.png"))
plt.close()

# 3. Quantum Feature Sensitivity
frozen_weights = np.load(os.path.join(PROJECT_ROOT, 'models', 'quantum_vqc_weights.npy'))
seed_indices = []
for cls in range(5):
    idx = np.where(y_train_enc == cls)[0]
    if len(idx) > 0: seed_indices.append(idx[0])
seed_indices = np.array(seed_indices)

fm = zz_feature_map(feature_dimension=8, reps=1)
ansatz = n_local(num_qubits=8, rotation_blocks=['ry'], entanglement_blocks='cx', entanglement='linear', reps=1)
vqc = VQC(feature_map=fm, ansatz=ansatz, loss='cross_entropy', optimizer=COBYLA(maxiter=1), sampler=StatevectorSampler(), initial_point=frozen_weights)

np.random.seed(42)
vqc.fit(X_train_np[seed_indices], y_train_enc[seed_indices])
res = OptimizerResult()
res.x = frozen_weights
res.fun = 0.0
vqc._fit_result = res

base_probs = vqc.predict_proba(X_test_np)
sensitivity = {}
for i, feat in enumerate(FEATURES):
    X_pert = X_test_np.copy()
    X_pert[:, i] += 1.0 # +1 std dev
    pert_probs = vqc.predict_proba(X_pert)
    change = np.mean(np.abs(pert_probs - base_probs))
    sensitivity[feat] = float(change)

plt.figure(figsize=(8,5))
sns.barplot(x=list(sensitivity.keys()), y=list(sensitivity.values()), color="#4C72B0")
plt.title("Quantum Feature Sensitivity (Mean Abs Prob Change for +1 Std Dev)")
plt.ylabel("Mean Abs Prob Change")
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "results", "figures", "phase8", "quantum_sensitivity.png"))
plt.close()

# 4. Error Analysis
with open(os.path.join(PROJECT_ROOT, 'models', 'fusion_meta_classifier.pkl'), 'rb') as f:
    meta = pickle.load(f)
test_lr_proba = np.load(os.path.join(PROJECT_ROOT, "results", "fusion_predictions", "test_lr_proba.npy"))
test_vqc_proba = np.load(os.path.join(PROJECT_ROOT, "results", "fusion_predictions", "test_vqc_proba.npy"))
X_meta_test = np.hstack([test_lr_proba, test_vqc_proba])
test_fus_pred = meta.predict(X_meta_test)

errors_fus = (test_fus_pred != y_test_enc)
err_indices = np.where(errors_fus)[0]

error_details = []
for idx in err_indices:
    error_details.append({
        "true_class": int(y_test_enc[idx]) + 1,
        "pred_class": int(test_fus_pred[idx]) + 1
    })

phase8_results = {
    "quantum_sensitivity": sensitivity,
    "error_analysis": {
        "total_test_samples": 75,
        "total_errors": int(np.sum(errors_fus)),
        "error_details": error_details
    }
}
with open(os.path.join(PROJECT_ROOT, "results", "phase8", "phase8_results.json"), "w") as f:
    json.dump(phase8_results, f, indent=2)

print("Phase 8 analysis complete.")
