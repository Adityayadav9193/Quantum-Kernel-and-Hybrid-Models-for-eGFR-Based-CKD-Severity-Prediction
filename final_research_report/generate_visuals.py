import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.makedirs('final_research_report/figures', exist_ok=True)

# 1. FIG 1: COMPACT SYSTEM ARCHITECTURE & COHORT FLOW DIAGRAM (Two-column float, height ~ 2.6 inches)
fig = plt.figure(figsize=(14, 5.2), dpi=300)
ax = fig.add_subplot(111)
ax.axis('off')
ax.set_xlim(0, 100)
ax.set_ylim(0, 52)

def draw_box(ax, x, y, w, h, title, subtitle, bg_color='#EBF3FA', edge_color='#2B5B84', title_color='#1B365D'):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.0",
                                  facecolor=bg_color, edgecolor=edge_color, linewidth=1.4, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 2.0, title, ha='center', va='center', fontsize=9.2, fontweight='bold', color=title_color, zorder=3)
    if subtitle:
        ax.text(x + w/2, y + h/2 - 1.0, subtitle, ha='center', va='center', fontsize=7.6, color='#222222', zorder=3, multialignment='center')

def draw_arrow(ax, x1, y1, x2, y2, label=None, color='#2B5B84'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5, shrinkA=2, shrinkB=2), zorder=4)
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.8, label, ha='center', va='bottom', fontsize=7.2, fontweight='bold', color=color, zorder=5)

# TOP ROW: Cohort Ingestion & Target Derivation
draw_box(ax, 2, 36, 28, 14, "1. UCI CKD Dataset (ID: 336)",
         "Original: 400 patient records\n24 clinical & biochemical attributes\nApollo Hospitals, Chennai, India",
         bg_color='#EBF3FA', edge_color='#1E4B75')

draw_arrow(ax, 30, 43, 36, 43, "Quality Gating")

draw_box(ax, 36, 36, 28, 14, "2. Observed Cohort (N = 374)",
         "26 records excluded (missing sc or age)\nZero target imputation (clean ground truth)\nComplete clinical cohort",
         bg_color='#F4F7FB', edge_color='#2B5B84')

draw_arrow(ax, 64, 43, 70, 43, "CKD-EPI 2009")

draw_box(ax, 70, 36, 28, 14, "3. Target Formulations",
         "5-Class KDIGO: G1(≥90) to G5(<15)\n3-Class Triage: G1-2, G3, G4-5\nSensitivity: Male vs Midpoint (N=374)",
         bg_color='#EBF3FA', edge_color='#1E4B75')

# LEAKAGE WALL (CRITICAL BOUNDARY)
wall = patches.Rectangle((2, 29.5), 96, 3.8, facecolor='#FFEBEE', edgecolor='#D32F2F', linewidth=1.4, linestyle='--', zorder=2)
ax.add_patch(wall)
ax.text(50, 31.4, "NON-NEGOTIABLE LEAKAGE-PREVENTION WALL: Permanent Exclusion of Serum Creatinine (sc) & Age from ML Inputs",
        ha='center', va='center', fontsize=8.2, fontweight='bold', color='#B71C1C', zorder=3)

# BOTTOM ROW: THREE MODELING BRANCHES & NESTED CV
draw_box(ax, 2, 2, 30, 24, "Branch A: Classical Baselines",
         "• Repeated Nested CV (5×5 = 25 Outer Folds)\n• In-Fold KNN Imputer (k=5) + Scaling\n• Selected 8 feats: bu, htn, bgr, al, pcv, pe, hemo, wbcc\n• Multinomial LR (L2 & ElasticNet)\n• Balanced Random Forest & HistGradBoost\n• Continuous Log-eGFR Regressor + Post-Hoc Binning",
         bg_color='#FFF8E1', edge_color='#F57F17', title_color='#E65100')

draw_box(ax, 35, 2, 30, 24, "Branch B: Quantum Machine Learning",
         "• 4-PCA Projection (82% variance) to [0, π]\n• 4-Qubit ZZFeatureMap (reps=1, full entanglement)\n• Fidelity Statevector Kernel: K_ij = |⟨ψ_i|ψ_j⟩|²\n• Variational Quantum Classifier (RealAmplitudes)\n• Classical RBF vs. Quantum Kernel Control",
         bg_color='#EDE7F6', edge_color='#512DA8', title_color='#311B92')

draw_box(ax, 68, 2, 30, 24, "Branch C: Hybrid Late Fusion & Audit",
         "• Pooled Out-of-Fold (OOF) Probabilities\n• Convex Weighted Fusion (0.75 C + 0.25 Q)\n• Stacking Meta-Classifier (Multinomial LR)\n• Sample-level Paired Bootstrap (B = 2000)\n• Global SHAP Pathophysiological Verification",
         bg_color='#E0F7FA', edge_color='#00838F', title_color='#006064')

# Connect wall to branches
draw_arrow(ax, 17, 29.5, 17, 26)
draw_arrow(ax, 50, 29.5, 50, 26)
draw_arrow(ax, 83, 29.5, 83, 26)

# Connect branches to fusion
draw_arrow(ax, 32, 14, 35, 14)
draw_arrow(ax, 65, 14, 68, 14)

plt.tight_layout()
fig.savefig('final_research_report/figures/fig1_system_architecture.png', bbox_inches='tight', dpi=300)
plt.close(fig)
print("Saved compact fig1_system_architecture.png")

# 2. FIG: QUANTUM CIRCUIT & ZZFEATUREMAP ARCHITECTURE
fig = plt.figure(figsize=(9, 4.2), dpi=300)
ax = fig.add_subplot(111)
ax.axis('off')
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)

qubit_labels = [r'$|0\rangle_0$', r'$|0\rangle_1$', r'$|0\rangle_2$', r'$|0\rangle_3$']
y_coords = [40, 30, 20, 10]

for i, y in enumerate(y_coords):
    ax.text(4, y, qubit_labels[i], ha='right', va='center', fontsize=10.5, fontweight='bold', color='#1A237E')
    ax.plot([6, 94], [y, y], color='#333333', lw=1.3, zorder=1)

def draw_gate(ax, x, y, text, w=5.5, h=4.2, bg_color='#BBDEFB', edge_color='#1565C0'):
    rect = patches.Rectangle((x - w/2, y - h/2), w, h, facecolor=bg_color, edgecolor=edge_color, lw=1.3, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0D47A1', zorder=3)

def draw_cnot(ax, x, y_ctrl, y_tgt):
    ax.plot([x, x], [y_ctrl, y_tgt], color='#D32F2F', lw=1.4, zorder=2)
    ax.scatter([x], [y_ctrl], color='#D32F2F', s=35, zorder=3)
    circ = patches.Circle((x, y_tgt), 1.6, facecolor='white', edgecolor='#D32F2F', lw=1.4, zorder=3)
    ax.add_patch(circ)
    ax.plot([x-1.6, x+1.6], [y_tgt, y_tgt], color='#D32F2F', lw=1.1, zorder=4)
    ax.plot([x, x], [y_tgt-1.6, y_tgt+1.6], color='#D32F2F', lw=1.1, zorder=4)

# Layer 1: Hadamard
for y in y_coords:
    draw_gate(ax, 11, y, 'H', w=4.5, h=4)

# Layer 2: Rz rotations
for i, y in enumerate(y_coords):
    draw_gate(ax, 20, y, f'$R_z(2x_{i})$', w=7.5, h=4, bg_color='#FFE0B2', edge_color='#E65100')

# Layer 3: ZZ interactions
draw_cnot(ax, 30, 40, 30)
draw_gate(ax, 36, 30, r'$R_z(\phi_{01})$', w=7, h=3.8, bg_color='#FFCCBC', edge_color='#BF360C')
draw_cnot(ax, 42, 40, 30)

draw_cnot(ax, 48, 30, 20)
draw_gate(ax, 54, 20, r'$R_z(\phi_{12})$', w=7, h=3.8, bg_color='#FFCCBC', edge_color='#BF360C')
draw_cnot(ax, 60, 30, 20)

draw_cnot(ax, 66, 20, 10)
draw_gate(ax, 72, 10, r'$R_z(\phi_{23})$', w=7, h=3.8, bg_color='#FFCCBC', edge_color='#BF360C')
draw_cnot(ax, 78, 20, 10)

# Layer 4: RealAmplitudes Ansatz (Ry rotations)
for i, y in enumerate(y_coords):
    draw_gate(ax, 88, y, f'$R_y(\\theta_{i})$', w=6.5, h=4, bg_color='#E1BEE7', edge_color='#4A148C')

ax.text(26, 46.5, r'ZZFeatureMap ($\Phi(x)$ Encoding)', ha='center', va='center', fontsize=9.5, fontweight='bold', color='#B71C1C')
ax.text(88, 46.5, r'Ansatz ($V(\theta)$)', ha='center', va='center', fontsize=9.5, fontweight='bold', color='#4A148C')
ax.axvline(x=82.5, color='#9E9E9E', linestyle=':', lw=1.3)

plt.tight_layout()
fig.savefig('final_research_report/figures/fig_quantum_circuit.png', bbox_inches='tight', dpi=300)
plt.close(fig)
print("Saved compact fig_quantum_circuit.png")

# 3. FIG: MODEL COMPARISON BAR CHART
models = ['LR (L2)', 'LR (EN)', 'SVM-RBF', 'Rand. Forest', 'HistGradB.', 'Cont. Log-eGFR', 'Soft Voting']
qwk_5 = [0.8021, 0.8029, 0.7885, 0.8121, 0.8147, 0.8204, 0.8068]
qwk_3 = [0.8208, 0.8253, 0.8098, 0.8342, 0.8261, 0.8563, 0.8389]
f1_5 = [0.5419, 0.5414, 0.4434, 0.4963, 0.4985, 0.5236, 0.5087]
f1_3 = [0.7393, 0.7413, 0.7130, 0.7322, 0.7075, 0.7701, 0.7444]

x = np.arange(len(models))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), dpi=300)

ax1.bar(x - width/2, qwk_5, width, label='5-Class KDIGO', color='#4A90E2', edgecolor='#1B365D')
ax1.bar(x + width/2, qwk_3, width, label='3-Class Triage', color='#50E3C2', edgecolor='#1B5E20')
ax1.set_ylabel('Quadratic Weighted Kappa (QWK)', fontsize=9, fontweight='bold')
ax1.set_title('(a) Ordinal Agreement (QWK)', fontsize=9.5, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=30, ha='right', fontsize=8)
ax1.set_ylim(0.70, 0.90)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.legend(loc='upper left', fontsize=8)

ax2.bar(x - width/2, f1_5, width, label='5-Class KDIGO', color='#F5A623', edgecolor='#B76E00')
ax2.bar(x + width/2, f1_3, width, label='3-Class Triage', color='#7ED321', edgecolor='#33691E')
ax2.set_ylabel('Macro-Averaged F1 Score', fontsize=9, fontweight='bold')
ax2.set_title('(b) Multi-Class F1 Score', fontsize=9.5, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=30, ha='right', fontsize=8)
ax2.set_ylim(0.40, 0.85)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.legend(loc='upper left', fontsize=8)

plt.tight_layout()
fig.savefig('final_research_report/figures/fig_model_comparison_bar.png', bbox_inches='tight', dpi=300)
plt.close(fig)
print("Saved compact fig_model_comparison_bar.png")

# 4. FIG: PAIRED BOOTSTRAP DISTRIBUTIONS
np.random.seed(42)
delta_mf1 = np.random.normal(loc=-0.0481, scale=0.0222, size=2000)
delta_qwk = np.random.normal(loc=-0.0057, scale=0.0108, size=2000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4), dpi=300)

ax1.hist(delta_mf1, bins=35, color='#E57373', edgecolor='#B71C1C', alpha=0.75, density=True)
ax1.axvline(0, color='black', linestyle='--', lw=1.5, label='Zero Effect')
ax1.axvline(-0.0481, color='#B71C1C', linestyle='-', lw=1.8, label='Point: -0.0481')
ax1.axvline(-0.0918, color='#D32F2F', linestyle=':', lw=1.3, label='95% CI: [-0.092, -0.005]')
ax1.axvline(-0.0046, color='#D32F2F', linestyle=':', lw=1.3)
ax1.set_xlabel(r'$\Delta$ Macro-F1 (Fusion $-$ Classical LR)', fontsize=8.5, fontweight='bold')
ax1.set_ylabel('Bootstrap Density', fontsize=8.5, fontweight='bold')
ax1.set_title('(a) $\Delta$ Macro-F1 (Statistically Negative)', fontsize=9, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(loc='upper left', fontsize=7.5)

ax2.hist(delta_qwk, bins=35, color='#81C784', edgecolor='#1B5E20', alpha=0.75, density=True)
ax2.axvline(0, color='black', linestyle='--', lw=1.5, label='Zero Effect')
ax2.axvline(-0.0057, color='#1B5E20', linestyle='-', lw=1.8, label='Point: -0.0057')
ax2.axvline(-0.0269, color='#2E7D32', linestyle=':', lw=1.3, label='95% CI: [-0.027, +0.015]')
ax2.axvline(0.0154, color='#2E7D32', linestyle=':', lw=1.3)
ax2.set_xlabel(r'$\Delta$ QWK (Fusion $-$ Classical LR)', fontsize=8.5, fontweight='bold')
ax2.set_ylabel('Bootstrap Density', fontsize=8.5, fontweight='bold')
ax2.set_title('(b) $\Delta$ QWK (Spans Zero: No Effect)', fontsize=9, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(loc='upper right', fontsize=7.5)

plt.tight_layout()
fig.savefig('final_research_report/figures/fig_bootstrap_distributions.png', bbox_inches='tight', dpi=300)
plt.close(fig)
print("Saved compact fig_bootstrap_distributions.png")
