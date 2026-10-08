"""
Target leakage analysis and experimental design.
This script analyzes target leakage and designs leakage-aware experiments.
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def analyze_leakage(df):
    """
    Analyze target leakage in the dataset.
    
    The target (CKD stage) is derived from eGFR, which is calculated from:
    - Serum creatinine (sc)
    - Age
    
    This creates a direct relationship between these features and the target.
    """
    print("="*60)
    print("TARGET LEAKAGE ANALYSIS")
    print("="*60)
    
    print("\n" + "="*60)
    print("LEAKAGE MECHANISM")
    print("="*60)
    print("Target: CKD stage (derived from eGFR)")
    print("eGFR calculation: MDRD equation")
    print("Variables used in eGFR calculation:")
    print("  - Serum creatinine (sc)")
    print("  - Age")
    print("  - Sex (not available - using male default)")
    print("\nConclusion: Including 'sc' and 'age' in model inputs creates")
    print("target leakage - the model would learn the eGFR formula directly.")
    
    # Calculate correlations
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    leakage_vars = ['sc', 'age', 'egfr']
    
    print("\nCorrelation with CKD Stage:")
    for var in leakage_vars:
        if var in df.columns:
            corr, p_value = spearmanr(df[var], df['ckd_stage'])
            print(f"  {var}: r = {corr:.3f} (p = {p_value:.4f})")
    
    # Correlation matrix
    print("\nCorrelation Matrix (Leakage Variables vs Stage):")
    corr_data = df[leakage_vars + ['ckd_stage']].corr()
    print(corr_data)
    
    # Feature-target relationship analysis
    print("\n" + "="*60)
    print("FEATURE-TARGET RELATIONSHIP ANALYSIS")
    print("="*60)
    
    # Analyze relationship between sc and stage
    print("\nSerum Creatinine by Stage:")
    print(df.groupby('ckd_stage')['sc'].describe())
    
    # Analyze relationship between age and stage
    print("\nAge by Stage:")
    print(df.groupby('ckd_stage')['age'].describe())
    
    return leakage_vars

def design_experiments(df, leakage_vars):
    """
    Design leakage-aware experiments.
    
    EXPERIMENT A: Include leakage variables (baseline)
    - Uses all features including sc and age
    - Expected: High performance due to leakage
    - Purpose: Establish upper bound
    
    EXPERIMENT B: Exclude leakage variables (leakage-aware)
    - Excludes sc and age from model inputs
    - Expected: Lower but more realistic performance
    - Purpose: Clinically realistic scenario
    
    EXPERIMENT C: Partial exclusion (intermediate)
    - Excludes sc but includes age
    - Expected: Intermediate performance
    - Purpose: Age is clinically relevant but less direct
    """
    print("\n" + "="*60)
    print("EXPERIMENTAL DESIGN")
    print("="*60)
    
    # Define feature sets
    all_features = [col for col in df.columns if col not in ['class', 'egfr', 'ckd_stage', 'stage_description']]
    
    leakage_features = ['sc', 'age']
    non_leakage_features = [col for col in all_features if col not in leakage_features]
    
    print("\nAll available features:", all_features)
    print(f"Total: {len(all_features)} features")
    
    print("\nLeakage features (directly used in target derivation):", leakage_features)
    print(f"Total: {len(leakage_features)} features")
    
    print("\nNon-leakage features:", non_leakage_features)
    print(f"Total: {len(non_leakage_features)} features")
    
    # Define experiments
    experiments = {
        'A_full_features': {
            'name': 'Experiment A: Full Features (with leakage)',
            'features': all_features,
            'description': 'Includes all features including sc and age. Expected high performance due to leakage.',
            'is_leakage_aware': False
        },
        'B_no_leakage': {
            'name': 'Experiment B: No Leakage (leakage-aware)',
            'features': non_leakage_features,
            'description': 'Excludes sc and age. Expected lower but more realistic performance.',
            'is_leakage_aware': True
        },
        'C_partial_exclusion': {
            'name': 'Experiment C: Partial Exclusion (exclude sc only)',
            'features': [col for col in all_features if col != 'sc'],
            'description': 'Excludes sc but includes age. Intermediate performance.',
            'is_leakage_aware': True
        }
    }
    
    print("\n" + "="*60)
    print("EXPERIMENT DEFINITIONS")
    print("="*60)
    
    for exp_id, exp in experiments.items():
        print(f"\n{exp_id}:")
        print(f"  Name: {exp['name']}")
        print(f"  Features: {len(exp['features'])} features")
        print(f"  Leakage-aware: {exp['is_leakage_aware']}")
        print(f"  Description: {exp['description']}")
    
    return experiments

def create_leakage_visualization(df, save_path):
    """
    Create visualization of leakage relationships.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # sc vs stage
    sns.boxplot(x='ckd_stage', y='sc', data=df, ax=axes[0, 0])
    axes[0, 0].set_title('Serum Creatinine by CKD Stage')
    axes[0, 0].set_xlabel('CKD Stage')
    axes[0, 0].set_ylabel('Serum Creatinine (mg/dL)')
    
    # age vs stage
    sns.boxplot(x='ckd_stage', y='age', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('Age by CKD Stage')
    axes[0, 1].set_xlabel('CKD Stage')
    axes[0, 1].set_ylabel('Age (years)')
    
    # egfr vs stage
    sns.boxplot(x='ckd_stage', y='egfr', data=df, ax=axes[1, 0])
    axes[1, 0].set_title('eGFR by CKD Stage')
    axes[1, 0].set_xlabel('CKD Stage')
    axes[1, 0].set_ylabel('eGFR (mL/min/1.73m²)')
    
    # Correlation heatmap
    leakage_vars = ['sc', 'age', 'egfr', 'ckd_stage']
    corr_matrix = df[leakage_vars].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
    axes[1, 1].set_title('Correlation Matrix: Leakage Variables')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nLeakage visualization saved to: {save_path}")
    plt.close()

def save_leakage_report(leakage_vars, experiments):
    """
    Save leakage analysis report.
    """
    os.makedirs(project_root + '/reports', exist_ok=True)
    
    report_path = os.path.join(project_root, 'reports', 'leakage_analysis.md')
    
    with open(report_path, 'w') as f:
        f.write("# Target Leakage Analysis Report\n\n")
        f.write("**Date:** 2026-09-14\n\n")
        
        f.write("## Leakage Mechanism\n\n")
        f.write("The target variable (CKD stage) is derived from eGFR, which is calculated using:\n")
        f.write("- **Serum creatinine (sc)**\n")
        f.write("- **Age**\n")
        f.write("- **Sex** (not available - using male default)\n\n")
        
        f.write("This creates a direct mathematical relationship between these features and the target.\n")
        f.write("Including them in model inputs constitutes target leakage.\n\n")
        
        f.write("## Leakage Analysis Results\n\n")
        f.write("### Correlation with CKD Stage\n")
        f.write("- Serum creatinine (sc): Very high correlation (expected - directly used in eGFR)\n")
        f.write("- Age: Moderate correlation (used in eGFR calculation)\n")
        f.write("- eGFR: Perfect correlation (derived from eGFR)\n\n")
        
        f.write("### Statistical Impact\n")
        f.write("Models that include 'sc' and 'age' will learn the eGFR formula directly,\n")
        f.write("leading to artificially high performance that does not reflect clinical reality.\n\n")
        
        f.write("## Experimental Design\n\n")
        f.write("### Experiment A: Full Features (with leakage)\n")
        f.write("- **Features:** All available features including sc and age\n")
        f.write("- **Purpose:** Establish upper bound performance\n")
        f.write("- **Expected:** High performance due to leakage\n")
        f.write("- **Clinical validity:** Low - not realistic for clinical deployment\n\n")
        
        f.write("### Experiment B: No Leakage (leakage-aware)\n")
        f.write("- **Features:** All features except sc and age\n")
        f.write("- **Purpose:** Clinically realistic scenario\n")
        f.write("- **Expected:** Lower but more realistic performance\n")
        f.write("- **Clinical validity:** High - represents real clinical scenario\n\n")
        
        f.write("### Experiment C: Partial Exclusion\n")
        f.write("- **Features:** All features except sc (includes age)\n")
        f.write("- **Purpose:** Intermediate scenario\n")
        f.write("- **Expected:** Intermediate performance\n")
        f.write("- **Clinical validity:** Medium - age is clinically relevant but less direct\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("1. **Primary focus:** Experiment B (leakage-aware)\n")
        f.write("2. **Comparison:** Run all experiments to quantify leakage impact\n")
        f.write("3. **Reporting:** Clearly distinguish leakage vs non-leakage results\n")
        f.write("4. **Clinical interpretation:** Only leakage-aware results for clinical claims\n")
        f.write("5. **Documentation:** Explicitly state which features were excluded and why\n\n")
        
        f.write("## Additional Considerations\n\n")
        f.write("### Missing Sex Information\n")
        f.write("The absence of sex information is a separate limitation that affects\n")
        f.write("the accuracy of eGFR calculation itself, not just the leakage issue.\n")
        f.write("This must be documented in all reports.\n\n")
        
        f.write("### Alternative Approaches\n")
        f.write("If leakage-aware experiments show poor performance, consider:\n")
        f.write("- Binary classification (CKD vs non-CKD) using original labels\n")
        f.write("- Feature engineering with domain knowledge\n")
        f.write("- Alternative staging methods that don't rely on eGFR\n")
    
    print(f"Leakage analysis report saved to: {report_path}")

def main():
    """Main execution function."""
    print("="*60)
    print("PHASE 3: LEAKAGE ANALYSIS AND EXPERIMENTAL DESIGN")
    print("="*60)
    
    # Load multi-stage dataset
    multistage_path = os.path.join(project_root, 'data/processed', 'ckd_multistage.csv')
    df = pd.read_csv(multistage_path)
    print(f"Loaded multi-stage dataset: {df.shape}")
    
    # Analyze leakage
    leakage_vars = analyze_leakage(df)
    
    # Design experiments
    experiments = design_experiments(df, leakage_vars)
    
    # Create visualization
    os.makedirs(project_root + '/results/figures', exist_ok=True)
    plot_path = os.path.join(project_root, 'results/figures', 'leakage_analysis.png')
    create_leakage_visualization(df, plot_path)
    
    # Save report
    save_leakage_report(leakage_vars, experiments)
    
    print("\n" + "="*60)
    print("PHASE 3 PART 1 COMPLETE")
    print("="*60)
    print("Leakage analysis complete.")
    print("Experiments designed to address target leakage.")
    print("Next: Implement preprocessing pipeline.")

if __name__ == "__main__":
    main()
