"""
Multi-stage CKD label derivation script.
This script derives multi-stage CKD labels from eGFR calculations.
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.clinical.egfr import EGFRCalculator, validate_egfr_inputs

def derive_multistage_labels(df, method='combined_approach'):
    """
    Derive multi-stage CKD labels from eGFR calculations.
    
    Args:
        df: Cleaned DataFrame with clinical data
        method: eGFR calculation method
        
    Returns:
        DataFrame with added eGFR and stage columns
    """
    print("="*60)
    print("MULTI-STAGE CKD LABEL DERIVATION")
    print("="*60)
    
    # Validate inputs
    is_valid, missing_vars, available_vars = validate_egfr_inputs(df)
    
    print("\nInput Validation:")
    print(f"  Available variables: {available_vars}")
    print(f"  Missing variables: {missing_vars}")
    
    if not is_valid:
        print("\nERROR: Required variables missing for eGFR calculation")
        return None
    
    # CRITICAL: Document the limitation
    print("\n" + "="*60)
    print("CRITICAL LIMITATION")
    print("="*60)
    print("The dataset does NOT contain sex information.")
    print("Standard CKD-EPI equation requires sex as a parameter.")
    print("Using MDRD equation without sex coefficient (male default).")
    print("This will OVERESTIMATE eGFR for females by ~25%.")
    print("This is a significant limitation and must be documented.")
    print("="*60)
    
    # Initialize calculator
    calculator = EGFRCalculator()
    
    # Calculate eGFR
    print("\nCalculating eGFR...")
    df['egfr'] = calculator.calculate_egfr(df['sc'], df['age'], method=method)
    
    # Assign CKD stages
    print("Assigning CKD stages...")
    df['ckd_stage'] = df['egfr'].apply(calculator.get_ckd_stage)
    
    # Add stage descriptions
    df['stage_description'] = df['ckd_stage'].apply(calculator.get_stage_description)
    
    return df

def analyze_stage_distribution(df):
    """
    Analyze the distribution of CKD stages.
    
    Args:
        df: DataFrame with stage labels
        
    Returns:
        Dictionary with stage distribution statistics
    """
    print("\n" + "="*60)
    print("STAGE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Stage counts
    stage_counts = df['ckd_stage'].value_counts().sort_index()
    stage_pct = (stage_counts / len(df)) * 100
    
    print("\nStage Distribution:")
    for stage in range(1, 6):
        count = stage_counts.get(stage, 0)
        pct = stage_pct.get(stage, 0)
        description = df[df['ckd_stage'] == stage]['stage_description'].iloc[0] if count > 0 else "No samples"
        print(f"  Stage {stage}: {count} samples ({pct:.2f}%)")
        print(f"    {description}")
    
    # Check for sparse stages
    print("\nSparse Stage Warning:")
    sparse_stages = [stage for stage in range(1, 6) if stage_counts.get(stage, 0) < 10]
    if sparse_stages:
        print(f"  WARNING: Sparse stages detected: {sparse_stages}")
        print(f"  This may affect model training for these stages.")
    else:
        print("  No sparse stages detected (all stages have ≥10 samples)")
    
    # eGFR statistics by stage
    print("\neGFR Statistics by Stage:")
    print(df.groupby('ckd_stage')['egfr'].describe())
    
    # Cross-tabulation with original binary label
    print("\nCross-tabulation with Original Binary Label:")
    crosstab = pd.crosstab(df['class'], df['ckd_stage'], margins=True)
    print(crosstab)
    
    return {
        'stage_counts': stage_counts,
        'stage_pct': stage_pct,
        'sparse_stages': sparse_stages,
        'crosstab': crosstab
    }

def create_stage_distribution_plot(df, save_path):
    """
    Create visualization of stage distribution.
    
    Args:
        df: DataFrame with stage labels
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Stage distribution
    plt.subplot(1, 2, 1)
    stage_counts = df['ckd_stage'].value_counts().sort_index()
    sns.barplot(x=stage_counts.index, y=stage_counts.values, palette='viridis')
    plt.title('CKD Stage Distribution')
    plt.xlabel('CKD Stage')
    plt.ylabel('Number of Samples')
    plt.xticks(range(1, 6))
    
    # eGFR distribution by stage
    plt.subplot(1, 2, 2)
    for stage in range(1, 6):
        stage_data = df[df['ckd_stage'] == stage]['egfr']
        if len(stage_data) > 0:
            plt.hist(stage_data, alpha=0.5, label=f'Stage {stage}', bins=20)
    plt.title('eGFR Distribution by Stage')
    plt.xlabel('eGFR (mL/min/1.73m²)')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nStage distribution plot saved to: {save_path}")
    plt.close()

def save_multistage_dataset(df, analysis_results):
    """
    Save the multi-stage dataset and analysis report.
    
    Args:
        df: DataFrame with multi-stage labels
        analysis_results: Dictionary with analysis results
    """
    os.makedirs(project_root + '/data/processed', exist_ok=True)
    os.makedirs(project_root + '/reports', exist_ok=True)
    os.makedirs(project_root + '/results/figures', exist_ok=True)
    
    # Save multi-stage dataset
    multistage_path = os.path.join(project_root, 'data/processed', 'ckd_multistage.csv')
    df.to_csv(multistage_path, index=False)
    print(f"\nMulti-stage dataset saved to: {multistage_path}")
    
    # Save analysis report
    report_path = os.path.join(project_root, 'reports', 'label_derivation.md')
    with open(report_path, 'w') as f:
        f.write("# Multi-Stage CKD Label Derivation Report\n\n")
        f.write("**Date:** 2026-09-14\n\n")
        f.write("## Methodology\n\n")
        f.write("### eGFR Calculation\n")
        f.write("- **Equation:** MDRD without sex coefficient (male default)\n")
        f.write("- **Variables Used:** Serum creatinine (sc), Age\n")
        f.write("- **CRITICAL LIMITATION:** Sex information not available in dataset\n")
        f.write("- **Impact:** eGFR overestimated for females by ~25%\n\n")
        
        f.write("### CKD Staging\n")
        f.write("- **Stage 1:** eGFR ≥ 90 mL/min/1.73m²\n")
        f.write("- **Stage 2:** eGFR 60-89 mL/min/1.73m²\n")
        f.write("- **Stage 3:** eGFR 30-59 mL/min/1.73m²\n")
        f.write("- **Stage 4:** eGFR 15-29 mL/min/1.73m²\n")
        f.write("- **Stage 5:** eGFR < 15 mL/min/1.73m²\n\n")
        
        f.write("### Clinical Note\n")
        f.write("eGFR alone is NOT sufficient to diagnose CKD. Clinical diagnosis requires\n")
        f.write("evidence of kidney damage (proteinuria, structural abnormalities, etc.) for Stages 1-2.\n\n")
        
        f.write("## Stage Distribution\n\n")
        f.write("| Stage | Count | Percentage | Description |\n")
        f.write("|-------|-------|-------------|-------------|\n")
        for stage in range(1, 6):
            count = analysis_results['stage_counts'].get(stage, 0)
            pct = analysis_results['stage_pct'].get(stage, 0)
            desc = df[df['ckd_stage'] == stage]['stage_description'].iloc[0] if count > 0 else "No samples"
            f.write(f"| {stage} | {count} | {pct:.2f}% | {desc} |\n")
        
        f.write("\n## Sparse Stages\n\n")
        if analysis_results['sparse_stages']:
            f.write(f"WARNING: Sparse stages detected: {analysis_results['sparse_stages']}\n")
            f.write("This may affect model training for these stages.\n")
        else:
            f.write("No sparse stages detected (all stages have ≥10 samples).\n")
        
        f.write("\n## Cross-tabulation with Original Binary Label\n\n")
        f.write(analysis_results['crosstab'].to_string())
        
        f.write("\n## Recommendations\n\n")
        if analysis_results['sparse_stages']:
            f.write("1. Consider merging sparse stages with adjacent stages\n")
            f.write("2. Use stratified sampling to ensure representation\n")
            f.write("3. Consider binary classification if multi-stage proves infeasible\n")
        f.write("4. Clearly document the sex limitation in all reports\n")
        f.write("5. Consider sex-independent validation if possible\n")
    
    print(f"Label derivation report saved to: {report_path}")
    
    # Create and save plot
    plot_path = os.path.join(project_root, 'results/figures', 'stage_distribution.png')
    create_stage_distribution_plot(df, plot_path)

def main():
    """Main execution function."""
    print("="*60)
    print("PHASE 2: MULTI-STAGE CKD LABEL DERIVATION")
    print("="*60)
    
    # Load cleaned dataset
    cleaned_path = os.path.join(project_root, 'data/processed', 'chronic_kidney_disease_cleaned.csv')
    df = pd.read_csv(cleaned_path)
    print(f"Loaded cleaned dataset: {df.shape}")
    
    # Derive multi-stage labels
    df = derive_multistage_labels(df)
    
    if df is None:
        print("Failed to derive multi-stage labels. Exiting.")
        return
    
    # Analyze stage distribution
    analysis_results = analyze_stage_distribution(df)
    
    # Save results
    save_multistage_dataset(df, analysis_results)
    
    print("\n" + "="*60)
    print("PHASE 2 COMPLETE")
    print("="*60)
    print("Multi-stage labels derived and saved.")
    print("Critical limitation documented: Sex information not available.")
    print("Next: Analyze whether to proceed with multi-stage or reduce to binary.")

if __name__ == "__main__":
    main()
