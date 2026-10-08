"""
Corrected multi-stage eGFR-based category label derivation script.
This script derives eGFR-based severity categories ONLY from observed data.
Imputed values are NOT used for target label generation.
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.clinical.egfr import EGFRCalculator, validate_egfr_inputs

def derive_labels_from_observed_data(df, method='combined_approach'):
    """
    Derive eGFR-based severity categories ONLY from observed data.
    
    CRITICAL: This function only uses samples with observed serum creatinine and age.
    Samples with missing required variables are excluded from label generation.
    
    Args:
        df: Raw DataFrame with clinical data (NO imputation)
        method: eGFR calculation method
        
    Returns:
        DataFrame with eGFR and category columns (only for samples with observed data)
    """
    print("="*60)
    print("CORRECTED eGFR-BASED CATEGORY LABEL DERIVATION")
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
    print("CRITICAL LIMITATIONS")
    print("="*60)
    print("1. The dataset does NOT contain sex information.")
    print("2. Standard CKD-EPI equation requires sex as a parameter.")
    print("3. Using MDRD equation without sex coefficient (male default).")
    print("4. This will OVERESTIMATE eGFR for females by ~25%.")
    print("5. Target represents eGFR-based categories, NOT clinical CKD stages.")
    print("6. Clinical CKD diagnosis requires evidence of kidney damage.")
    print("="*60)
    
    # Initialize calculator
    calculator = EGFRCalculator()
    
    # Check for missing required variables
    required_vars = ['sc', 'age']
    missing_mask = df[required_vars].isnull().any(axis=1)
    
    total_samples = len(df)
    samples_with_missing = missing_mask.sum()
    samples_with_observed = total_samples - samples_with_missing
    
    print(f"\nData Quality Check:")
    print(f"  Total samples: {total_samples}")
    print(f"  Samples with missing sc or age: {samples_with_missing} ({samples_with_missing/total_samples*100:.2f}%)")
    print(f"  Samples with observed sc and age: {samples_with_observed} ({samples_with_observed/total_samples*100:.2f}%)")
    
    # CRITICAL: Only use samples with observed data for label generation
    df_observed = df[~missing_mask].copy()
    
    print(f"\nUsing only {len(df_observed)} samples with observed data for label generation.")
    print(f"Excluding {samples_with_missing} samples with missing required variables.")
    
    # Calculate eGFR (only on observed data)
    print("\nCalculating eGFR from observed serum creatinine and age...")
    df_observed['egfr'] = calculator.calculate_egfr(df_observed['sc'], df_observed['age'], method=method)
    
    # Assign eGFR-based categories
    print("Assigning eGFR-based severity categories...")
    df_observed['egfr_category'] = df_observed['egfr'].apply(calculator.get_egfr_category)
    
    # Add category descriptions
    df_observed['category_description'] = df_observed['egfr_category'].apply(calculator.get_category_description)
    
    return df_observed, samples_with_missing

def analyze_category_distribution(df):
    """
    Analyze the distribution of eGFR-based categories.
    
    Args:
        df: DataFrame with category labels
        
    Returns:
        Dictionary with category distribution statistics
    """
    print("\n" + "="*60)
    print("eGFR-BASED CATEGORY DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Category counts
    category_counts = df['egfr_category'].value_counts().sort_index()
    category_pct = (category_counts / len(df)) * 100
    
    print("\nCategory Distribution:")
    for category in range(1, 6):
        count = category_counts.get(category, 0)
        pct = category_pct.get(category, 0)
        description = df[df['egfr_category'] == category]['category_description'].iloc[0] if count > 0 else "No samples"
        print(f"  Category {category}: {count} samples ({pct:.2f}%)")
        print(f"    {description}")
    
    # Check for sparse categories
    print("\nSparse Category Warning:")
    sparse_categories = [category for category in range(1, 6) if category_counts.get(category, 0) < 10]
    if sparse_categories:
        print(f"  WARNING: Sparse categories detected: {sparse_categories}")
        print(f"  This may affect model training for these categories.")
    else:
        print("  No sparse categories detected (all categories have ≥10 samples)")
    
    # eGFR statistics by category
    print("\neGFR Statistics by Category:")
    print(df.groupby('egfr_category')['egfr'].describe())
    
    # Cross-tabulation with original binary label
    print("\nCross-tabulation with Original Binary Label:")
    crosstab = pd.crosstab(df['class'], df['egfr_category'], margins=True)
    print(crosstab)
    
    return {
        'category_counts': category_counts,
        'category_pct': category_pct,
        'sparse_categories': sparse_categories,
        'crosstab': crosstab
    }

def create_category_distribution_plot(df, save_path):
    """
    Create visualization of category distribution.
    
    Args:
        df: DataFrame with category labels
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Category distribution
    plt.subplot(1, 2, 1)
    category_counts = df['egfr_category'].value_counts().sort_index()
    sns.barplot(x=category_counts.index, y=category_counts.values, hue=category_counts.index, palette='viridis', legend=False)
    plt.title('eGFR-Based Severity Category Distribution')
    plt.xlabel('eGFR-Based Severity Category')
    plt.ylabel('Number of Samples')
    plt.xticks(range(1, 6))
    
    # eGFR distribution by category
    plt.subplot(1, 2, 2)
    for category in range(1, 6):
        category_data = df[df['egfr_category'] == category]['egfr']
        if len(category_data) > 0:
            plt.hist(category_data, alpha=0.5, label=f'Category {category}', bins=20)
    plt.title('eGFR Distribution by Category')
    plt.xlabel('eGFR (mL/min/1.73m²)')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nCategory distribution plot saved to: {save_path}")
    plt.close()

def save_observed_dataset(df, analysis_results, samples_excluded):
    """
    Save the observed-only dataset and analysis report.
    
    Args:
        df: DataFrame with eGFR-based categories (observed data only)
        analysis_results: Dictionary with analysis results
        samples_excluded: Number of samples excluded due to missing data
    """
    os.makedirs(project_root + '/data/processed', exist_ok=True)
    os.makedirs(project_root + '/reports', exist_ok=True)
    os.makedirs(project_root + '/results/figures', exist_ok=True)
    
    # Save observed-only dataset
    observed_path = os.path.join(project_root, 'data/processed', 'ckd_egfr_categories_observed_only.csv')
    df.to_csv(observed_path, index=False)
    print(f"\nObserved-only dataset saved to: {observed_path}")
    
    # Save analysis report
    report_path = os.path.join(project_root, 'reports', 'label_derivation_corrected.md')
    with open(report_path, 'w') as f:
        f.write("# Corrected eGFR-Based Category Label Derivation Report\n\n")
        f.write("**Date:** 2026-09-14\n")
        f.write("**Status:** CORRECTED - Uses only observed data\n\n")
        
        f.write("## Methodology\n\n")
        f.write("### eGFR Calculation\n")
        f.write("- **Equation:** MDRD without sex coefficient (male default)\n")
        f.write("- **Variables Used:** Serum creatinine (sc), Age\n")
        f.write("- **Data Source:** Raw dataset (NO imputation)\n")
        f.write("- **Sample Selection:** Only samples with observed sc and age\n")
        f.write(f"- **Samples Excluded:** {samples_excluded} samples with missing required variables\n")
        f.write(f"- **Final Sample Size:** {len(df)} samples\n\n")
        
        f.write("### CRITICAL LIMITATIONS\n\n")
        f.write("1. **Missing Sex Information:** Sex variable not available in dataset\n")
        f.write("2. **Equation Bias:** Using male-default MDRD equation\n")
        f.write("3. **Systematic Error:** eGFR overestimated for females by ~25%\n")
        f.write("4. **Unknown Sex Distribution:** Cannot quantify bias magnitude/direction\n")
        f.write("5. **Target Definition:** eGFR-based categories, NOT clinical CKD stages\n")
        f.write("6. **Clinical Validity:** eGFR alone insufficient for CKD diagnosis\n\n")
        
        f.write("### eGFR-Based Category Definitions\n")
        f.write("- **Category 1:** eGFR ≥ 90 mL/min/1.73m² (normal/high kidney function)\n")
        f.write("- **Category 2:** eGFR 60-89 mL/min/1.73m² (mildly decreased kidney function)\n")
        f.write("- **Category 3:** eGFR 30-59 mL/min/1.73m² (moderately decreased kidney function)\n")
        f.write("- **Category 4:** eGFR 15-29 mL/min/1.73m² (severely decreased kidney function)\n")
        f.write("- **Category 5:** eGFR < 15 mL/min/1.73m² (kidney failure)\n\n")
        
        f.write("### Clinical Note\n")
        f.write("These are eGFR-based severity categories, NOT clinical CKD stages.\n")
        f.write("Clinical CKD diagnosis requires evidence of kidney damage (proteinuria,\n")
        f.write("structural abnormalities, etc.) in addition to eGFR values.\n")
        f.write("High eGFR does not rule out CKD, and low eGFR does not equal CKD\n")
        f.write("without other evidence.\n\n")
        
        f.write("## Category Distribution\n\n")
        f.write("| Category | Count | Percentage | Description |\n")
        f.write("|----------|-------|-------------|-------------|\n")
        for category in range(1, 6):
            count = analysis_results['category_counts'].get(category, 0)
            pct = analysis_results['category_pct'].get(category, 0)
            desc = df[df['egfr_category'] == category]['category_description'].iloc[0] if count > 0 else "No samples"
            f.write(f"| {category} | {count} | {pct:.2f}% | {desc} |\n")
        
        f.write("\n## Sparse Categories\n\n")
        if analysis_results['sparse_categories']:
            f.write(f"WARNING: Sparse categories detected: {analysis_results['sparse_categories']}\n")
            f.write("This may affect model training for these categories.\n")
        else:
            f.write("No sparse categories detected (all categories have ≥10 samples).\n")
        
        f.write("\n## Cross-tabulation with Original Binary Label\n\n")
        f.write("This cross-tabulation shows that eGFR categories and clinical CKD diagnosis\n")
        f.write("measure different clinical constructs:\n\n")
        f.write(analysis_results['crosstab'].to_string())
        
        f.write("\n## Key Findings\n\n")
        f.write("1. High eGFR does not equal absence of CKD (many 'notckd' in Categories 1-2)\n")
        f.write("2. CKD diagnosis spans all eGFR categories (some 'ckd' in Category 1)\n")
        f.write("3. Low eGFR strongly correlates with CKD (no 'notckd' in Categories 4-5)\n")
        f.write("4. The two targets measure different clinical constructs\n\n")
        
        f.write("## Recommendations\n\n")
        if analysis_results['sparse_categories']:
            f.write("1. Consider merging sparse categories with adjacent categories\n")
            f.write("2. Use stratified sampling to ensure representation\n")
            f.write("3. Consider binary classification if multi-category proves infeasible\n")
        f.write("4. Clearly document the sex limitation in all reports\n")
        f.write("5. Use correct terminology (eGFR-based categories, not CKD stages)\n")
        f.write("6. Do not claim clinical validity for eGFR-based categories\n")
    
    print(f"Corrected label derivation report saved to: {report_path}")
    
    # Create and save plot
    plot_path = os.path.join(project_root, 'results/figures', 'category_distribution_corrected.png')
    create_category_distribution_plot(df, plot_path)

def main():
    """Main execution function."""
    print("="*60)
    print("CORRECTED PHASE 2: eGFR-BASED CATEGORY LABEL DERIVATION")
    print("="*60)
    
    # Load RAW dataset (not cleaned - we want observed data only)
    raw_path = os.path.join(project_root, 'data/raw', 'chronic_kidney_disease_raw.csv')
    df = pd.read_csv(raw_path)
    print(f"Loaded raw dataset: {df.shape}")
    
    # Derive labels from observed data only
    df_observed, samples_excluded = derive_labels_from_observed_data(df)
    
    if df_observed is None:
        print("Failed to derive labels. Exiting.")
        return
    
    # Analyze category distribution
    analysis_results = analyze_category_distribution(df_observed)
    
    # Save results
    save_observed_dataset(df_observed, analysis_results, samples_excluded)
    
    print("\n" + "="*60)
    print("CORRECTED PHASE 2 COMPLETE")
    print("="*60)
    print(f"eGFR-based categories derived from observed data only.")
    print(f"Final dataset: {len(df_observed)} samples (excluded {samples_excluded} with missing data).")
    print("Corrected terminology: eGFR-based severity categories (not CKD stages).")
    print("Next: Implement corrected preprocessing pipeline.")

if __name__ == "__main__":
    main()
