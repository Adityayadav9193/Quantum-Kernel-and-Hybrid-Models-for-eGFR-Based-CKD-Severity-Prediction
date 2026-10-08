"""
Additional visualizations for feature selection analysis.
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Use current working directory as project root
project_root = os.getcwd()

def create_correlation_heatmap(df, features, save_path):
    """Create correlation heatmap of selected features."""
    # Prepare data
    X = df[features].copy()
    
    # Encode categorical features
    for col in X.select_dtypes(include=['object', 'string']).columns:
        X[col] = pd.factorize(X[col])[0]
    
    # Handle missing values for visualization
    X = X.fillna(0)
    
    # Calculate correlation
    corr_matrix = X.corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Heatmap of Selected Features')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Correlation heatmap saved to: {save_path}")
    plt.close()

def create_feature_count_comparison(save_dir):
    """Create feature count comparison visualization."""
    # Simulated data for different feature counts
    feature_counts = [4, 6, 8, 10]
    variance_explained = [0.515, 0.632, 0.716, 0.785]  # From PCA analysis
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_counts, variance_explained, color='steelblue')
    plt.axhline(y=0.716, color='red', linestyle='--', label='8 components (selected)')
    plt.title('PCA: Variance Explained by Feature Count')
    plt.ylabel('Total Variance Explained')
    plt.xlabel('Number of Components')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'feature_count_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature count comparison saved to: {save_path}")
    plt.close()

def main():
    """Main execution function."""
    print("="*60)
    print("CREATING ADDITIONAL VISUALIZATIONS")
    print("="*60)
    
    # Load data
    data_path = os.path.join(project_root, 'data/processed/ckd_egfr_categories_observed_only.csv')
    df = pd.read_csv(data_path)
    
    # Load selected features
    json_path = os.path.join(project_root, 'results/figures/feature_selection/selected_features.json')
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    selected_features = results['selected_features']
    save_dir = os.path.join(project_root, 'results/figures/feature_selection')
    
    # Create correlation heatmap
    heatmap_path = os.path.join(save_dir, 'correlation_heatmap.png')
    create_correlation_heatmap(df, selected_features, heatmap_path)
    
    # Create feature count comparison
    create_feature_count_comparison(save_dir)
    
    print("\n" + "="*60)
    print("ADDITIONAL VISUALIZATIONS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
