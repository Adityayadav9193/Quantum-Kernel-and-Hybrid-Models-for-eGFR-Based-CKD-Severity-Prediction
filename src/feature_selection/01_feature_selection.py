"""
Phase 4: Feature Selection Analysis
Uses corrected methodology with leakage-aware preprocessing.
Primary experiment: Experiment B (leakage-aware, excludes sc and age).
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Use current working directory as project root
project_root = os.getcwd()

def load_corrected_data():
    """Load the corrected observed-only dataset."""
    data_path = os.path.join(project_root, 'data/processed', 'ckd_egfr_categories_observed_only.csv')
    df = pd.read_csv(data_path)
    print(f"Loaded corrected dataset: {df.shape}")
    return df

def define_experiments(df):
    """Define the three experimental configurations."""
    all_features = [col for col in df.columns if col not in ['class', 'egfr', 'egfr_category', 'category_description']]
    leakage_features = ['sc', 'age']
    
    experiments = {
        'A_full_features': {
            'name': 'Experiment A: Full Features (with leakage)',
            'features': all_features,
            'excluded': [],
            'included_leakage': leakage_features,
            'leakage_aware': False,
            'clinical_validity': 'LOW'
        },
        'B_no_leakage': {
            'name': 'Experiment B: No Leakage (leakage-aware) - PRIMARY',
            'features': [f for f in all_features if f not in leakage_features],
            'excluded': leakage_features,
            'included_leakage': [],
            'leakage_aware': True,
            'clinical_validity': 'HIGH'
        },
        'C_partial_exclusion': {
            'name': 'Experiment C: Partial Exclusion (exclude sc only)',
            'features': [f for f in all_features if f != 'sc'],
            'excluded': ['sc'],
            'included_leakage': ['age'],
            'leakage_aware': True,
            'clinical_validity': 'MEDIUM'
        }
    }
    
    print("\nExperimental Configurations:")
    for exp_id, exp in experiments.items():
        print(f"  {exp_id}: {len(exp['features'])} features")
        print(f"    Excluded: {exp['excluded']}")
        print(f"    Leakage-aware: {exp['leakage_aware']}")
    
    return experiments

def prepare_data_for_feature_selection(df, features, target='egfr_category'):
    """
    Prepare data for feature selection with leakage control.
    
    CRITICAL: All feature selection must be done on training data only.
    """
    # Extract features and target
    X = df[features].copy()
    y = df[target].copy()
    
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    
    # Simple encoding for categorical columns (for feature selection only)
    X_encoded = X.copy()
    for col in categorical_cols:
        X_encoded[col] = pd.factorize(X_encoded[col])[0]
    
    # Handle missing values (other features may have missing values - that's OK)
    # For feature selection, we need to impute missing values
    if X_encoded.isnull().any().any():
        print("Warning: Missing values found in features, using median/mode imputation for feature selection")
        for col in numeric_cols:
            median_val = X_encoded[col].median()
            if pd.isna(median_val):
                median_val = 0
            X_encoded[col].fillna(median_val, inplace=True)
        for col in categorical_cols:
            # For categorical, use the most frequent value after encoding
            mode_val = X_encoded[col].mode()[0]
            if pd.isna(mode_val):
                mode_val = 0
            X_encoded[col].fillna(mode_val, inplace=True)
    
    # Ensure no NaN values remain
    if X_encoded.isnull().any().any():
        print("Filling remaining NaN values with 0")
        X_encoded = X_encoded.fillna(0)
    
    return X_encoded, y, numeric_cols, categorical_cols

def correlation_analysis(X, y, features, save_path):
    """Perform correlation analysis with target."""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Calculate correlation with target
    correlations = X.corrwith(y)
    
    # Sort by absolute correlation
    correlations_abs = correlations.abs().sort_values(ascending=False)
    
    print("Top features by correlation with target:")
    for feature in correlations_abs.head(15).index:
        corr_val = correlations[feature]
        print(f"  {feature}: {corr_val:.3f}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    correlations.sort_values(ascending=False).plot(kind='bar')
    plt.title('Feature Correlation with eGFR Category')
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Correlation plot saved to: {save_path}")
    plt.close()
    
    return correlations

def random_forest_importance(X_train, y_train, features, save_path):
    """Perform Random Forest feature importance analysis."""
    print("\n" + "="*60)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("="*60)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = pd.Series(rf.feature_importances_, index=features)
    
    # Sort by importance
    importances_sorted = importances.sort_values(ascending=False)
    
    print("Top features by Random Forest importance:")
    for feature in importances_sorted.head(15).index:
        imp_val = importances_sorted[feature]
        print(f"  {feature}: {imp_val:.4f}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    importances_sorted.head(15).plot(kind='bar')
    plt.title('Random Forest Feature Importance')
    plt.ylabel('Importance Score')
    plt.xlabel('Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"RF importance plot saved to: {save_path}")
    plt.close()
    
    return importances, rf

def mutual_information_analysis(X_train, y_train, features, save_path):
    """Perform mutual information analysis."""
    print("\n" + "="*60)
    print("MUTUAL INFORMATION ANALYSIS")
    print("="*60)
    
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    mi_series = pd.Series(mi_scores, index=features)
    
    # Sort by MI score
    mi_sorted = mi_series.sort_values(ascending=False)
    
    print("Top features by Mutual Information:")
    for feature in mi_sorted.head(15).index:
        mi_val = mi_sorted[feature]
        print(f"  {feature}: {mi_val:.4f}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    mi_sorted.head(15).plot(kind='bar')
    plt.title('Mutual Information with eGFR Category')
    plt.ylabel('Mutual Information Score')
    plt.xlabel('Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"MI plot saved to: {save_path}")
    plt.close()
    
    return mi_series

def feature_stability_analysis(X, y, features, n_splits=5, save_path=None):
    """Analyze feature selection stability using cross-validation."""
    print("\n" + "="*60)
    print("FEATURE STABILITY ANALYSIS")
    print("="*60)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    feature_selection_counts = {feature: 0 for feature in features}
    feature_ranks = []
    
    for fold, (train_idx, _) in enumerate(skf.split(X, y)):
        X_fold = X.iloc[train_idx]
        y_fold = y.iloc[train_idx]
        
        # Random Forest importance for this fold
        rf = RandomForestClassifier(n_estimators=50, random_state=42 + fold, n_jobs=-1)
        rf.fit(X_fold, y_fold)
        
        fold_importances = pd.Series(rf.feature_importances_, index=features)
        fold_ranks = fold_importances.rank(ascending=False)
        feature_ranks.append(fold_ranks)
        
        # Count top features
        top_features = fold_importances.nlargest(8).index
        for feature in top_features:
            feature_selection_counts[feature] += 1
    
    # Calculate selection frequency
    selection_freq = {k: v/n_splits for k, v in feature_selection_counts.items()}
    
    print("Feature selection frequency (top 8 features):")
    sorted_freq = sorted(selection_freq.items(), key=lambda x: x[1], reverse=True)
    for feature, freq in sorted_freq[:15]:
        print(f"  {feature}: {freq:.2f}")
    
    # Calculate rank stability
    rank_df = pd.DataFrame(feature_ranks)
    rank_std = rank_df.std()
    
    print("\nRank stability (lower std = more stable):")
    print(rank_std.sort_values().head(10))
    
    if save_path:
        plt.figure(figsize=(12, 6))
        freq_data = pd.DataFrame.from_dict(selection_freq, orient='index', columns=['frequency'])
        freq_data = freq_data.sort_values('frequency', ascending=False)
        freq_data.head(15).plot(kind='bar')
        plt.title('Feature Selection Frequency Across CV Folds')
        plt.ylabel('Selection Frequency')
        plt.xlabel('Features')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Stability plot saved to: {save_path}")
        plt.close()
    
    return selection_freq, rank_std

def pca_analysis(X_train, features, save_dir):
    """Analyze PCA as dimensionality reduction baseline."""
    print("\n" + "="*60)
    print("PCA ANALYSIS")
    print("="*60)
    
    # Scale data for PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Test different component counts
    component_counts = [4, 6, 8, 10]
    explained_variances = {}
    
    for n_components in component_counts:
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        total_var = pca.explained_variance_ratio_.sum()
        explained_variances[n_components] = {
            'total_variance': total_var,
            'individual_variance': pca.explained_variance_ratio_
        }
        
        print(f"{n_components} components: {total_var:.3f} total variance explained")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    component_counts = list(explained_variances.keys())
    total_vars = [explained_variances[n]['total_variance'] for n in component_counts]
    plt.bar(range(len(component_counts)), total_vars, tick_label=component_counts)
    plt.title('PCA: Total Variance Explained by Component Count')
    plt.ylabel('Total Variance Explained')
    plt.xlabel('Number of Components')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'pca_variance_explained.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"PCA plot saved to: {save_path}")
    plt.close()
    
    return explained_variances

def compare_experiments(df, experiments):
    """Compare feature selection across the three experimental configurations."""
    print("\n" + "="*60)
    print("EXPERIMENT A/B/C COMPARISON")
    print("="*60)
    
    results = {}
    
    for exp_id, exp in experiments.items():
        print(f"\n{exp['name']}")
        print(f"Features: {len(exp['features'])}")
        print(f"Excluded: {exp['excluded']}")
        
        # Prepare data
        X, y, _, _ = prepare_data_for_feature_selection(df, exp['features'])
        
        # Quick RF importance
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=exp['features'])
        
        results[exp_id] = {
            'importances': importances,
            'top_features': importances.nlargest(5).index.tolist()
        }
        
        print(f"Top 5 features: {results[exp_id]['top_features']}")
    
    return results

def select_final_features(correlations, rf_importances, mi_scores, selection_freq, rank_std, target_count=8):
    """Select final features based on multiple criteria."""
    print("\n" + "="*60)
    print("FINAL FEATURE SELECTION")
    print("="*60)
    
    # Normalize scores (0-1)
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min())
    
    corr_norm = normalize(correlations.abs())
    rf_norm = normalize(rf_importances)
    mi_norm = normalize(mi_scores)
    freq_norm = pd.Series({k: v for k, v in selection_freq.items()})
    rank_norm = normalize(1 / (rank_std + 0.1))  # Inverse: lower std = higher score
    
    # Combined score (weighted average)
    combined_score = (0.3 * rf_norm + 0.3 * mi_norm + 0.2 * corr_norm + 0.1 * freq_norm + 0.1 * rank_norm)
    
    # Select top features
    selected_features = combined_score.nlargest(target_count).index.tolist()
    
    print(f"Selected {target_count} features:")
    for i, feature in enumerate(selected_features, 1):
        print(f"  {i}. {feature}")
        print(f"     RF: {rf_importances[feature]:.4f}, MI: {mi_scores[feature]:.4f}, Corr: {correlations[feature]:.3f}")
        print(f"     Stability: {selection_freq[feature]:.2f}, Rank std: {rank_std[feature]:.3f}")
    
    return selected_features, combined_score

def save_results(selected_features, combined_score, experiments, explained_variances, save_dir):
    """Save feature selection results."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save selected features as JSON
    results = {
        'selected_features': selected_features,
        'feature_count': len(selected_features),
        'selection_method': 'Combined score (RF + MI + Correlation + Stability)',
        'target_qubit_count': len(selected_features),
        'experiments_compared': list(experiments.keys()),
        'pca_analyzed': True
    }
    
    json_path = os.path.join(save_dir, 'selected_features.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSelected features saved to: {json_path}")
    
    return json_path

def main():
    """Main execution function."""
    print("="*60)
    print("PHASE 4: FEATURE SELECTION ANALYSIS")
    print("="*60)
    
    # Create results directory
    results_dir = os.path.join(project_root, 'results/figures/feature_selection')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load corrected data
    df = load_corrected_data()
    
    # Define experiments
    experiments = define_experiments(df)
    
    # Use Experiment B (leakage-aware) as primary
    primary_exp = experiments['B_no_leakage']
    features = primary_exp['features']
    
    print(f"\nPrimary Experiment: {primary_exp['name']}")
    print(f"Features: {len(features)}")
    print(f"Excluded leakage features: {primary_exp['excluded']}")
    
    # Prepare data
    X, y, numeric_cols, categorical_cols = prepare_data_for_feature_selection(df, features)
    
    # Simple train split for feature selection (leakage control)
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\nUsing {len(X_train)} samples for feature selection (training data only)")
    
    # 1. Correlation analysis
    corr_path = os.path.join(results_dir, 'correlation_analysis.png')
    correlations = correlation_analysis(X_train, y_train, features, corr_path)
    
    # 2. Random Forest importance
    rf_path = os.path.join(results_dir, 'rf_feature_importance.png')
    rf_importances, rf_model = random_forest_importance(X_train, y_train, features, rf_path)
    
    # 3. Mutual information
    mi_path = os.path.join(results_dir, 'mutual_information.png')
    mi_scores = mutual_information_analysis(X_train, y_train, features, mi_path)
    
    # 4. Feature stability analysis
    stability_path = os.path.join(results_dir, 'feature_stability.png')
    selection_freq, rank_std = feature_stability_analysis(X, y, features, save_path=stability_path)
    
    # 5. PCA analysis
    explained_variances = pca_analysis(X_train, features, results_dir)
    
    # 6. Compare experiments
    exp_comparison = compare_experiments(df, experiments)
    
    # 7. Select final features
    selected_features, combined_score = select_final_features(
        correlations, rf_importances, mi_scores, selection_freq, rank_std, target_count=8
    )
    
    # 8. Save results
    json_path = save_results(selected_features, combined_score, experiments, explained_variances, results_dir)
    
    print("\n" + "="*60)
    print("PHASE 4 COMPLETE")
    print("="*60)
    print(f"Selected {len(selected_features)} features for quantum branch")
    print("All analyses saved to results/figures/feature_selection/")
    print("Next: Create detailed feature selection report")

if __name__ == "__main__":
    main()
