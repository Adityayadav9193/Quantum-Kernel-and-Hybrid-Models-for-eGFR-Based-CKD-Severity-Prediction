"""
CORRECTED Phase 4 Feature Selection Analysis.
Uses established Phase 3 training data (239 samples) and consistent methodology.
"""

import pandas as pd
import numpy as np
import os
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

def use_established_training_split(df, features, target='egfr_category'):
    """
    Use the ESTABLISHED Phase 3 training data (239 samples).
    This is the CORRECT methodology.
    """
    print("\n" + "="*60)
    print("USING ESTABLISHED PHASE 3 TRAINING DATA")
    print("="*60)
    
    # Recreate the established split using the same parameters as Phase 3
    from sklearn.model_selection import train_test_split
    X = df[features].copy()
    y = df[target].copy()
    
    # First split: train+val vs test (20% test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: train vs val (20% of remaining for validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )
    
    print(f"Established training data: {len(X_train)} samples (from Phase 3)")
    print(f"Validation data: {len(X_val)} samples (from Phase 3)")
    print(f"Test data: {len(X_test)} samples (from Phase 3)")
    print(f"Feature selection will use ONLY training data: {len(X_train)} samples")
    
    return X_train, y_train

def prepare_data_for_feature_selection(X_train, features):
    """Prepare training data for feature selection."""
    # Identify numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()
    
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    
    # Simple encoding for categorical columns
    X_encoded = X_train.copy()
    for col in categorical_cols:
        X_encoded[col] = pd.factorize(X_encoded[col])[0]
    
    # Handle missing values
    if X_encoded.isnull().any().any():
        print("Imputing missing values for feature selection")
        for col in numeric_cols:
            median_val = X_encoded[col].median()
            if pd.isna(median_val):
                median_val = 0
            X_encoded[col].fillna(median_val, inplace=True)
        for col in categorical_cols:
            mode_val = X_encoded[col].mode()[0]
            if pd.isna(mode_val):
                mode_val = 0
            X_encoded[col].fillna(mode_val, inplace=True)
    
    # Ensure no NaN values remain
    X_encoded = X_encoded.fillna(0)
    
    return X_encoded, numeric_cols, categorical_cols

def correlation_analysis(X_train, y_train, features, save_path):
    """Perform correlation analysis."""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    correlations = X_train.corrwith(pd.Series(y_train))
    correlations_abs = correlations.abs().sort_values(ascending=False)
    
    print("Top features by correlation:")
    for feature in correlations_abs.head(15).index:
        corr_val = correlations[feature]
        print(f"  {feature}: {corr_val:.3f}")
    
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
    """Perform Random Forest feature importance."""
    print("\n" + "="*60)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("="*60)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    importances = pd.Series(rf.feature_importances_, index=features)
    importances_sorted = importances.sort_values(ascending=False)
    
    print("Top features by Random Forest importance:")
    for feature in importances_sorted.head(15).index:
        imp_val = importances_sorted[feature]
        print(f"  {feature}: {imp_val:.4f}")
    
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
    mi_sorted = mi_series.sort_values(ascending=False)
    
    print("Top features by Mutual Information:")
    for feature in mi_sorted.head(15).index:
        mi_val = mi_sorted[feature]
        print(f"  {feature}: {mi_val:.4f}")
    
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

def feature_stability_analysis(X_train, y_train, features, n_splits=5, save_path=None):
    """Analyze feature selection stability using cross-validation."""
    print("\n" + "="*60)
    print("FEATURE STABILITY ANALYSIS")
    print("="*60)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    feature_selection_counts = {feature: 0 for feature in features}
    feature_ranks = []
    
    for fold, (train_idx, _) in enumerate(skf.split(X_train, y_train)):
        X_fold = X_train.iloc[train_idx]
        y_fold = y_train.iloc[train_idx]
        
        rf = RandomForestClassifier(n_estimators=50, random_state=42 + fold, n_jobs=-1)
        rf.fit(X_fold, y_fold)
        
        fold_importances = pd.Series(rf.feature_importances_, index=features)
        fold_ranks = fold_importances.rank(ascending=False)
        feature_ranks.append(fold_ranks)
        
        top_features = fold_importances.nlargest(8).index
        for feature in top_features:
            feature_selection_counts[feature] += 1
    
    selection_freq = {k: v/n_splits for k, v in feature_selection_counts.items()}
    
    print("Feature selection frequency (top 8 features):")
    sorted_freq = sorted(selection_freq.items(), key=lambda x: x[1], reverse=True)
    for feature, freq in sorted_freq[:15]:
        print(f"  {feature}: {freq:.2f}")
    
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
    """Analyze PCA using CORRECT training data."""
    print("\n" + "="*60)
    print("PCA ANALYSIS (CORRECTED METHODOLOGY)")
    print("="*60)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
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
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(component_counts)), [explained_variances[n]['total_variance'] for n in component_counts], tick_label=component_counts)
    plt.title('PCA: Total Variance Explained by Component Count')
    plt.ylabel('Total Variance Explained')
    plt.xlabel('Number of Components')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'pca_variance_explained.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"PCA plot saved to: {save_path}")
    plt.close()
    
    return explained_variances

def apply_consistent_selection_rule(correlations, rf_importances, mi_scores, selection_freq, rank_std, target_count=8):
    """
    Apply CONSISTENT selection rule: stability frequency as primary criterion.
    
    Selection Rule:
    1. Primary: Stability frequency (features selected in 100% of CV folds)
    2. Secondary: If insufficient stable features, use RF + MI agreement
    3. Tertiary: Use correlation for breaking ties
    """
    print("\n" + "="*60)
    print("APPLYING CONSISTENT SELECTION RULE")
    print("="*60)
    
    print("Selection Rule:")
    print("  1. Primary: Stability frequency (100% selection)")
    print("  2. Secondary: RF + MI agreement (top features in both)")
    print("  3. Tertiary: Correlation for tie-breaking")
    
    # Step 1: Get features with 100% stability
    stable_features = [feat for feat, freq in selection_freq.items() if freq == 1.0]
    print(f"\nFeatures with 100% stability: {len(stable_features)}")
    print(f"  {stable_features}")
    
    # Step 2: If we have enough stable features, use them
    if len(stable_features) >= target_count:
        selected_features = stable_features[:target_count]
        print(f"\nSufficient stable features available. Selecting top {target_count} by RF importance among stable:")
        
        # Rank stable features by RF importance
        stable_importance = {feat: rf_importances[feat] for feat in stable_features}
        sorted_stable = sorted(stable_importance.items(), key=lambda x: x[1], reverse=True)
        selected_features = [x[0] for x in sorted_stable[:target_count]]
    else:
        # Need to include additional features based on RF + MI agreement
        print(f"\nInsufficient stable features ({len(stable_features)}). Adding RF + MI agreement.")
        
        # Get top features by RF
        rf_top = set(rf_importances.nlargest(target_count).index)
        # Get top features by MI
        mi_top = set(mi_scores.nlargest(target_count).index)
        
        # Intersection
        agreement_features = list(rf_top & mi_top)
        print(f"  RF + MI agreement: {agreement_features}")
        
        # Add stable features first, then agreement features (without duplicates)
        selected_features = stable_features + [f for f in agreement_features if f not in stable_features]
        
        # If still not enough, add top by RF importance
        if len(selected_features) < target_count:
            remaining_needed = target_count - len(selected_features)
            rf_top_all = rf_importances.nlargest(target_count + remaining_needed).index.tolist()
            additional = [f for f in rf_top_all if f not in selected_features]
            selected_features = selected_features + additional[:remaining_needed]
        
        selected_features = selected_features[:target_count]
    
    print(f"\nFinal {target_count} features (consistent rule):")
    for i, feature in enumerate(selected_features, 1):
        rf_val = rf_importances[feature]
        mi_val = mi_scores[feature]
        corr_val = correlations[feature]
        freq_val = selection_freq.get(feature, 0)
        rank_val = rank_std[feature]
        print(f"  {i}. {feature}")
        print(f"     RF: {rf_val:.4f}, MI: {mi_val:.4f}, Corr: {corr_val:.3f}")
        print(f"     Stability: {freq_val:.2f}, Rank std: {rank_val:.3f}")
    
    return selected_features

def compare_feature_counts(selected_features, correlations, rf_importances, mi_scores, save_dir):
    """Compare different feature counts."""
    print("\n" + "="*60)
    print("FEATURE COUNT COMPARISON")
    print("="*60)
    
    feature_counts = [4, 6, 8, 10]
    comparison_results = {}
    
    for count in feature_counts:
        if count <= len(selected_features):
            features = selected_features[:count]
        else:
            # Use top features by stability frequency
            print(f"Need {count} features but only have {len(selected_features)} stable features")
            print(f"Using RF importance to add features")
            all_importances = rf_importances
            additional = all_importances.nlargest(count).index.tolist()
            features = selected_features + [f for f in additional if f not in selected_features]
            features = features[:count]
        
        # Calculate combined score for these features
        combined_score = (rf_importances[features].sum() + mi_scores[features].sum()) / len(features)
        comparison_results[count] = {
            'features': features,
            'combined_score': combined_score
        }
        
        print(f"{count} features: {features}")
        print(f"  Combined score: {combined_score:.4f}")
    
    return comparison_results

def main():
    """Main execution function."""
    print("="*60)
    print("CORRECTED PHASE 4: FEATURE SELECTION ANALYSIS")
    print("="*60)
    
    # Create results directory
    results_dir = os.path.join(project_root, 'results/figures/feature_selection')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load corrected data
    df = load_corrected_data()
    
    # Define Experiment B (leakage-aware)
    all_features = [col for col in df.columns if col not in ['class', 'egfr', 'egfr_category', 'category_description']]
    leakage_features = ['sc', 'age']
    features = [f for f in all_features if f not in leakage_features]
    
    print(f"\nExperiment B: Leakage-aware exploratory configuration")
    print(f"Features: {len(features)}")
    print(f"Excluded leakage features: {leakage_features}")
    
    # Use ESTABLISHED training data (239 samples from Phase 3)
    X_train, y_train = use_established_training_split(df, features)
    
    # Prepare data
    X_train_processed, numeric_cols, categorical_cols = prepare_data_for_feature_selection(X_train, features)
    
    print(f"\nUsing ESTABLISHED training data: {len(X_train_processed)} samples")
    
    # 1. Correlation analysis
    corr_path = os.path.join(results_dir, 'correlation_analysis_corrected.png')
    correlations = correlation_analysis(X_train_processed, y_train, features, corr_path)
    
    # 2. Random Forest importance
    rf_path = os.path.join(results_dir, 'rf_feature_importance_corrected.png')
    rf_importances, rf_model = random_forest_importance(X_train_processed, y_train, features, rf_path)
    
    # 3. Mutual information
    mi_path = os.path.join(results_dir, 'mutual_information_corrected.png')
    mi_scores = mutual_information_analysis(X_train_processed, y_train, features, mi_path)
    
    # 4. Feature stability analysis
    stability_path = os.path.join(results_dir, 'feature_stability_corrected.png')
    selection_freq, rank_std = feature_stability_analysis(X_train_processed, y_train, features, save_path=stability_path)
    
    # 5. PCA analysis (CORRECTED - uses established training data)
    explained_variances = pca_analysis(X_train_processed, features, results_dir)
    
    # 6. Apply CONSISTENT selection rule
    selected_features = apply_consistent_selection_rule(
        correlations, rf_importances, mi_scores, selection_freq, rank_std, target_count=8
    )
    
    # 7. Compare feature counts
    comparison_results = compare_feature_counts(selected_features, correlations, rf_importances, mi_scores, results_dir)
    
    # 8. Save results
    results = {
        'selected_features': selected_features,
        'feature_count': len(selected_features),
        'selection_method': 'Consistent rule: Stability frequency primary, RF+MI secondary',
        'training_samples_used': len(X_train_processed),
        'split_protocol': 'Established Phase 3 training data (239 samples)',
        'excluded_leakage_features': leakage_features,
        'experiment': 'B_leakage_aware_exploratory'
    }
    
    json_path = os.path.join(results_dir, 'selected_features_corrected.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nCorrected selected features saved to: {json_path}")
    
    print("\n" + "="*60)
    print("CORRECTED PHASE 4 COMPLETE")
    print("="*60)
    print(f"Selected {len(selected_features)} features using consistent methodology")
    print("All analyses saved to results/figures/feature_selection/")
    print("Next: Update reports with corrected results")

if __name__ == "__main__":
    main()
