"""
Phase 5: Classical Baselines Visualizations
Creates confusion matrices, model comparisons, and per-class results.
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle

# Use current working directory as project root
project_root = os.getcwd()

def load_results():
    """Load classical baseline results."""
    results_path = os.path.join(project_root, 'results/classical_results.json')
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results

def create_confusion_matrices(results):
    """Create confusion matrices for each model."""
    print("="*60)
    print("CREATING CONFUSION MATRICES")
    print("="*60)
    
    figures_dir = os.path.join(project_root, 'results/figures/classical')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Validation confusion matrices
    for model_name, val_results in results['val_results'].items():
        cm = np.array(val_results['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[1, 2, 3, 4, 5],
                   yticklabels=[1, 2, 3, 4, 5])
        plt.title(f'{model_name} - Validation Confusion Matrix')
        plt.ylabel('True eGFR Category')
        plt.xlabel('Predicted eGFR Category')
        plt.tight_layout()
        
        save_path = os.path.join(figures_dir, f'{model_name.replace(" ", "_")}_val_cm.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    # Test confusion matrix for best model
    best_model = results['best_model']
    test_cm = np.array(results['test_results']['confusion_matrix'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=[1, 2, 3, 4, 5],
               yticklabels=[1, 2, 3, 4, 5])
    plt.title(f'{best_model} - Test Confusion Matrix')
    plt.ylabel('True eGFR Category')
    plt.xlabel('Predicted eGFR Category')
    plt.tight_layout()
    
    save_path = os.path.join(figures_dir, f'{best_model.replace(" ", "_")}_test_cm.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def create_model_comparison(results):
    """Create model comparison plots."""
    print("\n" + "="*60)
    print("CREATING MODEL COMPARISON PLOTS")
    print("="*60)
    
    figures_dir = os.path.join(project_root, 'results/figures/classical')
    
    # Extract validation metrics
    models = list(results['val_results'].keys())
    accuracies = [results['val_results'][m]['accuracy'] for m in models]
    macro_f1s = [results['val_results'][m]['macro_f1'] for m in models]
    weighted_f1s = [results['val_results'][m]['weighted_f1'] for m in models]
    
    # Model comparison by Macro-F1
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, macro_f1s, color='steelblue')
    plt.title('Model Comparison - Validation Macro-F1')
    plt.ylabel('Macro-F1 Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    save_path = os.path.join(figures_dir, 'model_comparison_macro_f1.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    # Model comparison by Accuracy
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color='coral')
    plt.title('Model Comparison - Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    save_path = os.path.join(figures_dir, 'model_comparison_accuracy.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def create_per_class_comparison(results):
    """Create per-class F1 comparison."""
    print("\n" + "="*60)
    print("CREATING PER-CLASS F1 COMPARISON")
    print("="*60)
    
    figures_dir = os.path.join(project_root, 'results/figures/classical')
    
    models = list(results['val_results'].keys())
    categories = [1, 2, 3, 4, 5]
    
    # Create per-class F1 matrix
    per_class_f1 = {}
    for model_name in models:
        per_class_f1[model_name] = results['val_results'][model_name]['per_class_f1']
    
    # Create DataFrame for easier plotting
    df_f1 = pd.DataFrame(per_class_f1, index=categories)
    
    plt.figure(figsize=(10, 6))
    df_f1.plot(kind='bar', figsize=(10, 6))
    plt.title('Per-Class F1 Score Comparison (Validation)')
    plt.ylabel('F1 Score')
    plt.xlabel('eGFR Category')
    plt.xticks(rotation=0)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1)
    plt.tight_layout()
    
    save_path = os.path.join(figures_dir, 'per_class_f1_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def create_cv_comparison(results):
    """Create cross-validation comparison."""
    print("\n" + "="*60)
    print("CREATING CROSS-VALIDATION COMPARISON")
    print("="*60)
    
    figures_dir = os.path.join(project_root, 'results/figures/classical')
    
    models = list(results['cv_results'].keys())
    cv_acc_means = [results['cv_results'][m]['mean_accuracy'] for m in models]
    cv_acc_stds = [results['cv_results'][m]['std_accuracy'] for m in models]
    cv_f1_means = [results['cv_results'][m]['mean_macro_f1'] for m in models]
    cv_f1_stds = [results['cv_results'][m]['std_macro_f1'] for m in models]
    
    # CV Macro-F1 comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, cv_f1_means, yerr=cv_f1_stds, capsize=5, color='mediumseagreen')
    plt.title('Cross-Validation Macro-F1 Comparison')
    plt.ylabel('Mean Macro-F1 ± Std')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    
    for bar, mean, std in zip(bars, cv_f1_means, cv_f1_stds):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    save_path = os.path.join(figures_dir, 'cv_macro_f1_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def create_summary_table(results):
    """Create summary table of results."""
    print("\n" + "="*60)
    print("CREATING SUMMARY TABLE")
    print("="*60)
    
    figures_dir = os.path.join(project_root, 'results/figures/classical')
    
    # Create summary DataFrame
    summary_data = []
    for model_name in results['val_results'].keys():
        val_data = results['val_results'][model_name]
        cv_data = results['cv_results'][model_name]
        
        summary_data.append({
            'Model': model_name,
            'CV Macro-F1': f"{cv_data['mean_macro_f1']:.4f} ± {cv_data['std_macro_f1']:.4f}",
            'CV Accuracy': f"{cv_data['mean_accuracy']:.4f} ± {cv_data['std_accuracy']:.4f}",
            'Val Macro-F1': f"{val_data['macro_f1']:.4f}",
            'Val Accuracy': f"{val_data['accuracy']:.4f}",
            'Val ROC-AUC': f"{val_data['roc_auc']:.4f}" if val_data['roc_auc'] else "N/A"
        })
    
    # Add test results for best model
    best_model = results['best_model']
    test_data = results['test_results']
    summary_data.append({
        'Model': f'{best_model} (Test)',
        'CV Macro-F1': '-',
        'CV Accuracy': '-',
        'Val Macro-F1': '-',
        'Val Accuracy': '-',
        'Val ROC-AUC': '-'
    })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Create table visualization
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df_summary.values, colLabels=df_summary.columns,
                   cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title('Classical Baselines Summary', pad=20)
    plt.tight_layout()
    
    save_path = os.path.join(figures_dir, 'summary_table.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    # Save as CSV
    csv_path = os.path.join(project_root, 'results/classical_results.csv')
    df_summary.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

def main():
    """Main execution function."""
    print("="*60)
    print("PHASE 5: CLASSICAL BASELINES VISUALIZATIONS")
    print("="*60)
    
    # Load results
    results = load_results()
    
    # Create visualizations
    create_confusion_matrices(results)
    create_model_comparison(results)
    create_per_class_comparison(results)
    create_cv_comparison(results)
    create_summary_table(results)
    
    print("\n" + "="*60)
    print("VISUALIZATIONS COMPLETE")
    print("="*60)
    print("All visualizations saved to results/figures/classical/")

if __name__ == "__main__":
    main()
