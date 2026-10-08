"""
Phase 4 Consistency Audit Script.
Identifies and resolves inconsistencies in Phase 4 implementation.
"""

import pandas as pd
import numpy as np
import os
import json

# Use current working directory as project root
project_root = os.getcwd()

def audit_training_split_consistency():
    """Audit the training split consistency issue."""
    print("="*60)
    print("AUDIT 1: TRAINING SPLIT CONSISTENCY")
    print("="*60)
    
    print("\nEstablished Split Protocol (Phase 3):")
    print("  Train: 239 samples (63.9%)")
    print("  Validation: 60 samples (16.0%)")
    print("  Test: 75 samples (20.1%)")
    print("  Total: 374 samples")
    
    print("\nPhase 4 Implementation:")
    print("  Feature selection used: 299 samples (80% of 374)")
    print("  This was from a NEW 80/20 split, not the established protocol")
    
    print("\nINCONSISTENCY IDENTIFIED:")
    print("  Phase 4 created a new split instead of using established Phase 3 split")
    print("  This is methodologically incorrect")
    
    print("\nCORRECT PROTOCOL:")
    print("  Feature selection should use ONLY the established training data (239 samples)")
    print("  Validation data (60 samples) should NOT be used for feature selection")
    print("  Test data (75 samples) must remain completely untouched")
    
    return False  # FAIL - inconsistency found

def audit_feature_selection_inconsistency():
    """Audit the feature selection vs stability inconsistency."""
    print("\n" + "="*60)
    print("AUDIT 2: FEATURE SELECTION vs STABILITY INCONSISTENCY")
    print("="*60)
    
    # Load selected features
    json_path = os.path.join(project_root, 'results/figures/feature_selection/selected_features.json')
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    final_selected = results['selected_features']
    
    print("\nFinal Selected Features (from combined score):")
    print(f"  {final_selected}")
    
    print("\nStability Analysis Results (100% selection frequency):")
    print("  bgr, bu, sod, pot, hemo, pcv, wbcc, rbcc")
    
    print("\nINCONSISTENCY IDENTIFIED:")
    print("  Final set includes: al, htn")
    print("  Stability set includes: pot, wbcc")
    print("  The combined score method selected different features than stability frequency")
    
    print("\nROOT CAUSE:")
    print("  Combined score: 0.3*RF + 0.3*MI + 0.2*Corr + 0.1*Stability + 0.1*Rank")
    print("  This weighted combination overrode pure stability frequency")
    
    return False  # FAIL - inconsistency found

def audit_leakage_terminology():
    """Audit the terminology issues."""
    print("\n" + "="*60)
    print("AUDIT 3: TERMINOLOGY ISSUES")
    print("="*60)
    
    print("\nISSUE: 'Clinical Validity: HIGH'")
    print("CURRENT: Experiment B uses this terminology")
    print("PROBLEM: Inappropriate for exploratory analysis")
    
    print("\nCORRECT TERMINOLOGY:")
    print("  'Leakage-aware exploratory configuration'")
    print("  Do not claim clinical validity based only on excluding age/sc")
    
    print("\nISSUE: 'Balanced classes'")
    print("CURRENT: Report may describe categories as balanced")
    print("PROBLEM: Categories are moderately imbalanced")
    
    print("\nCORRECT TERMINOLOGY:")
    print("  'Moderately imbalanced'")
    print("  Report exact distribution: Cat1=27%, Cat2=22%, Cat3=20%, Cat4=14%, Cat5=17%")
    
    return False  # FAIL - terminology issues found

def audit_pca_implementation():
    """Audit the PCA implementation for leakage."""
    print("\n" + "="*60)
    print("AUDIT 4: PCA IMPLEMENTATION")
    print("="*60)
    
    print("\nPCA Implementation Review:")
    print("  PCA was fitted on the same 299 samples used for feature selection")
    print("  This is the NEW split, not the established Phase 3 split")
    
    print("\nLEAKAGE CONCERN:")
    print("  PCA used training data only (299 samples)")
    print("  However, this was from the NEW split, not established protocol")
    print("  Must use established training data (239 samples)")
    
    print("\nPCA STATUS:")
    print("  Baseline comparison only (not for final pipeline)")
    print("  Direct feature selection preferred for interpretability")
    
    return False  # FAIL - used wrong split

def main():
    """Main audit function."""
    print("="*60)
    print("PHASE 4 CONSISTENCY AUDIT")
    print("="*60)
    
    audits = [
        ("Training Split Consistency", audit_training_split_consistency),
        ("Feature Selection Inconsistency", audit_feature_selection_inconsistency),
        ("Terminology Issues", audit_leakage_terminology),
        ("PCA Implementation", audit_pca_implementation)
    ]
    
    results = {}
    for name, audit_func in audits:
        try:
            audit_func()
            results[name] = "FAIL"  # All audits should find issues
        except Exception as e:
            print(f"✗ {name} ERROR: {e}")
            results[name] = "ERROR"
    
    print("\n" + "="*60)
    print("AUDIT SUMMARY")
    print("="*60)
    
    for name, result in results.items():
        status = "✗" if result == "FAIL" else "✓"
        print(f"{status} {name}: {result}")
    
    print("\n" + "="*60)
    print("PHASE 4 CONSISTENCY STATUS: FAIL")
    print("="*60)
    print("Multiple inconsistencies identified that must be resolved.")
    print("DO NOT proceed to Phase 5 until all issues are fixed.")

if __name__ == "__main__":
    main()
