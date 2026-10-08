"""
Corrected preprocessing pipeline for eGFR-based category classification.
This module implements robust preprocessing with train/test split and feature set management.
All preprocessing transformations are fit ONLY on training data to prevent leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class CKDPreprocessor:
    """
    Preprocessor for eGFR-based category classification.
    
    CRITICAL: All preprocessing transformations (including imputation) are fit ONLY on training data
    to prevent test-set information leakage.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.feature_columns = None
        self.categorical_columns = None
        self.numerical_columns = None
        self.label_mapping = None
        
    def identify_column_types(self, df, feature_columns):
        """
        Identify numerical and categorical columns.
        
        Args:
            df: DataFrame
            feature_columns: List of feature column names
            
        Returns:
            tuple: (numerical_columns, categorical_columns)
        """
        numerical_columns = []
        categorical_columns = []
        
        for col in feature_columns:
            if df[col].dtype in ['int64', 'float64']:
                numerical_columns.append(col)
            else:
                categorical_columns.append(col)
        
        return numerical_columns, categorical_columns
    
    def impute_missing_values(self, df, numerical_columns, categorical_columns, fit=True):
        """
        Impute missing values using appropriate strategies.
        
        CRITICAL: Imputation parameters are fit ONLY on training data.
        
        Args:
            df: DataFrame
            numerical_columns: List of numerical column names
            categorical_columns: List of categorical column names
            fit: Whether to fit the imputers (True for train, False for test)
            
        Returns:
            DataFrame with imputed values
        """
        df_imputed = df.copy()
        
        # Impute numerical columns with median
        if numerical_columns:
            if fit:
                self.numeric_imputer.fit(df_imputed[numerical_columns])
            df_imputed[numerical_columns] = self.numeric_imputer.transform(df_imputed[numerical_columns])
        
        # Impute categorical columns with mode
        if categorical_columns:
            if fit:
                self.categorical_imputer.fit(df_imputed[categorical_columns])
            df_imputed[categorical_columns] = self.categorical_imputer.transform(df_imputed[categorical_columns])
        
        return df_imputed
    
    def encode_categorical_features(self, df, categorical_columns, fit=True):
        """
        Encode categorical features using label encoding.
        
        Args:
            df: DataFrame
            categorical_columns: List of categorical column names
            fit: Whether to fit the encoders (True for train, False for test)
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        
        if not hasattr(self, 'categorical_encoders'):
            self.categorical_encoders = {}
        
        for col in categorical_columns:
            if fit:
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
                self.categorical_encoders[col] = encoder
            else:
                if col in self.categorical_encoders:
                    # Handle unseen categories
                    encoder = self.categorical_encoders[col]
                    unique_values = set(df_encoded[col].astype(str).unique())
                    encoder_classes = set(encoder.classes_)
                    
                    # Add unseen categories to encoder
                    unseen_categories = unique_values - encoder_classes
                    if unseen_categories:
                        # Map unseen categories to a special value
                        df_encoded[col] = df_encoded[col].astype(str).apply(
                            lambda x: x if x in encoder_classes else 'unseen'
                        )
                        # Fit encoder on extended classes
                        extended_classes = list(encoder.classes_) + list(unseen_categories)
                        encoder.fit(extended_classes)
                        self.categorical_encoders[col] = encoder
                    
                    df_encoded[col] = encoder.transform(df_encoded[col].astype(str))
                else:
                    # Fallback: simple label encoding
                    encoder = LabelEncoder()
                    df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
                    self.categorical_encoders[col] = encoder
        
        return df_encoded
    
    def scale_numerical_features(self, df, numerical_columns, fit=True):
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df: DataFrame
            numerical_columns: List of numerical column names
            fit: Whether to fit the scaler (True for train, False for test)
            
        Returns:
            DataFrame with scaled numerical features
        """
        df_scaled = df.copy()
        
        if fit:
            self.scaler.fit(df_scaled[numerical_columns])
        
        df_scaled[numerical_columns] = self.scaler.transform(df_scaled[numerical_columns])
        
        return df_scaled
    
    def encode_labels(self, labels, fit=True):
        """
        Encode target labels (CKD stages).
        
        Args:
            labels: Target labels
            fit: Whether to fit the label encoder
            
        Returns:
            Encoded labels
        """
        if fit:
            encoded_labels = self.label_encoder.fit_transform(labels)
            self.label_mapping = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        else:
            encoded_labels = self.label_encoder.transform(labels)
        
        return encoded_labels
    
    def split_data(self, df, feature_columns, target_column='egfr_category', 
                   test_size=0.2, val_size=0.2, stratify=True):
        """
        Split data into train/validation/test sets with stratification.
        
        Args:
            df: DataFrame
            feature_columns: List of feature column names
            target_column: Name of target column (egfr_category)
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation set
            stratify: Whether to use stratified sampling
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        self.feature_columns = feature_columns
        
        # Extract features and target
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # First split: train+val vs test
        if stratify:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
        else:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
        
        # Second split: train vs val
        if stratify:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_size, 
                random_state=self.random_state, stratify=y_train_val
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_size, 
                random_state=self.random_state
            )
        
        print(f"Data split complete:")
        print(f"  Train: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
        print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
        print(f"  Test: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess_train(self, X_train, y_train):
        """
        Preprocess training data (fit and transform).
        
        CRITICAL: All transformations (imputation, encoding, scaling) are fit on training data only.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            tuple: (X_train_processed, y_train_encoded)
        """
        # Identify column types
        self.numerical_columns, self.categorical_columns = self.identify_column_types(
            X_train, X_train.columns
        )
        
        print(f"Column types identified:")
        print(f"  Numerical: {len(self.numerical_columns)} columns")
        print(f"  Categorical: {len(self.categorical_columns)} columns")
        
        # Impute missing values (fit on training data)
        print("Imputing missing values (fit on training data)...")
        X_train_processed = self.impute_missing_values(
            X_train, self.numerical_columns, self.categorical_columns, fit=True
        )
        
        # Encode categorical features
        X_train_processed = self.encode_categorical_features(
            X_train_processed, self.categorical_columns, fit=True
        )
        
        # Scale numerical features
        X_train_processed = self.scale_numerical_features(
            X_train_processed, self.numerical_columns, fit=True
        )
        
        # Encode labels
        y_train_encoded = self.encode_labels(y_train, fit=True)
        
        print(f"Label mapping: {self.label_mapping}")
        
        return X_train_processed, y_train_encoded
    
    def preprocess_test(self, X_test, y_test=None):
        """
        Preprocess test data (transform only, no fitting).
        
        CRITICAL: Uses parameters fitted on training data only.
        
        Args:
            X_test: Test features
            y_test: Test labels (optional)
            
        Returns:
            tuple: (X_test_processed, y_test_encoded)
        """
        # Impute missing values (using training parameters)
        X_test_processed = self.impute_missing_values(
            X_test, self.numerical_columns, self.categorical_columns, fit=False
        )
        
        # Encode categorical features (no fitting)
        X_test_processed = self.encode_categorical_features(
            X_test_processed, self.categorical_columns, fit=False
        )
        
        # Scale numerical features (no fitting)
        X_test_processed = self.scale_numerical_features(
            X_test_processed, self.numerical_columns, fit=False
        )
        
        # Encode labels if provided
        if y_test is not None:
            y_test_encoded = self.encode_labels(y_test, fit=False)
        else:
            y_test_encoded = None
        
        return X_test_processed, y_test_encoded
    
    def save_preprocessor(self, save_path):
        """
        Save preprocessor to disk.
        
        Args:
            save_path: Path to save preprocessor
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Preprocessor saved to: {save_path}")
    
    def load_preprocessor(self, load_path):
        """
        Load preprocessor from disk.
        
        Args:
            load_path: Path to load preprocessor from
        """
        with open(load_path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        # Copy attributes
        self.random_state = preprocessor.random_state
        self.label_encoder = preprocessor.label_encoder
        self.scaler = preprocessor.scaler
        self.numeric_imputer = preprocessor.numeric_imputer
        self.categorical_imputer = preprocessor.categorical_imputer
        self.feature_columns = preprocessor.feature_columns
        self.categorical_columns = preprocessor.categorical_columns
        self.numerical_columns = preprocessor.numerical_columns
        self.label_mapping = preprocessor.label_mapping
        self.categorical_encoders = preprocessor.categorical_encoders
        
        print(f"Preprocessor loaded from: {load_path}")
        
        return self


def create_experiment_feature_sets():
    """
    Create feature sets for different leakage experiments.
    
    Returns:
        dict: Dictionary of experiment configurations
    """
    all_features = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 
                   'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 
                   'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    
    # Direct eGFR generation variables (target leakage)
    leakage_features = ['sc', 'age']
    
    experiments = {
        'A_full_features': {
            'name': 'Experiment A: Full Features (with leakage)',
            'features': all_features,
            'excluded_features': [],
            'included_leakage_features': leakage_features,
            'description': 'All features including direct eGFR-generation variables (sc, age). Upper-bound/reference experiment only.',
            'is_leakage_aware': False,
            'clinical_validity': 'LOW - not clinically realistic'
        },
        'B_no_leakage': {
            'name': 'Experiment B: No Leakage (leakage-aware)',
            'features': [f for f in all_features if f not in leakage_features],
            'excluded_features': leakage_features,
            'included_leakage_features': [],
            'description': 'Excludes direct eGFR-generation variables (sc, age). Primary scientifically conservative experiment.',
            'is_leakage_aware': True,
            'clinical_validity': 'HIGH - clinically realistic scenario'
        },
        'C_partial_exclusion': {
            'name': 'Experiment C: Partial Exclusion (exclude sc only)',
            'features': [f for f in all_features if f != 'sc'],
            'excluded_features': ['sc'],
            'included_leakage_features': ['age'],
            'description': 'Excludes sc (primary leakage variable) but includes age. Intermediate experiment.',
            'is_leakage_aware': True,
            'clinical_validity': 'MEDIUM - partially realistic'
        }
    }
    
    return experiments


if __name__ == "__main__":
    # Test the preprocessor
    print("Testing CKD Preprocessor")
    print("="*60)
    
    # Load data
    data_path = os.path.join(project_root, 'data/processed', 'ckd_multistage.csv')
    df = pd.read_csv(data_path)
    print(f"Loaded data: {df.shape}")
    
    # Create preprocessor
    preprocessor = CKDPreprocessor(random_state=42)
    
    # Define feature set (leakage-aware)
    experiments = create_experiment_feature_sets()
    feature_columns = experiments['no_leakage']['features']
    
    print(f"Using feature set: {experiments['no_leakage']['name']}")
    print(f"Features: {len(feature_columns)}")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        df, feature_columns, target_column='ckd_stage'
    )
    
    # Preprocess training data
    X_train_processed, y_train_encoded = preprocessor.preprocess_train(X_train, y_train)
    
    # Preprocess validation data
    X_val_processed, y_val_encoded = preprocessor.preprocess_test(X_val, y_val)
    
    # Preprocess test data
    X_test_processed, y_test_encoded = preprocessor.preprocess_test(X_test, y_test)
    
    print("\nPreprocessing complete!")
    print(f"Train shape: {X_train_processed.shape}")
    print(f"Validation shape: {X_val_processed.shape}")
    print(f"Test shape: {X_test_processed.shape}")
