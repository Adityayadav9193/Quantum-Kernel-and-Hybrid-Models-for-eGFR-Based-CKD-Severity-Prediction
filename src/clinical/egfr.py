"""
eGFR calculation module for kidney function categorization.
This module implements various eGFR equations and handles the limitation of missing sex information.

IMPORTANT: The target variable derived from eGFR represents "eGFR-based severity categories",
NOT clinically confirmed CKD stages. Clinical CKD diagnosis requires evidence of kidney damage
(proteinuria, structural abnormalities, etc.) in addition to eGFR values.
"""

import numpy as np
import pandas as pd

class EGFRCalculator:
    """
    eGFR Calculator with multiple equation implementations.
    
    CRITICAL LIMITATION: The dataset does not contain sex information.
    Standard CKD-EPI equation requires sex as a parameter.
    This module implements sex-independent alternatives with clear documentation.
    
    TARGET DEFINITION: eGFR-based severity categories (1-5), NOT clinical CKD stages.
    """
    
    def __init__(self):
        self.available_equations = [
            'mdrd_no_sex',
            'ckdepi_no_sex_approx',
            'pediatric_schwartz',
            'combined_approach'
        ]
    
    def mdrd_no_sex(self, serum_creatinine, age):
        """
        MDRD (Modification of Diet in Renal Disease) equation without sex coefficient.
        
        NOTE: This is NOT the standard MDRD equation. The standard MDRD equation
        includes a sex coefficient (0.742 for females, 1.0 for males).
        This implementation uses the male coefficient as a default, which will
        overestimate eGFR for females by approximately 25%.
        
        Equation: eGFR = 175 * (Serum Creatinine)^-1.154 * (Age)^-0.203
        
        Args:
            serum_creatinine: Serum creatinine in mg/dL
            age: Age in years
            
        Returns:
            eGFR in mL/min/1.73m²
        """
        # Handle zero or negative creatinine
        serum_creatinine = np.maximum(serum_creatinine, 0.1)
        
        egfr = 175 * (serum_creatinine ** -1.154) * (age ** -0.203)
        
        return egfr
    
    def ckdepi_no_sex_approx(self, serum_creatinine, age):
        """
        CKD-EPI equation approximation without sex information.
        
        NOTE: This is NOT the standard CKD-EPI equation. The standard equation
        requires sex for the race/sex coefficients. This uses the male coefficients
        as a default, which will overestimate eGFR for females.
        
        Simplified male approximation:
        eGFR = 141 * (min(SCr/κ,1)^α) * (max(SCr/κ,1)^-1.209) * (0.993^Age)
        where κ = 0.9, α = -0.411 for males
        
        Args:
            serum_creatinine: Serum creatinine in mg/dL
            age: Age in years
            
        Returns:
            eGFR in mL/min/1.73m²
        """
        # Handle zero or negative creatinine
        serum_creatinine = np.maximum(serum_creatinine, 0.1)
        
        kappa = 0.9  # for males
        alpha = -0.411  # for males
        
        if serum_creatinine <= kappa:
            egfr = 141 * (serum_creatinine/kappa)**alpha * (0.993**age)
        else:
            egfr = 141 * (serum_creatinine/kappa)**-1.209 * (0.993**age)
        
        return egfr
    
    def pediatric_schwartz(self, serum_creatinine, age, height=None):
        """
        Schwartz equation for pediatric patients (age < 18 years).
        
        This equation does not require sex and is appropriate for children.
        k = 0.413 (if using IDMS-traceable creatinine)
        
        Equation: eGFR = (k * Height) / Serum Creatinine
        
        NOTE: Height is required but not available in the dataset.
        This implementation returns None for pediatric cases.
        
        Args:
            serum_creatinine: Serum creatinine in mg/dL
            age: Age in years
            height: Height in cm (optional, defaults to None)
            
        Returns:
            eGFR in mL/min/1.73m² or None if height not available
        """
        if height is None:
            return None
        
        k = 0.413  # IDMS-traceable creatinine coefficient
        egfr = (k * height) / serum_creatinine
        
        return egfr
    
    def combined_approach(self, serum_creatinine, age):
        """
        Combined approach using MDRD without sex coefficient.
        
        This is the most defensible approach given the dataset limitations:
        1. Uses MDRD equation (widely validated)
        2. Uses male coefficient as default (most conservative assumption)
        3. Clearly documents the limitation
        
        Args:
            serum_creatinine: Serum creatinine in mg/dL
            age: Age in years
            
        Returns:
            eGFR in mL/min/1.73m²
        """
        return self.mdrd_no_sex(serum_creatinine, age)
    
    def calculate_egfr(self, serum_creatinine, age, method='combined_approach'):
        """
        Calculate eGFR using specified method.
        
        Args:
            serum_creatinine: Serum creatinine in mg/dL
            age: Age in years
            method: Calculation method ('mdrd_no_sex', 'ckdepi_no_sex_approx', 'combined_approach')
            
        Returns:
            eGFR in mL/min/1.73m²
        """
        if method == 'mdrd_no_sex':
            return self.mdrd_no_sex(serum_creatinine, age)
        elif method == 'ckdepi_no_sex_approx':
            return self.ckdepi_no_sex_approx(serum_creatinine, age)
        elif method == 'combined_approach':
            return self.combined_approach(serum_creatinine, age)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_egfr_category(self, egfr):
        """
        Assign eGFR-based severity category based on eGFR value.
        
        NOTE: These are eGFR-based categories, NOT clinical CKD stages.
        Clinical CKD diagnosis requires evidence of kidney damage (proteinuria,
        structural abnormalities, etc.) in addition to eGFR values.
        
        Categories based on KDIGO eGFR ranges:
        Category 1: eGFR >= 90 (normal or high kidney function)
        Category 2: eGFR 60-89 (mildly decreased kidney function)
        Category 3: eGFR 30-59 (moderately decreased kidney function)
        Category 4: eGFR 15-29 (severely decreased kidney function)
        Category 5: eGFR < 15 (kidney failure)
        
        Args:
            egfr: eGFR value in mL/min/1.73m²
            
        Returns:
            eGFR-based severity category (1-5) as integer
        """
        if egfr >= 90:
            return 1
        elif egfr >= 60:
            return 2
        elif egfr >= 30:
            return 3
        elif egfr >= 15:
            return 4
        else:
            return 5
    
    def get_category_description(self, category):
        """
        Get description of eGFR-based severity category.
        
        Args:
            category: eGFR-based severity category (1-5)
            
        Returns:
            String description of the category
        """
        descriptions = {
            1: "Category 1: Normal or high kidney function (eGFR ≥ 90)",
            2: "Category 2: Mildly decreased kidney function (eGFR 60-89)",
            3: "Category 3: Moderately decreased kidney function (eGFR 30-59)",
            4: "Category 4: Severely decreased kidney function (eGFR 15-29)",
            5: "Category 5: Kidney failure (eGFR < 15)"
        }
        return descriptions.get(category, "Unknown category")


def validate_egfr_inputs(df):
    """
    Validate that required inputs for eGFR calculation are present.
    
    Args:
        df: DataFrame with clinical data
        
    Returns:
        tuple: (is_valid, missing_vars, available_vars)
    """
    required_vars = ['sc', 'age']
    optional_vars = ['sex']  # Not available in this dataset
    
    missing_vars = []
    available_vars = []
    
    for var in required_vars:
        if var in df.columns:
            available_vars.append(var)
        else:
            missing_vars.append(var)
    
    for var in optional_vars:
        if var in df.columns:
            available_vars.append(var)
        else:
            missing_vars.append(var)
    
    is_valid = len(missing_vars) == 0 or all(var in optional_vars for var in missing_vars)
    
    return is_valid, missing_vars, available_vars


if __name__ == "__main__":
    # Test the calculator
    calculator = EGFRCalculator()
    
    # Test cases
    test_cases = [
        (1.2, 55),  # Normal adult
        (2.5, 60),  # Moderate kidney function decrease
        (5.0, 70),  # Severe kidney function decrease
        (0.8, 30),  # Young adult
    ]
    
    print("eGFR Calculator Test")
    print("="*60)
    for sc, age in test_cases:
        egfr = calculator.calculate_egfr(sc, age)
        category = calculator.get_egfr_category(egfr)
        description = calculator.get_category_description(category)
        print(f"SCr={sc}, Age={age} -> eGFR={egfr:.1f}, Category={category}")
        print(f"  {description}")
        print()
