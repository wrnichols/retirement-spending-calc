#!/usr/bin/env python3

import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

try:
    import streamlit_app as app
    print("Successfully imported streamlit_app")
    
    # Test the updated function
    inputs = {
        'current_age': 60,
        'years_to_ret': 5,
        'years_to_ss': 7,
        'current_assets': 200000,
        'risk_tolerance': 'Aggressive',
        'desired_monthly_spending': 10000,
        'ss_min': 1605*12, 
        'ss_fra': 2500*12, 
        'ss_max': 2750*12, 
        'fra_age': 67,
        'ltc_insurance': 100000, 
        'pension_annual': 0, 
        'other_income': 0
    }
    
    print("Testing updated retirement calculator...")
    results = app.retirement_spending_calculator(**inputs)
    
    print("Calculator function executed successfully")
    print(f"Ultra-conservative (99%): ${results['viable_spending_monthly']['lower_bound_99_percent']:,.0f}/month")
    print(f"Conservative (95%): ${results['viable_spending_monthly']['base_95_percent']:,.0f}/month") 
    print(f"Moderate (90%): ${results['viable_spending_monthly']['upper_bound_90_percent']:,.0f}/month")
    print(f"Assets at retirement: ${results['assets_at_ret']:,.0f}")
    print(f"LTC suggested coverage: ${results['ltc_coverage']['suggested_annual']:,.0f}/year")
    print(f"LTC warning: {results['ltc_coverage']['warning']}")
    print(f"Desired spending realistic: {results['desired_monthly_spending_status']['is_realistic']}")
    print(f"Status: {results['desired_monthly_spending_status']['status']}")
    print(f"Required assets: {results['desired_monthly_spending_status']['required_assets_estimate']}")
    print(f"Projected legacy: ${results['projected_legacy']:,.0f}")
    
    print("\nAll tests passed successfully!")

except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()