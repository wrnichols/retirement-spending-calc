#!/usr/bin/env python3

import sys
import os
import numpy as np

# Add current directory to path
sys.path.insert(0, os.getcwd())

try:
    import streamlit_app as app
    
    # Test the function with debugging
    inputs = {
        'current_age': 60,
        'years_to_ret': 5,
        'years_to_ss': 7,
        'current_assets': 500000,
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
    
    print("Debug: Testing success rates directly...")
    
    # Test with 1M assets
    inputs['current_assets'] = 1000000
    
    # Let's manually test some trial spending levels
    result = app.retirement_spending_calculator(**inputs)
    print(f"Assets at retirement: ${result['assets_at_ret']:,.0f}")
    print(f"Trial spendings range: {result['trial_spendings'][:5]} ... {result['trial_spendings'][-5:]}")
    print(f"Success rates range: {result['success_rates'][:5]} ... {result['success_rates'][-5:]}")
    print(f"Max success rate: {max(result['success_rates']):.3f}")
    print(f"SUCCESS_LOWER threshold: {app.SUCCESS_LOWER}")
    
    # Test a simple case manually
    ret_calc = app.retirement_spending_calculator
    
    # Try with very low spending to see if we get any success
    simple_inputs = inputs.copy()
    simple_inputs['current_assets'] = 1000000
    simple_inputs['ltc_insurance'] = 0  # No LTC for simplicity
    simple_inputs['desired_monthly_spending'] = 2000  # Lower desired spending
    
    simple_result = ret_calc(**simple_inputs)
    print(f"\nSimple test results:")
    print(f"99%: ${simple_result['viable_spending_monthly']['lower_bound_99_percent']:,.2f}")
    print(f"95%: ${simple_result['viable_spending_monthly']['base_95_percent']:,.2f}") 
    print(f"90%: ${simple_result['viable_spending_monthly']['upper_bound_90_percent']:,.2f}")

except Exception as e:
    print(f"Error during debugging: {e}")
    import traceback
    traceback.print_exc()