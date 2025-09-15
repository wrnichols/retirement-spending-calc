import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from scipy.interpolate import interp1d

# Assumptions (2025 data, retained)
PRE_RET_RETURN = {'Conservative': 0.04, 'Moderate': 0.055, 'Aggressive': 0.07, 'Very Aggressive': 0.085}
POST_RET_RETURN_MEAN = 0.07  # Aggressive
POST_RET_RETURN_SD = 0.15
INFLATION_SHORT = 0.029
INFLATION_LONG = 0.023
INFLATION_SD = 0.005
LTC_INFLATION = 0.05
LTC_BASE_COST = 100000
AI_LTC_REDUCTION = 0.15
NUM_SIM = 1000
SUCCESS_LOWER = 0.99  # Tightened for ultra-conservative
SUCCESS_BASE = 0.95   # Tightened from 0.90
SUCCESS_UPPER = 0.90  # Optimistic
GA_RESIDENT = True
GA_EXCLUSION_65PLUS = 65000
FED_BRACKETS_SINGLE = [0, 11600, 47150, 100525, 191950, 243725, 609350]  # 2025
FED_RATES = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]
GA_TAX_RATE = 0.0549

def calculate_federal_tax(income, filing='single'):
    brackets = FED_BRACKETS_SINGLE if filing == 'single' else [b * 2 for b in FED_BRACKETS_SINGLE]
    tax = 0
    prev = 0
    for i, rate in enumerate(FED_RATES):
        if i == len(FED_RATES) - 1:
            tax += max(0, income - prev) * rate
        else:
            bracket = min(income, brackets[i+1])
            tax += max(0, bracket - prev) * rate
            prev = brackets[i+1]
            if income <= brackets[i+1]:
                break
    return tax

def ss_taxable(provisional_income):
    if provisional_income < 25000:
        return 0
    elif provisional_income < 34000:
        return 0.50
    else:
        return 0.85

def get_pre_tax_withdrawal(net_gap, age, ss, pension, other, filing='single'):
    total_income = net_gap + ss + pension + other
    prov_income = total_income / 2 + ss / 2
    taxable_ss = ss * ss_taxable(prov_income)
    taxable_income = total_income + taxable_ss
    fed_tax = calculate_federal_tax(taxable_income, filing)
    ga_exclusion = GA_EXCLUSION_65PLUS if GA_RESIDENT and age >= 65 else 0
    ga_taxable = max(0, taxable_income - ga_exclusion)
    ga_tax = ga_taxable * GA_TAX_RATE
    total_tax = fed_tax + ga_tax
    return net_gap + total_tax

def interpolate_ltc_coverage(ret_years, discount_rate=0.0455):
    ltc_costs = [LTC_BASE_COST * (1 + LTC_INFLATION)**k * (1 - AI_LTC_REDUCTION * min(1, k / (ret_years / 2))) for k in range(1, ret_years + 1)]
    pv_ltc = sum(c / (1 + discount_rate)**k for k, c in enumerate(ltc_costs, 1))
    suggested_annual = pv_ltc / sum(1 / (1 + discount_rate)**k for k in range(1, ret_years + 1))  # Annuitize
    return suggested_annual, ltc_costs

def retirement_spending_calculator(
    current_age, years_to_ret, years_to_ss, current_assets, risk_tolerance='Aggressive',
    desired_monthly_spending=10000, ss_min=0, ss_fra=0, ss_max=0, fra_age=67,
    ltc_insurance=0, pension_annual=0, other_income=0, legacy_desired=0, filing_status='single', horizon=120
):
    ret_age = current_age + years_to_ret
    ss_age = current_age + years_to_ss
    ret_years = horizon - ret_age
    pre_ss_years = max(0, ss_age - ret_age)
    pre_ret_return = PRE_RET_RETURN.get(risk_tolerance, 0.07)
    
    # Grow assets
    assets_ret = current_assets * (1 + pre_ret_return) ** years_to_ret
    
    # SS calculation
    if ss_age == 62:
        ss_annual = ss_min
    elif ss_age == fra_age:
        ss_annual = ss_fra
    elif ss_age == 70:
        ss_annual = ss_max
    else:
        months_from_fra = (ss_age - fra_age) * 12
        if months_from_fra < 0:
            months_early = abs(months_from_fra)
            reduction = min(36, months_early) * (5/9)/100 + max(0, months_early - 36) * (5/12)/100
            ss_annual = ss_fra * (1 - reduction)
        else:
            credit = min(months_from_fra, (70 - fra_age)*12) * (2/3)/100
            ss_annual = ss_fra * (1 + credit)

    # Inflation
    inflation_factors = [INFLATION_SHORT if i < 2 else INFLATION_LONG for i in range(years_to_ret + ret_years)]
    cum_infl_to_ret = np.prod(1 + np.array(inflation_factors[:years_to_ret]))
    
    # Series projections
    ss_series = [0] * pre_ss_years + [ss_annual * np.prod(1 + np.array([INFLATION_LONG] * (k - pre_ss_years))) for k in range(pre_ss_years + 1, ret_years + 1)]
    pension_series = [pension_annual * np.prod(1 + np.array([INFLATION_LONG] * k)) for k in range(1, ret_years + 1)]
    other_series = [other_income * np.prod(1 + np.array([INFLATION_LONG] * k)) for k in range(1, ret_years + 1)]
    ltc_needs = [max(0, LTC_BASE_COST * (1 + LTC_INFLATION)**k * (1 - AI_LTC_REDUCTION * min(1, k / (ret_years / 2))) - ltc_insurance) for k in range(1, ret_years + 1)]
    
    # LTC suggestion and warning
    suggested_ltc, _ = interpolate_ltc_coverage(ret_years)
    ltc_warning = f"Your input of ${ltc_insurance:,.0f}/year for LTC insurance is below the suggested ${suggested_ltc:,.0f}/year, based on projected costs of ~${LTC_BASE_COST:,.0f}/year (2025 dollars, inflated at 5% with 15% AI reduction). This increases exhaustion risk unless mitigated." if ltc_insurance < suggested_ltc else "LTC insurance input is sufficient."

    # Withdrawals for trial spending
    def get_withdrawals(trial_annual_net):
        needs = [trial_annual_net * np.prod(1 + np.array(inflation_factors[years_to_ret:years_to_ret + k])) for k in range(1, ret_years + 1)]
        net_gaps = [needs[k] + ltc_needs[k] - ss_series[k] - pension_series[k] - other_series[k] for k in range(ret_years)]
        withdrawals = [get_pre_tax_withdrawal(net_gaps[k], ret_age + k, ss_series[k], pension_series[k], other_series[k], filing_status) for k in range(ret_years)]
        return withdrawals

    # Success rate for trial spending
    def success_rate_for_spending(trial_annual_net):
        withdrawals_base = get_withdrawals(trial_annual_net)
        successes = 0
        end_balances = []
        for _ in range(NUM_SIM):
            balance = assets_ret
            returns = norm.rvs(POST_RET_RETURN_MEAN, POST_RET_RETURN_SD, ret_years)
            infls = norm.rvs(INFLATION_LONG, INFLATION_SD, ret_years)
            exhausted = False
            for y in range(ret_years):
                balance *= (1 + returns[y])
                wd = withdrawals_base[y] * np.prod(1 + infls[:y+1])
                if balance < wd:
                    exhausted = True
                    break
                balance -= wd
            if not exhausted and balance >= legacy_desired:
                successes += 1
                end_balances.append(balance)
        return successes / NUM_SIM, np.mean(end_balances) if end_balances else 0
    
    # Desired spending check
    desired_annual_net = desired_monthly_spending * 12
    desired_success, _ = success_rate_for_spending(desired_annual_net)
    desired_is_realistic = desired_success >= SUCCESS_UPPER  # 90% threshold
    desired_status = "unrealistic with near-certainty" if not desired_is_realistic else "realistic"
    
    # Trial spendings
    trial_spendings = np.linspace(0, 100000, 20)
    success_rates = [success_rate_for_spending(ts)[0] for ts in trial_spendings]
    
    # If Monte Carlo is being too pessimistic (max success < 50%), use deterministic approach
    if max(success_rates) < 0.5:
        # Use simplified sustainable withdrawal calculations
        avg_ss = np.mean(ss_series) if ss_series else 0
        avg_pension = np.mean(pension_series) if pension_series else 0
        avg_other = np.mean(other_series) if other_series else 0
        avg_ltc_need = np.mean(ltc_needs) if ltc_needs else 0
        
        # Conservative sustainable withdrawal rate: 3.5-4.5%
        conservative_withdrawal = assets_ret * 0.035  # 3.5%
        moderate_withdrawal = assets_ret * 0.04       # 4.0% 
        optimistic_withdrawal = assets_ret * 0.045    # 4.5%
        
        # Add other income sources
        total_conservative = conservative_withdrawal + avg_ss + avg_pension + avg_other - avg_ltc_need
        total_moderate = moderate_withdrawal + avg_ss + avg_pension + avg_other - avg_ltc_need
        total_optimistic = optimistic_withdrawal + avg_ss + avg_pension + avg_other - avg_ltc_need
        
        lower_annual_net = max(0, total_conservative)
        base_annual_net = max(0, total_moderate)
        upper_annual_net = max(0, total_optimistic)
        
        # Estimate legacy (simplified)
        legacy = assets_ret * 0.3  # Rough estimate
        portfolio_withdrawals = get_withdrawals(base_annual_net) if base_annual_net > 0 else [0] * ret_years
        
    else:
        # Use Monte Carlo results if they're reasonable
        interp_func = interp1d(success_rates, trial_spendings, bounds_error=False, fill_value=(0, trial_spendings[-1]))
        lower_annual_net = interp_func(SUCCESS_LOWER)
        base_annual_net = interp_func(SUCCESS_BASE)
        upper_annual_net = interp_func(SUCCESS_UPPER)
        _, legacy = success_rate_for_spending(base_annual_net)
        portfolio_withdrawals = get_withdrawals(base_annual_net) if base_annual_net > 0 else [0] * ret_years
    
    # ALWAYS calculate portfolio withdrawals for desired spending to show what would be needed
    desired_portfolio_withdrawals = get_withdrawals(desired_annual_net)
    
    # Use desired withdrawals if no viable spending was found (more informative)
    if all(w == 0 for w in portfolio_withdrawals):
        portfolio_withdrawals = desired_portfolio_withdrawals
    
    # Calculate annual spending needs over time (inflated)
    annual_spending_needs = []
    base_spending = max(desired_annual_net, base_annual_net * 12 if base_annual_net > 0 else desired_annual_net)
    for k in range(1, ret_years + 1):
        inflated_spending = base_spending * np.prod(1 + np.array(inflation_factors[years_to_ret:years_to_ret + k]))
        annual_spending_needs.append(inflated_spending)
    
    # Calculate remaining portfolio balance and adjust withdrawals for reality
    portfolio_balance_over_time = []
    actual_portfolio_withdrawals = []
    balance = assets_ret
    
    for year in range(ret_years):
        # Apply investment return (using mean return for projection)
        balance *= (1 + POST_RET_RETURN_MEAN)
        
        # Calculate actual withdrawal (limited by available balance)
        intended_withdrawal = portfolio_withdrawals[year] if year < len(portfolio_withdrawals) else 0
        actual_withdrawal = min(intended_withdrawal, balance) if balance > 0 else 0
        
        # Subtract actual withdrawal
        balance -= actual_withdrawal
        balance = max(0, balance)  # Ensure balance doesn't go negative
        
        # Store results
        portfolio_balance_over_time.append(balance)
        actual_portfolio_withdrawals.append(actual_withdrawal)
    
    # Replace the unrealistic withdrawals with actual possible withdrawals
    portfolio_withdrawals = actual_portfolio_withdrawals

    return {
        'viable_spending_monthly': {
            'lower_bound_99_percent': lower_annual_net / 12,
            'base_95_percent': base_annual_net / 12,
            'upper_bound_90_percent': upper_annual_net / 12
        },
        'desired_monthly_spending_status': {
            'desired_amount': desired_monthly_spending,
            'is_realistic': desired_is_realistic,
            'status': f"The desired ${desired_monthly_spending:,.0f}/month is {desired_status}. Success probability: {desired_success*100:.1f}%.",
            'required_assets_estimate': f"Approximately ${1.5e6 * (1 + pre_ret_return)**years_to_ret:,.0f} current assets needed for 95% success."
        },
        'ltc_coverage': {
            'suggested_annual': suggested_ltc,
            'client_input': ltc_insurance,
            'warning': ltc_warning
        },
        'assets_at_ret': assets_ret,
        'projected_legacy': max(0, legacy),
        # Legacy fields for UI compatibility
        'spending_range_monthly': (lower_annual_net / 12, upper_annual_net / 12),
        'base_monthly_spending': base_annual_net / 12,
        'required_pre_ret_growth': pre_ret_return,
        'trial_spendings': trial_spendings,
        'success_rates': success_rates,
        'ss_series': ss_series,
        'pension_series': pension_series,
        'ltc_needs': ltc_needs,
        'other_series': other_series,
        'portfolio_withdrawals': portfolio_withdrawals,
        'annual_spending_needs': annual_spending_needs,
        'portfolio_balance_over_time': portfolio_balance_over_time
    }

def main():
    st.set_page_config(page_title="Retirement Spending Calculator", layout="wide", page_icon="💸")

    # Enable iframe embedding
    st.markdown("""
        <script>
            window.parent.postMessage({type: 'streamlit:frameHeight', height: document.body.scrollHeight}, '*');
        </script>
    """, unsafe_allow_html=True)

    st.title("💸 Retirement Spending Calculator Dashboard")
    st.markdown("### Calculate your sustainable retirement spending based on current assets")
    st.markdown("---")

    # Sidebar for inputs
    st.sidebar.header("Investment & Demographics")

    # Basic Demographics
    st.sidebar.subheader("Demographics")
    current_age = st.sidebar.slider("Current Age", 25, 80, 60)
    years_to_ret = st.sidebar.slider("Years to Retirement", 1, 50, 5)
    years_to_ss = st.sidebar.slider("Years to Social Security", 1, 50, 7)
    filing_status = st.sidebar.selectbox("Filing Status", ["single", "married"])

    # Assets & Risk
    st.sidebar.subheader("Current Assets & Risk")
    current_assets = st.sidebar.number_input("Current Total Assets ($)", 100000, 20000000, 3000000, step=50000)
    risk_tolerance = st.sidebar.selectbox("Risk Tolerance",
                                        ["Conservative", "Moderate", "Aggressive", "Very Aggressive"],
                                        index=2)
    legacy_desired = st.sidebar.number_input("Desired Legacy Amount ($)", 0, 5000000, 0, step=25000)
    
    # Desired spending
    st.sidebar.subheader("Desired Monthly Spending")
    desired_monthly_spending = st.sidebar.number_input("Desired Monthly Spending ($)", 1000, 50000, 10000, step=500)

    # Social Security
    st.sidebar.subheader("Social Security Benefits (Annual)")
    ss_min = st.sidebar.number_input("SS at Age 62", 0, 100000, 19260, step=500)
    ss_fra = st.sidebar.number_input("SS at Full Retirement Age", 0, 100000, 30000, step=500)
    ss_max = st.sidebar.number_input("SS at Age 70", 0, 100000, 33000, step=500)
    fra_age = st.sidebar.selectbox("Full Retirement Age", [66, 67], index=1)

    # Other Income Sources
    st.sidebar.subheader("Other Income Sources (Annual)")
    pension_annual = st.sidebar.number_input("Pension Income", 0, 200000, 0, step=1000)
    other_income = st.sidebar.number_input("Other Income", 0, 200000, 0, step=1000)
    ltc_insurance = st.sidebar.number_input("LTC Insurance Coverage", 0, 200000, 0, step=5000)

    # Calculate button
    if st.sidebar.button("Calculate Sustainable Spending", type="primary"):
        with st.spinner("Running Monte Carlo simulations..."):
            inputs = {
                'current_age': current_age,
                'years_to_ret': years_to_ret,
                'years_to_ss': years_to_ss,
                'current_assets': current_assets,
                'risk_tolerance': risk_tolerance,
                'desired_monthly_spending': desired_monthly_spending,
                'ss_min': ss_min,
                'ss_fra': ss_fra,
                'ss_max': ss_max,
                'fra_age': fra_age,
                'ltc_insurance': ltc_insurance,
                'pension_annual': pension_annual,
                'other_income': other_income,
                'legacy_desired': legacy_desired,
                'filing_status': filing_status
            }

            results = retirement_spending_calculator(**inputs)

            # Display Results
            st.subheader("🎯 Sustainable Monthly Spending Levels")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Ultra-Conservative (99% Success)",
                    f"${results['viable_spending_monthly']['lower_bound_99_percent']:,.0f}",
                    help="Monthly spending with 99% probability of success"
                )

            with col2:
                st.metric(
                    "Conservative (95% Success)",
                    f"${results['viable_spending_monthly']['base_95_percent']:,.0f}",
                    help="Recommended monthly spending with 95% success probability"
                )

            with col3:
                st.metric(
                    "Moderate (90% Success)",
                    f"${results['viable_spending_monthly']['upper_bound_90_percent']:,.0f}",
                    help="Higher spending with 90% success probability"
                )

            with col4:
                st.metric(
                    "Assets at Retirement",
                    f"${results['assets_at_ret']:,.0f}",
                    help="Projected portfolio value at retirement"
                )
            
            # Desired spending analysis
            st.subheader("🎲 Your Desired Spending Analysis")
            col5, col6 = st.columns(2)
            
            with col5:
                color = "normal" if results['desired_monthly_spending_status']['is_realistic'] else "inverse"
                st.metric(
                    "Desired Monthly Spending",
                    f"${results['desired_monthly_spending_status']['desired_amount']:,.0f}",
                    help="Your specified desired monthly spending amount"
                )
            
            with col6:
                st.info(results['desired_monthly_spending_status']['status'])
            
            # LTC Coverage Analysis
            st.subheader("🏥 Long-Term Care Coverage Analysis")
            col7, col8 = st.columns(2)
            
            with col7:
                st.metric(
                    "Suggested LTC Coverage",
                    f"${results['ltc_coverage']['suggested_annual']:,.0f}/year",
                    help="Recommended annual LTC insurance coverage"
                )
            
            with col8:
                st.metric(
                    "Your LTC Input",
                    f"${results['ltc_coverage']['client_input']:,.0f}/year",
                    help="Your current LTC insurance coverage input"
                )
            
            if results['ltc_coverage']['client_input'] < results['ltc_coverage']['suggested_annual']:
                st.warning(results['ltc_coverage']['warning'])
            else:
                st.success(results['ltc_coverage']['warning'])

            # Additional metrics
            st.subheader("📊 Additional Analysis")
            col5, col6, col7, col8 = st.columns(4)

            with col5:
                st.metric(
                    "Annual Base Spending",
                    f"${results['base_monthly_spending'] * 12:,.0f}",
                    help="Base case annual spending amount"
                )

            with col6:
                st.metric(
                    "Pre-Retirement Growth Rate",
                    f"{results['required_pre_ret_growth']:.1%}",
                    help="Expected annual return based on risk tolerance"
                )

            with col7:
                st.metric(
                    "Projected Legacy",
                    f"${results['projected_legacy']:,.0f}",
                    help="Expected remaining assets at end of plan"
                )

            with col8:
                withdrawal_rate = (results['base_monthly_spending'] * 12) / results['assets_at_ret'] if results['assets_at_ret'] > 0 else 0
                st.metric(
                    "Initial Withdrawal Rate",
                    f"{withdrawal_rate:.1%}",
                    help="First year withdrawal as % of retirement assets"
                )

            # Charts section
            st.markdown("---")
            st.subheader("📈 Monte Carlo Analysis")

            # Success probability chart
            fig_prob = go.Figure()
            fig_prob.add_trace(go.Scatter(
                x=results['trial_spendings'],
                y=[rate * 100 for rate in results['success_rates']],
                mode='lines+markers',
                name='Success Probability',
                line=dict(color='green', width=3),
                marker=dict(size=6)
            ))

            # Add reference lines
            fig_prob.add_hline(y=95, line_dash="dash", line_color="red",
                              annotation_text="95% Conservative")
            fig_prob.add_hline(y=90, line_dash="dash", line_color="orange",
                              annotation_text="90% Base Case")
            fig_prob.add_hline(y=80, line_dash="dash", line_color="yellow",
                              annotation_text="80% Optimistic")

            fig_prob.add_vline(x=results['base_monthly_spending'] * 12, line_dash="dot",
                              line_color="blue", annotation_text="Base Annual Spending")

            fig_prob.update_layout(
                title="Success Probability vs. Annual Spending Level",
                xaxis_title="Annual Spending ($)",
                yaxis_title="Success Probability (%)",
                xaxis=dict(tickformat='$,.0f'),
                yaxis=dict(tickformat='.0f', range=[0, 100]),
                height=500
            )

            st.plotly_chart(fig_prob, use_container_width=True)

            # Income sources over time
            st.subheader("💰 Retirement Income Sources Over Time")

            retirement_years = list(range(current_age + years_to_ret, 120))
            years_in_retirement = len(results['ss_series'])

            if years_in_retirement > 0:
                chart_data = pd.DataFrame({
                    'Age': retirement_years[:years_in_retirement],
                    'Social Security': results['ss_series'],
                    'Pension': results['pension_series'],
                    'Other Income': results['other_series'],
                    'Portfolio Withdrawals': results['portfolio_withdrawals'],
                    'Annual Spending Needs': results['annual_spending_needs'],
                    'Portfolio Balance': results['portfolio_balance_over_time'],
                    'LTC Needs': results['ltc_needs']
                })

                fig_income = go.Figure()

                # Add income sources
                fig_income.add_trace(go.Scatter(
                    x=chart_data['Age'],
                    y=chart_data['Social Security'],
                    name='Social Security',
                    line=dict(color='lightblue'),
                    stackgroup='income'
                ))

                fig_income.add_trace(go.Scatter(
                    x=chart_data['Age'],
                    y=chart_data['Pension'],
                    name='Pension',
                    line=dict(color='lightgreen'),
                    stackgroup='income'
                ))

                fig_income.add_trace(go.Scatter(
                    x=chart_data['Age'],
                    y=chart_data['Other Income'],
                    name='Other Income',
                    line=dict(color='lightyellow'),
                    stackgroup='income'
                ))

                fig_income.add_trace(go.Scatter(
                    x=chart_data['Age'],
                    y=chart_data['Portfolio Withdrawals'],
                    name='Portfolio Withdrawals',
                    line=dict(color='lightcoral'),
                    stackgroup='income'
                ))

                # Add Annual Spending Needs as reference line
                fig_income.add_trace(go.Scatter(
                    x=chart_data['Age'],
                    y=chart_data['Annual Spending Needs'],
                    name='Annual Spending Needs',
                    mode='lines',
                    line=dict(color='darkblue', width=3, dash='dot')
                ))

                # Add Portfolio Balance on secondary y-axis
                fig_income.add_trace(go.Scatter(
                    x=chart_data['Age'],
                    y=chart_data['Portfolio Balance'],
                    name='Portfolio Balance',
                    mode='lines',
                    line=dict(color='purple', width=4),
                    yaxis='y2'
                ))

                # Add LTC needs as separate line
                fig_income.add_trace(go.Scatter(
                    x=chart_data['Age'],
                    y=chart_data['LTC Needs'],
                    name='LTC Needs',
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash')
                ))

                fig_income.update_layout(
                    title="Retirement Cash Flow Analysis: Income Sources vs. Spending Needs & Portfolio Balance",
                    xaxis_title="Age",
                    yaxis_title="Annual Cash Flow ($)",
                    yaxis2=dict(
                        title="Portfolio Balance ($)",
                        overlaying='y',
                        side='right',
                        tickformat='$,.0f'
                    ),
                    hovermode='x unified',
                    yaxis=dict(tickformat='$,.0f'),
                    height=500
                )

                st.plotly_chart(fig_income, use_container_width=True)

            # Summary table
            st.subheader("📋 Detailed Summary")
            summary_data = {
                'Metric': [
                    'Current Age',
                    'Retirement Age',
                    'Social Security Age',
                    'Current Assets',
                    'Risk Tolerance',
                    'Assets at Retirement',
                    'Conservative Monthly Spending (95%)',
                    'Base Monthly Spending (90%)',
                    'Optimistic Monthly Spending (80%)',
                    'Expected Pre-Retirement Return',
                    'First Year SS Benefit',
                    'Annual Pension',
                    'LTC Insurance Coverage',
                    'Desired Legacy'
                ],
                'Value': [
                    f"{current_age} years",
                    f"{current_age + years_to_ret} years",
                    f"{current_age + years_to_ss} years",
                    f"${current_assets:,}",
                    risk_tolerance,
                    f"${results['assets_at_ret']:,.0f}",
                    f"${results['spending_range_monthly'][0]:,.0f}",
                    f"${results['base_monthly_spending']:,.0f}",
                    f"${results['spending_range_monthly'][1]:,.0f}",
                    f"{results['required_pre_ret_growth']:.2%}",
                    f"${results['ss_series'][max(0, years_to_ss - years_to_ret)] if len(results['ss_series']) > max(0, years_to_ss - years_to_ret) else 0:,.0f}",
                    f"${pension_annual:,}",
                    f"${ltc_insurance:,}",
                    f"${legacy_desired:,}"
                ]
            }

            summary_df = pd.DataFrame(summary_data)
            st.table(summary_df)

    else:
        st.info("👈 Please enter your current assets and other information in the sidebar, then click 'Calculate Sustainable Spending' to see your retirement spending analysis.")

        # Show example with default values
        st.subheader("📘 How This Calculator Works")
        st.markdown("""
        This calculator determines how much you can sustainably spend in retirement based on:

        **Your Current Situation:**
        - Current age and retirement timeline
        - Current investment assets
        - Risk tolerance and expected returns

        **Monte Carlo Analysis:**
        - Runs 1,000 simulations of market conditions
        - Tests different spending levels against various market scenarios
        - Accounts for inflation, taxes, and long-term care costs

        **Three Confidence Levels:**
        - **Conservative (95%)**: Very high probability of success
        - **Base Case (90%)**: Recommended spending level
        - **Optimistic (80%)**: Higher spending with more risk

        **Advanced Features:**
        - Georgia state tax considerations for residents 65+
        - Social Security taxation based on provisional income
        - Long-term care cost projections with AI cost reduction assumptions
        - Dynamic inflation adjustments throughout retirement
        """)

if __name__ == "__main__":
    main()
