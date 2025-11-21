#!/usr/bin/env python3
"""
Monte Carlo Simulation for Real Estate-Backed Credit Product
==============================================================

This script simulates a structured credit product backed by real estate loans:
- 10 crore corpus from 10 investors (1 crore each, pro-rata rights)
- Corpus lent to pool of real estate projects as secured loans
- 10-year horizon with stochastic interest rates, collateral values, and defaults

FINANCIAL INTUITION:
-------------------
1. BGM-Style Interest Rate Model:
   - Models forward rates as lognormal processes (classic LIBOR Market Model approach)
   - Captures term structure dynamics and stochastic discounting
   - Forward rates drive both loan pricing and investor cash flow valuation
   - Single-factor simplification: all rates driven by one Brownian motion
   
2. Real Estate Collateral (GBM):
   - Property values follow Geometric Brownian Motion
   - Reflects uncertainty in real estate appreciation/depreciation
   - Correlation structure captures systemic real estate market risk
   - Individual project idiosyncratic risk layered on top
   
3. Default Modeling:
   - Base default probability reflects credit quality
   - Collateral coverage (LTV) modulates default risk dynamically
   - Lower collateral → higher default probability (structural model intuition)
   - Recovery linked to liquidation of collateral at market value
   
4. Investor IRR Distribution:
   - Shows risk-return profile of the product
   - Captures credit risk, interest rate risk, and real estate market risk
   - Allows stress testing and scenario analysis

LIMITATIONS & SIMPLIFICATIONS:
-----------------------------
- Single-factor interest rate model (no term structure twists)
- Simple correlation structure (exponential decay or constant)
- Rough calibration (not fitted to market data)
- No prepayment modeling
- No operational costs beyond basic liquidation
- Equal investor rights (no tranching or waterfalls)
- Annual time steps (can be refined to quarterly/monthly)
- Taxes and fees simplified or omitted

Author: Senior Quant Developer
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

class SimulationConfig:
    """Central configuration for all simulation parameters."""
    
    # === Pool Structure ===
    TOTAL_CORPUS = 100_000_000  # 10 crore in rupees
    NUM_INVESTORS = 10
    INVESTMENT_PER_INVESTOR = TOTAL_CORPUS / NUM_INVESTORS
    NUM_PROJECTS = 10
    LOAN_PER_PROJECT = TOTAL_CORPUS / NUM_PROJECTS
    
    # === Time Grid ===
    HORIZON_YEARS = 10
    TIME_STEPS_PER_YEAR = 1  # Annual steps (change to 2 for semi-annual, 4 for quarterly)
    DT = 1.0 / TIME_STEPS_PER_YEAR
    NUM_STEPS = HORIZON_YEARS * TIME_STEPS_PER_YEAR
    
    # === Interest Rate Model (BGM-style) ===
    INITIAL_FORWARD_RATE = 0.08  # 8% flat forward curve
    FORWARD_RATE_DRIFT = 0.0     # Drift in forward rate (often 0 under risk-neutral measure)
    FORWARD_RATE_VOL = 0.15      # 15% volatility for forward rates
    FORWARD_RATE_CORR = 0.80     # Correlation between adjacent forward rates
    
    # === Loan Terms ===
    LOAN_COUPON = 0.12           # 12% annual coupon
    LOAN_MATURITY_YEARS = 10
    AMORTIZATION = False         # False = bullet payment, True = equal amortization
    
    # === Collateral (Real Estate) Dynamics ===
    INITIAL_COLLATERAL_PER_PROJECT = 20_000_000  # 2 crore per project (200% collateral coverage)
    RE_DRIFT = 0.05              # 5% expected annual appreciation
    RE_VOL = 0.15                # 15% volatility
    RE_SYSTEMIC_FACTOR = 0.60    # 60% of variance from systemic factor
    RE_IDIOSYNCRATIC_FACTOR = 0.40  # 40% idiosyncratic
    
    # === Default Model ===
    BASE_DEFAULT_PROB = 0.03     # 3% annual base default probability
    DEFAULT_THRESHOLD_1 = 1.2    # If collateral/loan < 1.2, increase default prob
    DEFAULT_THRESHOLD_2 = 1.0    # If collateral/loan < 1.0, increase further
    DEFAULT_PROB_MULT_1 = 2.0    # Multiplier when below threshold 1
    DEFAULT_PROB_MULT_2 = 4.0    # Multiplier when below threshold 2
    
    # === Recovery ===
    RECOVERY_RATE = 0.70         # 70% of collateral value recovered
    LIQUIDATION_COST_RATE = 0.05 # 5% liquidation costs
    
    # === Monte Carlo ===
    NUM_SIMULATIONS = 5000       # Number of paths (reduce to 500 for quick tests)
    RANDOM_SEED = 42             # For reproducibility
    
    # === Output ===
    OUTPUT_DIR = "simulation_output"
    SAVE_PLOTS = True


# ============================================================================
# INTEREST RATE MODEL (BGM-STYLE)
# ============================================================================

def simulate_forward_rates(config: SimulationConfig, num_sims: int) -> np.ndarray:
    """
    Simulate forward rates using a simplified BGM (LIBOR Market Model) approach.
    
    Uses a single-factor lognormal model for forward rates:
    df_t = mu * f_t * dt + sigma * f_t * dW_t
    
    Parameters:
    -----------
    config : SimulationConfig
    num_sims : int - number of Monte Carlo paths
    
    Returns:
    --------
    forward_rates : np.ndarray of shape (num_sims, num_steps+1, num_tenors)
        Forward rates for each path, time step, and tenor
    """
    np.random.seed(config.RANDOM_SEED)
    
    num_steps = config.NUM_STEPS
    dt = config.DT
    
    # Define forward rate tenors (one per time step in our simple model)
    num_tenors = num_steps + 1
    
    # Initialize forward rates (all start at initial flat curve)
    forward_rates = np.zeros((num_sims, num_steps + 1, num_tenors))
    forward_rates[:, 0, :] = config.INITIAL_FORWARD_RATE
    
    # Correlation matrix for forward rates (exponential decay with distance)
    correlation_matrix = np.zeros((num_tenors, num_tenors))
    for i in range(num_tenors):
        for j in range(num_tenors):
            distance = abs(i - j)
            correlation_matrix[i, j] = config.FORWARD_RATE_CORR ** distance
    
    # Cholesky decomposition for correlated random numbers
    chol_matrix = np.linalg.cholesky(correlation_matrix)
    
    # Simulate forward rate paths
    for step in range(num_steps):
        # Generate correlated random shocks
        z_independent = np.random.standard_normal((num_sims, num_tenors))
        z_correlated = z_independent @ chol_matrix.T
        
        # Update forward rates using lognormal dynamics
        drift = config.FORWARD_RATE_DRIFT * dt
        diffusion = config.FORWARD_RATE_VOL * np.sqrt(dt) * z_correlated
        
        forward_rates[:, step + 1, :] = forward_rates[:, step, :] * np.exp(
            drift - 0.5 * config.FORWARD_RATE_VOL**2 * dt + diffusion
        )
        
        # Floor forward rates at 0.5% to avoid negative rates
        forward_rates[:, step + 1, :] = np.maximum(forward_rates[:, step + 1, :], 0.005)
    
    return forward_rates


def compute_discount_factors(forward_rates: np.ndarray, config: SimulationConfig) -> np.ndarray:
    """
    Compute discount factors from simulated forward rates.
    
    Discount factor from time 0 to time T:
    DF(0, T) = exp(-sum(forward_rate_i * dt))
    
    Parameters:
    -----------
    forward_rates : np.ndarray of shape (num_sims, num_steps+1, num_tenors)
    config : SimulationConfig
    
    Returns:
    --------
    discount_factors : np.ndarray of shape (num_sims, num_steps+1)
        Discount factors for each path and time step
    """
    num_sims, num_steps_plus_1, _ = forward_rates.shape
    discount_factors = np.ones((num_sims, num_steps_plus_1))
    
    dt = config.DT
    
    for step in range(1, num_steps_plus_1):
        # Use forward rate at current step for discounting
        # (simplified: using diagonal of forward rate matrix)
        short_rate = forward_rates[:, step, min(step, forward_rates.shape[2] - 1)]
        discount_factors[:, step] = discount_factors[:, step - 1] * np.exp(-short_rate * dt)
    
    return discount_factors


# ============================================================================
# REAL ESTATE COLLATERAL MODEL
# ============================================================================

def simulate_collateral_paths(config: SimulationConfig, num_sims: int) -> np.ndarray:
    """
    Simulate real estate collateral values using Geometric Brownian Motion (GBM)
    with a factor structure: systemic + idiosyncratic risk.
    
    dS_t = mu * S_t * dt + sigma * S_t * dW_t
    
    Parameters:
    -----------
    config : SimulationConfig
    num_sims : int - number of Monte Carlo paths
    
    Returns:
    --------
    collateral_values : np.ndarray of shape (num_sims, num_steps+1, num_projects)
        Collateral value for each path, time step, and project
    """
    num_steps = config.NUM_STEPS
    num_projects = config.NUM_PROJECTS
    dt = config.DT
    
    # Initialize collateral values
    collateral_values = np.zeros((num_sims, num_steps + 1, num_projects))
    collateral_values[:, 0, :] = config.INITIAL_COLLATERAL_PER_PROJECT
    
    # Volatility decomposition
    systemic_vol = config.RE_VOL * np.sqrt(config.RE_SYSTEMIC_FACTOR)
    idiosyncratic_vol = config.RE_VOL * np.sqrt(config.RE_IDIOSYNCRATIC_FACTOR)
    
    # Simulate paths
    for step in range(num_steps):
        # Systemic factor (common to all projects in each path)
        z_systemic = np.random.standard_normal(num_sims)
        
        # Idiosyncratic factors (independent for each project)
        z_idiosyncratic = np.random.standard_normal((num_sims, num_projects))
        
        # GBM dynamics
        drift = (config.RE_DRIFT - 0.5 * config.RE_VOL**2) * dt
        diffusion_systemic = systemic_vol * np.sqrt(dt) * z_systemic[:, np.newaxis]
        diffusion_idiosyncratic = idiosyncratic_vol * np.sqrt(dt) * z_idiosyncratic
        
        total_return = drift + diffusion_systemic + diffusion_idiosyncratic
        
        collateral_values[:, step + 1, :] = collateral_values[:, step, :] * np.exp(total_return)
    
    return collateral_values


# ============================================================================
# DEFAULT AND RECOVERY MODEL
# ============================================================================

def simulate_defaults(
    collateral_values: np.ndarray,
    config: SimulationConfig,
    num_sims: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate default events for each loan based on collateral coverage.
    
    Default probability increases when collateral coverage deteriorates.
    
    Parameters:
    -----------
    collateral_values : np.ndarray of shape (num_sims, num_steps+1, num_projects)
    config : SimulationConfig
    num_sims : int
    
    Returns:
    --------
    default_indicator : np.ndarray of shape (num_sims, num_steps+1, num_projects)
        1 if project has defaulted by that time, 0 otherwise (cumulative)
    default_time : np.ndarray of shape (num_sims, num_projects)
        Time step when default occurred (or num_steps+1 if no default)
    """
    num_steps = config.NUM_STEPS
    num_projects = config.NUM_PROJECTS
    
    default_indicator = np.zeros((num_sims, num_steps + 1, num_projects))
    default_time = np.full((num_sims, num_projects), num_steps + 1)
    
    loan_amount = config.LOAN_PER_PROJECT
    
    for sim in range(num_sims):
        for proj in range(num_projects):
            for step in range(1, num_steps + 1):
                # Skip if already defaulted
                if default_indicator[sim, step - 1, proj] == 1:
                    default_indicator[sim, step, proj] = 1
                    continue
                
                # Calculate collateral coverage ratio
                coverage_ratio = collateral_values[sim, step, proj] / loan_amount
                
                # Determine default probability based on coverage
                if coverage_ratio >= config.DEFAULT_THRESHOLD_1:
                    default_prob = config.BASE_DEFAULT_PROB * config.DT
                elif coverage_ratio >= config.DEFAULT_THRESHOLD_2:
                    default_prob = config.BASE_DEFAULT_PROB * config.DEFAULT_PROB_MULT_1 * config.DT
                else:
                    default_prob = config.BASE_DEFAULT_PROB * config.DEFAULT_PROB_MULT_2 * config.DT
                
                # Simulate default event
                if np.random.random() < default_prob:
                    default_indicator[sim, step, proj] = 1
                    default_time[sim, proj] = step
    
    return default_indicator, default_time


def compute_recovery_value(
    collateral_value: float,
    config: SimulationConfig
) -> float:
    """
    Compute recovery value from liquidated collateral.
    
    Recovery = Collateral * Recovery_Rate - Liquidation_Costs
    """
    gross_recovery = collateral_value * config.RECOVERY_RATE
    liquidation_costs = collateral_value * config.LIQUIDATION_COST_RATE
    net_recovery = max(0, gross_recovery - liquidation_costs)
    return net_recovery


# ============================================================================
# CASH FLOW GENERATION
# ============================================================================

def generate_loan_cash_flows(
    collateral_values: np.ndarray,
    default_indicator: np.ndarray,
    default_time: np.ndarray,
    config: SimulationConfig,
    num_sims: int
) -> np.ndarray:
    """
    Generate cash flows from the loan pool for each simulation path.
    
    Cash flows include:
    - Coupon payments (annual)
    - Principal repayment (at maturity or default)
    - Recovery proceeds (on default)
    
    Parameters:
    -----------
    collateral_values : np.ndarray
    default_indicator : np.ndarray
    default_time : np.ndarray
    config : SimulationConfig
    num_sims : int
    
    Returns:
    --------
    pool_cash_flows : np.ndarray of shape (num_sims, num_steps+1)
        Total cash flows from the pool at each time step
    """
    num_steps = config.NUM_STEPS
    num_projects = config.NUM_PROJECTS
    
    pool_cash_flows = np.zeros((num_sims, num_steps + 1))
    
    loan_amount = config.LOAN_PER_PROJECT
    annual_coupon = loan_amount * config.LOAN_COUPON
    
    for sim in range(num_sims):
        for proj in range(num_projects):
            default_step = default_time[sim, proj]
            
            # Generate scheduled cash flows until default or maturity
            for step in range(1, num_steps + 1):
                if step < default_step:
                    # Regular coupon payment (pay at end of each year)
                    if step % config.TIME_STEPS_PER_YEAR == 0:
                        pool_cash_flows[sim, step] += annual_coupon
                    
                    # Principal at maturity (if no default)
                    if step == num_steps and default_step > num_steps:
                        pool_cash_flows[sim, step] += loan_amount
                
                elif step == default_step:
                    # Recovery cash flow at default
                    collateral_at_default = collateral_values[sim, step, proj]
                    recovery = compute_recovery_value(collateral_at_default, config)
                    pool_cash_flows[sim, step] += recovery
                    break
    
    return pool_cash_flows


# ============================================================================
# INVESTOR METRICS
# ============================================================================

def compute_investor_irr(
    pool_cash_flows: np.ndarray,
    config: SimulationConfig,
    num_sims: int
) -> np.ndarray:
    """
    Compute IRR for each investor on each simulation path.
    
    Since all investors have equal pro-rata rights, they all have the same IRR.
    
    Parameters:
    -----------
    pool_cash_flows : np.ndarray of shape (num_sims, num_steps+1)
    config : SimulationConfig
    num_sims : int
    
    Returns:
    --------
    irr_array : np.ndarray of shape (num_sims,)
        IRR for each simulation path
    """
    irr_array = np.zeros(num_sims)
    
    initial_investment = config.INVESTMENT_PER_INVESTOR
    num_investors = config.NUM_INVESTORS
    
    for sim in range(num_sims):
        # Investor's share of cash flows (pro-rata)
        investor_cash_flows = pool_cash_flows[sim, :] / num_investors
        
        # Build complete cash flow timeline (initial investment at t=0)
        cf_timeline = np.zeros(config.NUM_STEPS + 1)
        cf_timeline[0] = -initial_investment
        cf_timeline[1:] = investor_cash_flows[1:]
        
        # Compute IRR using numpy's IRR function
        # (convert to annual basis)
        time_points = np.arange(len(cf_timeline)) * config.DT
        
        # Use Newton-Raphson to solve for IRR
        irr = compute_irr_newton(cf_timeline, time_points)
        irr_array[sim] = irr
    
    return irr_array


def compute_irr_newton(cash_flows: np.ndarray, time_points: np.ndarray, max_iter: int = 100) -> float:
    """
    Compute IRR using Newton-Raphson method.
    
    NPV(r) = sum(CF_t / (1+r)^t) = 0
    """
    # Initial guess
    r = 0.10
    
    for _ in range(max_iter):
        # NPV and derivative
        npv = 0.0
        npv_derivative = 0.0
        
        for i, cf in enumerate(cash_flows):
            t = time_points[i]
            discount_factor = (1 + r) ** t
            npv += cf / discount_factor
            npv_derivative += -t * cf / ((1 + r) ** (t + 1))
        
        # Newton-Raphson update
        if abs(npv_derivative) < 1e-10:
            break
        
        r_new = r - npv / npv_derivative
        
        if abs(r_new - r) < 1e-6:
            return r_new
        
        r = r_new
        
        # Bounds check
        if r < -0.99:
            r = -0.99
        elif r > 5.0:
            r = 5.0
    
    return r


def compute_investor_npv(
    pool_cash_flows: np.ndarray,
    discount_factors: np.ndarray,
    config: SimulationConfig,
    num_sims: int
) -> np.ndarray:
    """
    Compute NPV for each investor using simulated discount factors.
    
    Parameters:
    -----------
    pool_cash_flows : np.ndarray of shape (num_sims, num_steps+1)
    discount_factors : np.ndarray of shape (num_sims, num_steps+1)
    config : SimulationConfig
    num_sims : int
    
    Returns:
    --------
    npv_array : np.ndarray of shape (num_sims,)
        NPV for each simulation path
    """
    npv_array = np.zeros(num_sims)
    
    initial_investment = config.INVESTMENT_PER_INVESTOR
    num_investors = config.NUM_INVESTORS
    
    for sim in range(num_sims):
        # Investor's share of cash flows
        investor_cash_flows = pool_cash_flows[sim, :] / num_investors
        
        # Discount cash flows
        pv_inflows = np.sum(investor_cash_flows[1:] * discount_factors[sim, 1:])
        npv = -initial_investment + pv_inflows
        
        npv_array[sim] = npv
    
    return npv_array


# ============================================================================
# MAIN MONTE CARLO ENGINE
# ============================================================================

def run_monte_carlo(config: SimulationConfig) -> Dict:
    """
    Main Monte Carlo simulation engine.
    
    Returns:
    --------
    results : dict containing:
        - irr_array
        - npv_array
        - pool_cash_flows
        - collateral_values (sample)
        - discount_factors (sample)
        - default_rate
    """
    print("=" * 80)
    print("REAL ESTATE POOL MONTE CARLO SIMULATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Total Corpus: ₹{config.TOTAL_CORPUS:,.0f} ({config.TOTAL_CORPUS/10_000_000:.0f} crore)")
    print(f"  Number of Investors: {config.NUM_INVESTORS}")
    print(f"  Number of Projects: {config.NUM_PROJECTS}")
    print(f"  Loan per Project: ₹{config.LOAN_PER_PROJECT:,.0f}")
    print(f"  Collateral per Project: ₹{config.INITIAL_COLLATERAL_PER_PROJECT:,.0f}")
    print(f"  Initial LTV: {config.LOAN_PER_PROJECT / config.INITIAL_COLLATERAL_PER_PROJECT:.1%}")
    print(f"  Loan Coupon: {config.LOAN_COUPON:.1%}")
    print(f"  Horizon: {config.HORIZON_YEARS} years")
    print(f"  Time Steps: {config.NUM_STEPS}")
    print(f"  Monte Carlo Paths: {config.NUM_SIMULATIONS:,}")
    
    num_sims = config.NUM_SIMULATIONS
    
    # Step 1: Simulate Interest Rates
    print("\n[1/6] Simulating forward rates (BGM model)...")
    forward_rates = simulate_forward_rates(config, num_sims)
    discount_factors = compute_discount_factors(forward_rates, config)
    
    # Step 2: Simulate Collateral Values
    print("[2/6] Simulating real estate collateral values (GBM)...")
    collateral_values = simulate_collateral_paths(config, num_sims)
    
    # Step 3: Simulate Defaults
    print("[3/6] Simulating default events...")
    default_indicator, default_time = simulate_defaults(collateral_values, config, num_sims)
    
    # Calculate default statistics
    total_defaults = np.sum(default_time <= config.NUM_STEPS)
    default_rate = total_defaults / (num_sims * config.NUM_PROJECTS)
    print(f"     Total defaults: {total_defaults:,} out of {num_sims * config.NUM_PROJECTS:,} loans")
    print(f"     Default rate: {default_rate:.2%}")
    
    # Step 4: Generate Cash Flows
    print("[4/6] Generating cash flows...")
    pool_cash_flows = generate_loan_cash_flows(
        collateral_values, default_indicator, default_time, config, num_sims
    )
    
    # Step 5: Compute Investor IRR
    print("[5/6] Computing investor IRR...")
    irr_array = compute_investor_irr(pool_cash_flows, config, num_sims)
    
    # Step 6: Compute Investor NPV
    print("[6/6] Computing investor NPV...")
    npv_array = compute_investor_npv(pool_cash_flows, discount_factors, config, num_sims)
    
    print("\nSimulation complete!\n")
    
    return {
        'irr_array': irr_array,
        'npv_array': npv_array,
        'pool_cash_flows': pool_cash_flows,
        'collateral_values': collateral_values,
        'discount_factors': discount_factors,
        'default_rate': default_rate,
        'forward_rates': forward_rates,
        'default_indicator': default_indicator,
    }


# ============================================================================
# ANALYSIS AND REPORTING
# ============================================================================

def print_summary_statistics(results: Dict, config: SimulationConfig):
    """Print summary statistics of simulation results."""
    irr_array = results['irr_array']
    npv_array = results['npv_array']
    
    print("=" * 80)
    print("INVESTOR IRR STATISTICS")
    print("=" * 80)
    
    print(f"\nMean IRR:                {np.mean(irr_array):.2%}")
    print(f"Median IRR:              {np.median(irr_array):.2%}")
    print(f"Std Dev:                 {np.std(irr_array):.2%}")
    print(f"\nPercentiles:")
    print(f"  5th percentile:        {np.percentile(irr_array, 5):.2%}")
    print(f"  25th percentile (Q1):  {np.percentile(irr_array, 25):.2%}")
    print(f"  50th percentile (Med): {np.percentile(irr_array, 50):.2%}")
    print(f"  75th percentile (Q3):  {np.percentile(irr_array, 75):.2%}")
    print(f"  95th percentile:       {np.percentile(irr_array, 95):.2%}")
    
    print(f"\nProbability Buckets:")
    print(f"  IRR < 10%:             {np.mean(irr_array < 0.10):.2%}")
    print(f"  IRR between 10-14%:    {np.mean((irr_array >= 0.10) & (irr_array < 0.14)):.2%}")
    print(f"  IRR between 14-16%:    {np.mean((irr_array >= 0.14) & (irr_array < 0.16)):.2%}")
    print(f"  IRR > 16%:             {np.mean(irr_array >= 0.16):.2%}")
    
    print("\n" + "=" * 80)
    print("INVESTOR NPV STATISTICS (₹)")
    print("=" * 80)
    
    print(f"\nMean NPV:                ₹{np.mean(npv_array):,.0f}")
    print(f"Median NPV:              ₹{np.median(npv_array):,.0f}")
    print(f"Std Dev:                 ₹{np.std(npv_array):,.0f}")
    print(f"\nPercentiles:")
    print(f"  5th percentile:        ₹{np.percentile(npv_array, 5):,.0f}")
    print(f"  25th percentile:       ₹{np.percentile(npv_array, 25):,.0f}")
    print(f"  50th percentile:       ₹{np.percentile(npv_array, 50):,.0f}")
    print(f"  75th percentile:       ₹{np.percentile(npv_array, 75):,.0f}")
    print(f"  95th percentile:       ₹{np.percentile(npv_array, 95):,.0f}")
    
    print(f"\nProbability of Loss:     {np.mean(npv_array < 0):.2%}")
    
    # Expected annual cash flows
    print("\n" + "=" * 80)
    print("EXPECTED ANNUAL CASH FLOWS TO INVESTORS")
    print("=" * 80)
    
    pool_cf = results['pool_cash_flows']
    num_investors = config.NUM_INVESTORS
    
    print(f"\n{'Year':<6} {'Expected CF (₹)':<20} {'Std Dev (₹)':<20}")
    print("-" * 50)
    for year in range(1, config.HORIZON_YEARS + 1):
        step = year * config.TIME_STEPS_PER_YEAR
        if step <= config.NUM_STEPS:
            investor_cf = pool_cf[:, step] / num_investors
            mean_cf = np.mean(investor_cf)
            std_cf = np.std(investor_cf)
            print(f"{year:<6} {mean_cf:>18,.0f}  {std_cf:>18,.0f}")


def create_plots(results: Dict, config: SimulationConfig):
    """Generate and save visualization plots."""
    import os
    
    if config.SAVE_PLOTS and not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    
    irr_array = results['irr_array']
    npv_array = results['npv_array']
    pool_cf = results['pool_cash_flows']
    collateral = results['collateral_values']
    discount = results['discount_factors']
    
    # Plot 1: IRR Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(irr_array * 100, bins=50, alpha=0.7, edgecolor='black', density=True)
    plt.axvline(np.mean(irr_array) * 100, color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(irr_array):.2%}')
    plt.axvline(np.median(irr_array) * 100, color='green', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(irr_array):.2%}')
    plt.xlabel('IRR (%)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Investor IRR', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if config.SAVE_PLOTS:
        plt.savefig(f'{config.OUTPUT_DIR}/irr_histogram.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {config.OUTPUT_DIR}/irr_histogram.png")
    plt.close()
    
    # Plot 2: IRR CDF
    plt.figure(figsize=(10, 6))
    sorted_irr = np.sort(irr_array)
    cdf = np.arange(1, len(sorted_irr) + 1) / len(sorted_irr)
    plt.plot(sorted_irr * 100, cdf, linewidth=2)
    plt.axvline(14, color='red', linestyle='--', alpha=0.5, label='Target IRR: 14%')
    plt.axvline(16, color='orange', linestyle='--', alpha=0.5, label='Target IRR: 16%')
    plt.xlabel('IRR (%)', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title('Cumulative Distribution of Investor IRR', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if config.SAVE_PLOTS:
        plt.savefig(f'{config.OUTPUT_DIR}/irr_cdf.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {config.OUTPUT_DIR}/irr_cdf.png")
    plt.close()
    
    # Plot 3: Sample Collateral Path
    plt.figure(figsize=(12, 6))
    sample_path = 0
    years = np.arange(config.NUM_STEPS + 1) * config.DT
    
    for proj in range(min(5, config.NUM_PROJECTS)):  # Plot first 5 projects
        plt.plot(years, collateral[sample_path, :, proj] / 10_000_000, 
                label=f'Project {proj+1}', linewidth=2, alpha=0.7)
    
    plt.axhline(config.LOAN_PER_PROJECT / 10_000_000, color='red', 
                linestyle='--', linewidth=2, label='Loan Amount')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Collateral Value (₹ Crore)', fontsize=12)
    plt.title('Sample Collateral Value Evolution (Path 1)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if config.SAVE_PLOTS:
        plt.savefig(f'{config.OUTPUT_DIR}/collateral_paths.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {config.OUTPUT_DIR}/collateral_paths.png")
    plt.close()
    
    # Plot 4: Sample Cash Flow Timeline
    plt.figure(figsize=(12, 6))
    sample_investor_cf = pool_cf[sample_path, 1:] / config.NUM_INVESTORS / 10_000_000
    years_cf = np.arange(1, config.NUM_STEPS + 1) * config.DT
    
    plt.bar(years_cf, sample_investor_cf, alpha=0.7, edgecolor='black')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Cash Flow (₹ Crore)', fontsize=12)
    plt.title('Sample Investor Cash Flow Timeline (Path 1)', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, axis='y')
    
    if config.SAVE_PLOTS:
        plt.savefig(f'{config.OUTPUT_DIR}/cash_flow_timeline.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {config.OUTPUT_DIR}/cash_flow_timeline.png")
    plt.close()
    
    # Plot 5: Discount Factor Evolution
    plt.figure(figsize=(10, 6))
    
    # Plot multiple sample paths
    for path in range(min(20, config.NUM_SIMULATIONS)):
        plt.plot(years, discount[path, :], alpha=0.2, color='blue', linewidth=1)
    
    # Plot mean
    mean_discount = np.mean(discount, axis=0)
    plt.plot(years, mean_discount, color='red', linewidth=3, label='Mean')
    
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Discount Factor', fontsize=12)
    plt.title('Discount Factor Evolution (Sample Paths)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if config.SAVE_PLOTS:
        plt.savefig(f'{config.OUTPUT_DIR}/discount_factors.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {config.OUTPUT_DIR}/discount_factors.png")
    plt.close()
    
    # Plot 6: Expected Cash Flows by Year
    plt.figure(figsize=(12, 6))
    expected_cf = np.mean(pool_cf[:, 1:] / config.NUM_INVESTORS, axis=0) / 10_000_000
    std_cf = np.std(pool_cf[:, 1:] / config.NUM_INVESTORS, axis=0) / 10_000_000
    years_cf = np.arange(1, config.NUM_STEPS + 1) * config.DT
    
    plt.bar(years_cf, expected_cf, alpha=0.7, edgecolor='black', label='Expected CF')
    plt.errorbar(years_cf, expected_cf, yerr=std_cf, fmt='none', 
                ecolor='red', capsize=5, alpha=0.7, label='± 1 Std Dev')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Cash Flow (₹ Crore)', fontsize=12)
    plt.title('Expected Annual Cash Flow to Investor', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    
    if config.SAVE_PLOTS:
        plt.savefig(f'{config.OUTPUT_DIR}/expected_cash_flows.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {config.OUTPUT_DIR}/expected_cash_flows.png")
    plt.close()
    
    print(f"\nAll plots saved to '{config.OUTPUT_DIR}/' directory")


def save_results_to_csv(results: Dict, config: SimulationConfig):
    """Save detailed results to CSV files."""
    import os
    
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    
    # Save IRR and NPV results
    df_summary = pd.DataFrame({
        'path_id': range(config.NUM_SIMULATIONS),
        'irr': results['irr_array'],
        'npv': results['npv_array'],
    })
    
    csv_path = f'{config.OUTPUT_DIR}/simulation_results.csv'
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved detailed results to: {csv_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    # Initialize configuration
    config = SimulationConfig()
    
    # Run Monte Carlo simulation
    results = run_monte_carlo(config)
    
    # Print summary statistics
    print_summary_statistics(results, config)
    
    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80 + "\n")
    create_plots(results, config)
    
    # Save results to CSV
    save_results_to_csv(results, config)
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    print(f"\nDefault Rate: {results['default_rate']:.2%}")
    print(f"Mean Investor IRR: {np.mean(results['irr_array']):.2%}")
    print(f"Median Investor IRR: {np.median(results['irr_array']):.2%}")
    print(f"\nAll outputs saved to '{config.OUTPUT_DIR}/' directory")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
