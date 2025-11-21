# Real Estate Pool Monte Carlo Simulation

A comprehensive Monte Carlo simulation engine for analyzing a real-estate-backed credit product with stochastic interest rates, collateral dynamics, defaults, and investor cash flows.

## Overview

This simulation models:
- **10 crore corpus** from 10 investors (₹1 crore each with pro-rata rights)
- **Loan pool** secured by real estate collateral across 10 projects
- **10-year horizon** with stochastic modeling of:
  - Interest rates (BGM-style forward rate model)
  - Real estate collateral values (Geometric Brownian Motion)
  - Default events and recoveries
  - Investor cash flows and IRR distributions

## Quick Start

### Prerequisites

Install required Python packages:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy pandas matplotlib scipy dash plotly
```

### Run the Simulation

**Option 1: Command Line (Static Analysis)**

```bash
python simulate_real_estate_pool.py
```

This will:
1. Run 5,000 Monte Carlo paths
2. Print summary statistics to console
3. Generate 6 visualization plots
4. Save detailed results to CSV

All outputs are saved to `simulation_output/` directory.

**Option 2: Interactive Dashboard (Recommended)**

```bash
python dashboard.py
```

Then open your browser to: **http://127.0.0.1:8050/**

The dashboard allows you to:
- 🎛️ **Adjust parameters dynamically** using sliders
- 📊 **See results update in real-time**
- 📈 **Explore 6 interactive visualizations**
- 🔍 **Compare scenarios instantly**
- 💾 **No need to re-run scripts manually**

## Configuration

All parameters are centralized in the `SimulationConfig` class at the top of the script. Key parameters you can modify:

### Pool Structure
```python
TOTAL_CORPUS = 100_000_000          # 10 crore total
NUM_INVESTORS = 10
NUM_PROJECTS = 10
INITIAL_COLLATERAL_PER_PROJECT = 20_000_000  # 2 crore per project
```

### Time Grid
```python
HORIZON_YEARS = 10
TIME_STEPS_PER_YEAR = 1  # Change to 2 for semi-annual, 4 for quarterly
```

### Interest Rate Model (BGM-style)
```python
INITIAL_FORWARD_RATE = 0.08  # 8% flat forward curve
FORWARD_RATE_VOL = 0.15      # 15% volatility
FORWARD_RATE_CORR = 0.80     # Correlation between forward rates
```

### Loan Terms
```python
LOAN_COUPON = 0.12           # 12% annual coupon
LOAN_MATURITY_YEARS = 10
AMORTIZATION = False         # True for amortization, False for bullet
```

### Real Estate Dynamics
```python
RE_DRIFT = 0.05              # 5% expected appreciation
RE_VOL = 0.15                # 15% volatility
RE_SYSTEMIC_FACTOR = 0.60    # 60% systemic risk
RE_IDIOSYNCRATIC_FACTOR = 0.40  # 40% idiosyncratic risk
```

### Default Model
```python
BASE_DEFAULT_PROB = 0.03     # 3% annual base default probability
DEFAULT_THRESHOLD_1 = 1.2    # Collateral/Loan threshold 1
DEFAULT_THRESHOLD_2 = 1.0    # Collateral/Loan threshold 2
DEFAULT_PROB_MULT_1 = 2.0    # Multiplier below threshold 1
DEFAULT_PROB_MULT_2 = 4.0    # Multiplier below threshold 2
```

### Recovery
```python
RECOVERY_RATE = 0.70         # 70% of collateral recovered
LIQUIDATION_COST_RATE = 0.05 # 5% liquidation costs
```

### Monte Carlo
```python
NUM_SIMULATIONS = 5000       # Number of paths (reduce to 500 for quick tests)
RANDOM_SEED = 42             # For reproducibility
```

## Common Modifications

### Quick Test Run (Faster Execution)

Change in `SimulationConfig`:
```python
NUM_SIMULATIONS = 500        # Reduced from 5000
NUM_PROJECTS = 5             # Reduced from 10
```

### Increase to 50 Projects

```python
NUM_PROJECTS = 50
LOAN_PER_PROJECT = TOTAL_CORPUS / NUM_PROJECTS
# Adjust collateral accordingly if needed
```

### Change to Quarterly Time Steps

```python
TIME_STEPS_PER_YEAR = 4      # Quarterly instead of annual
```

### More Conservative Scenario (Higher Default Risk)

```python
BASE_DEFAULT_PROB = 0.05     # Increase to 5%
RE_DRIFT = 0.03              # Lower appreciation to 3%
RE_VOL = 0.20                # Increase volatility to 20%
```

### More Aggressive Scenario (Lower Default Risk)

```python
BASE_DEFAULT_PROB = 0.01     # Decrease to 1%
INITIAL_COLLATERAL_PER_PROJECT = 30_000_000  # 3 crore (300% coverage)
RE_DRIFT = 0.07              # Higher appreciation to 7%
```

## Output Files

After running the simulation, the following files are created in `simulation_output/`:

### Plots (PNG format)
1. **irr_histogram.png** - Distribution of investor IRR
2. **irr_cdf.png** - Cumulative distribution of IRR
3. **collateral_paths.png** - Sample evolution of collateral values
4. **cash_flow_timeline.png** - Sample investor cash flow timeline
5. **discount_factors.png** - Evolution of discount factors
6. **expected_cash_flows.png** - Expected annual cash flows with error bars

### Data Files
- **simulation_results.csv** - Detailed results for each simulation path (path_id, IRR, NPV)

## Understanding the Output

### Console Statistics

The simulation prints:

1. **IRR Statistics**
   - Mean, median, standard deviation
   - Percentiles (5th, 25th, 50th, 75th, 95th)
   - Probability buckets:
     - Below 10%
     - Between 10-14%
     - Between 14-16%
     - Above 16%

2. **NPV Statistics**
   - Distribution of net present value
   - Probability of loss

3. **Expected Annual Cash Flows**
   - Year-by-year expected cash flows to each investor
   - Standard deviation showing uncertainty

### Interpretation

- **Target IRR Range**: 14-16%
- **Mean IRR**: Should be in the 12-15% range under base scenario
- **5th Percentile IRR**: Shows downside risk (stressed scenarios)
- **95th Percentile IRR**: Shows upside potential
- **Default Rate**: Expected to be 5-15% over 10 years depending on parameters

## Model Components

### 1. BGM-Style Interest Rate Model

Simulates forward rates using:
```
df_t = mu * f_t * dt + sigma * f_t * dW_t
```

- Lognormal dynamics ensure non-negative rates
- Single-factor model (all rates driven by one Brownian motion)
- Correlation structure via exponential decay
- Generates discount factors for NPV calculations

### 2. Real Estate Collateral (GBM)

```
dS_t = mu * S_t * dt + sigma * S_t * dW_t
```

- Systemic factor: captures market-wide real estate movements
- Idiosyncratic factor: project-specific risk
- Initial LTV: 50% (₹1 crore loan on ₹2 crore collateral)

### 3. Default Model

Dynamic default probability based on collateral coverage:
- **High coverage (>120%)**: Base default rate
- **Moderate coverage (100-120%)**: 2× base rate
- **Low coverage (<100%)**: 4× base rate

On default:
- Collateral liquidated at current market value
- Recovery = Collateral × Recovery_Rate - Liquidation_Costs
- Losses allocated to investors pro-rata

### 4. Cash Flow Waterfall

1. Collect loan coupons (12% annually)
2. Collect principal repayments
3. Process recovery proceeds on defaults
4. Distribute to investors pro-rata (equal shares)

## Adapting to Jupyter Notebook

To use this in a Jupyter notebook:

1. Copy the entire script into a notebook
2. Break it into cells:
   - Cell 1: Imports and configuration
   - Cell 2: Interest rate functions
   - Cell 3: Collateral functions
   - Cell 4: Default functions
   - Cell 5: Cash flow functions
   - Cell 6: Investor metrics
   - Cell 7: Monte Carlo engine
   - Cell 8: Analysis and plotting
   - Cell 9: Run main()

3. For interactive exploration:
```python
# Run with custom parameters
config = SimulationConfig()
config.NUM_SIMULATIONS = 1000
config.RE_VOL = 0.20
results = run_monte_carlo(config)
```

## Limitations

This is an illustrative model with simplifications:

- **Interest rates**: Single-factor BGM (no term structure twists or mean reversion)
- **Calibration**: Rough parameters, not fitted to market data
- **Prepayments**: Not modeled
- **Fees**: Simplified (no management fees, performance fees, etc.)
- **Taxes**: Not included
- **Legal/operational risks**: Not modeled
- **Liquidity risk**: Not captured
- **No tranching**: All investors have identical pari passu rights

## Advanced Usage

### Scenario Analysis

Create multiple configurations:

```python
# Base case
config_base = SimulationConfig()
results_base = run_monte_carlo(config_base)

# Stressed case
config_stress = SimulationConfig()
config_stress.BASE_DEFAULT_PROB = 0.08
config_stress.RE_DRIFT = 0.02
results_stress = run_monte_carlo(config_stress)

# Compare distributions
print(f"Base Mean IRR: {np.mean(results_base['irr_array']):.2%}")
print(f"Stress Mean IRR: {np.mean(results_stress['irr_array']):.2%}")
```

### Custom Analysis

Access detailed results:

```python
results = run_monte_carlo(config)

# Analyze worst 5% of scenarios
worst_5pct_idx = results['irr_array'] < np.percentile(results['irr_array'], 5)
worst_defaults = results['default_indicator'][worst_5pct_idx].mean()

print(f"Average defaults in worst 5% scenarios: {worst_defaults:.1%}")
```

## Questions or Issues?

This is a pedagogical model designed to demonstrate key concepts in structured credit modeling. For production use, consider:

- Multi-factor interest rate models (e.g., HJM with multiple factors)
- Calibration to market data (interest rate swaps, cap/floor volatilities)
- More sophisticated default modeling (e.g., Merton structural models)
- Prepayment models
- Regulatory capital calculations
- Stress testing frameworks

## Interactive Dashboard Features

The dashboard (`dashboard.py`) provides a modern, interactive interface:

### Available Controls

1. **Number of Simulations** (100 - 5,000 paths)
   - Adjust simulation size vs. speed tradeoff
   - Recommended: 1,000 for exploration, 5,000 for final analysis

2. **Loan Coupon** (8% - 18%)
   - Change the interest rate charged to borrowers
   - Higher coupon → higher investor returns (if no defaults)

3. **RE Expected Return** (-2% to 10%)
   - Control real estate price drift
   - Negative values model declining markets

4. **RE Volatility** (5% - 35%)
   - Adjust real estate price uncertainty
   - Higher volatility → wider IRR distribution

5. **Base Default Probability** (1% - 10%)
   - Set baseline annual default rate
   - Dynamic adjustment based on collateral coverage

6. **Recovery Rate** (40% - 90%)
   - Control how much is recovered from defaulted loans
   - Lower recovery → higher losses

7. **Initial Collateral** (₹1.2Cr - ₹3.0Cr per project)
   - Adjust initial LTV ratio
   - Higher collateral → lower default risk

8. **Interest Rate Volatility** (5% - 30%)
   - Control forward rate uncertainty
   - Affects NPV calculations

### Real-Time Visualizations

1. **Summary Cards**: Instant display of mean IRR, median IRR, 5th percentile, default rate
2. **IRR Histogram**: Distribution shape with target lines at 14% and 16%
3. **IRR CDF**: Cumulative probability for risk assessment
4. **Collateral Paths**: Sample evolution of property values
5. **Expected Cash Flows**: Annual investor cash flows with error bars
6. **IRR Box Plot**: Statistical distribution with quartiles
7. **Probability Buckets**: Visual breakdown of IRR ranges

### Usage Tips

- Start with **1,000 simulations** for fast iteration
- Increase to **5,000** when you've found interesting scenarios
- Use the **collateral slider** to test LTV sensitivity
- Adjust **RE volatility** to model market stress
- Compare **mean vs. median IRR** to detect skewness

## License

This code is provided for educational and research purposes.
