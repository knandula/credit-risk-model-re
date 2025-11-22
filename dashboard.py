#!/usr/bin/env python3
"""
Interactive Dashboard for Real Estate Pool Monte Carlo Simulation
==================================================================

A Plotly Dash dashboard that allows real-time parameter adjustment
and visualization of simulation results.

Run with: python dashboard.py
Then open browser to: http://127.0.0.1:8050/

Author: Senior Quant Developer
Date: November 2025
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd
from simulate_real_estate_pool import (
    SimulationConfig,
    simulate_forward_rates,
    compute_discount_factors,
    simulate_collateral_paths,
    simulate_defaults,
    generate_loan_cash_flows,
    compute_investor_irr,
    compute_investor_npv,
)

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Real Estate Pool Risk Dashboard"

# Global variable to store current results
current_results = None

# Add responsive meta tag
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @media (max-width: 768px) {
                /* Mobile styles */
                .left-panel {
                    width: 100% !important;
                    position: relative !important;
                    height: auto !important;
                    float: none !important;
                }
                .right-panel {
                    width: 100% !important;
                    margin-left: 0 !important;
                    float: none !important;
                }
                .header-fixed {
                    position: relative !important;
                    margin-left: 0 !important;
                    left: 0 !important;
                    width: 100% !important;
                }
                .stat-card {
                    width: 48% !important;
                    margin-right: 2% !important;
                    margin-bottom: 10px !important;
                }
                .chart-container {
                    width: 100% !important;
                    margin-right: 0 !important;
                    margin-bottom: 15px !important;
                }
            }
            @media (max-width: 480px) {
                /* Extra small mobile */
                .stat-card {
                    width: 100% !important;
                    margin-right: 0 !important;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ============================================================================
# DASHBOARD LAYOUT
# ============================================================================

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Real Estate - Credit Risk Modelling", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '5px', 'marginTop': '10px', 'fontSize': '22px'}),        
    ], className='header-fixed', style={'padding': '10px', 'backgroundColor': '#ecf0f1',  'position': 'fixed', 'top': '0', 'right': '0', 'left': '20%', 'zIndex': '100', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    
    # Main container
    html.Div([
        # Left Panel - Controls
        html.Div([
            html.H3("Simulation Parameters", style={'color': '#34495e', 'marginTop': '0'}),
            html.Hr(style={'marginBottom': '15px'}),
            
            # Monte Carlo Settings
            html.Div([
                html.Label("Number of Simulations:", style={'fontWeight': 'bold', 'fontSize': '13px'}),
                html.P("Controls simulation accuracy. More paths = more accurate results but slower. Use 1,000 for quick tests, 5,000 for final analysis.", 
                       style={'fontSize': '11px', 'color': '#7f8c8d', 'marginTop': '3px', 'marginBottom': '8px', 'lineHeight': '1.3'}),
                dcc.Slider(
                    id='num-sims-slider',
                    min=100,
                    max=5000,
                    step=100,
                    value=1000,
                    marks={100: '100', 1000: '1K', 2500: '2.5K', 5000: '5K'},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'marginBottom': '20px'}),
            
            # Loan Coupon
            html.Div([
                html.Label("Loan Coupon (%):", style={'fontWeight': 'bold', 'fontSize': '13px'}),
                html.P("Annual interest rate charged on loans. Higher coupon increases investor returns if no defaults occur. Directly impacts cash flow generation.", 
                       style={'fontSize': '11px', 'color': '#7f8c8d', 'marginTop': '3px', 'marginBottom': '8px', 'lineHeight': '1.3'}),
                dcc.Slider(
                    id='loan-coupon-slider',
                    min=8,
                    max=18,
                    step=0.5,
                    value=12,
                    marks={i: f'{i}%' for i in range(8, 19, 2)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'marginBottom': '20px'}),
            
            # Real Estate Drift
            html.Div([
                html.Label("RE Expected Return (%):", style={'fontWeight': 'bold', 'fontSize': '13px'}),
                html.P("Expected annual appreciation of real estate collateral. Positive values model growth markets, negative for declining markets. Affects default probability and recovery.", 
                       style={'fontSize': '11px', 'color': '#7f8c8d', 'marginTop': '3px', 'marginBottom': '8px', 'lineHeight': '1.3'}),
                dcc.Slider(
                    id='re-drift-slider',
                    min=-2,
                    max=10,
                    step=0.5,
                    value=5,
                    marks={i: f'{i}%' for i in range(-2, 11, 2)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'marginBottom': '20px'}),
            
            # Real Estate Volatility
            html.Div([
                html.Label("RE Volatility (%):", style={'fontWeight': 'bold', 'fontSize': '13px'}),
                html.P("Uncertainty in property value changes. Higher volatility = wider range of outcomes. Increases both upside potential and downside risk of collateral values.", 
                       style={'fontSize': '11px', 'color': '#7f8c8d', 'marginTop': '3px', 'marginBottom': '8px', 'lineHeight': '1.3'}),
                dcc.Slider(
                    id='re-vol-slider',
                    min=5,
                    max=35,
                    step=1,
                    value=15,
                    marks={i: f'{i}%' for i in range(5, 36, 5)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'marginBottom': '20px'}),
            
            # Base Default Probability
            html.Div([
                html.Label("Base Default Probability (%):", style={'fontWeight': 'bold', 'fontSize': '13px'}),
                html.P("Annual probability of loan default when collateral is adequate. Increases automatically when collateral falls below loan value. Key driver of credit risk.", 
                       style={'fontSize': '11px', 'color': '#7f8c8d', 'marginTop': '3px', 'marginBottom': '8px', 'lineHeight': '1.3'}),
                dcc.Slider(
                    id='default-prob-slider',
                    min=1,
                    max=10,
                    step=0.5,
                    value=3,
                    marks={i: f'{i}%' for i in range(1, 11, 2)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'marginBottom': '20px'}),
            
            # Recovery Rate
            html.Div([
                html.Label("Recovery Rate (%):", style={'fontWeight': 'bold', 'fontSize': '13px'}),
                html.P("Percentage of collateral value recovered after liquidation on default. Accounts for distressed sale haircuts. Higher recovery = lower losses when defaults occur.", 
                       style={'fontSize': '11px', 'color': '#7f8c8d', 'marginTop': '3px', 'marginBottom': '8px', 'lineHeight': '1.3'}),
                dcc.Slider(
                    id='recovery-rate-slider',
                    min=40,
                    max=90,
                    step=5,
                    value=70,
                    marks={i: f'{i}%' for i in range(40, 91, 10)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'marginBottom': '20px'}),
            
            # Initial Collateral
            html.Div([
                html.Label("Initial Collateral (Crore):", style={'fontWeight': 'bold', 'fontSize': '13px'}),
                html.P("Starting value of real estate backing each ₹1Cr loan. Higher collateral = lower LTV ratio = lower default risk. Provides safety cushion against property value declines.", 
                       style={'fontSize': '11px', 'color': '#7f8c8d', 'marginTop': '3px', 'marginBottom': '8px', 'lineHeight': '1.3'}),
                dcc.Slider(
                    id='collateral-slider',
                    min=1.2,
                    max=3.0,
                    step=0.1,
                    value=2.0,
                    marks={i: f'₹{i}Cr' for i in [1.2, 1.5, 2.0, 2.5, 3.0]},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'marginBottom': '20px'}),
            
            # Interest Rate Volatility
            html.Div([
                html.Label("Interest Rate Volatility (%):", style={'fontWeight': 'bold', 'fontSize': '13px'}),
                html.P("Uncertainty in future interest rates. Affects discount factors used for NPV calculation. Higher volatility creates wider range of present value outcomes.", 
                       style={'fontSize': '11px', 'color': '#7f8c8d', 'marginTop': '3px', 'marginBottom': '8px', 'lineHeight': '1.3'}),
                dcc.Slider(
                    id='ir-vol-slider',
                    min=5,
                    max=30,
                    step=1,
                    value=15,
                    marks={i: f'{i}%' for i in range(5, 31, 5)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'marginBottom': '20px'}),
            
            # Run button
            html.Button('Run Simulation', id='run-button', n_clicks=0,
                       style={
                           'width': '100%',
                           'height': '50px',
                           'fontSize': '18px',
                           'fontWeight': 'bold',
                           'backgroundColor': '#27ae60',
                           'color': 'white',
                           'border': 'none',
                           'borderRadius': '5px',
                           'cursor': 'pointer',
                           'marginTop': '20px'
                       }),
            
            # Loading indicator
            dcc.Loading(
                id="loading",
                type="circle",
                children=html.Div(id="loading-output")
            ),
            
        ], className='left-panel', style={
            'width': '20%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'padding': '15px',
            'backgroundColor': '#f8f9fa',
            'height': '100vh',
            'overflowY': 'auto',
            'boxSizing': 'border-box',
            'position': 'fixed',
            'left': '0',
            'top': '0'
        }),
        
        # Right Panel - Visualizations
        html.Div(className='right-panel', children=[
            # Spacer for fixed header
            html.Div(style={'height': '70px'}),
            
            # Summary Statistics Cards
            html.Div([
                html.Div([
                    html.H4("Mean IRR", style={'color': '#7f8c8d', 'marginBottom': '5px', 'fontSize': '13px'}),
                    html.H2(id='mean-irr-card', children="--", 
                           style={'color': '#2980b9', 'marginTop': '0', 'marginBottom': '5px', 'fontSize': '24px'}),
                    html.P("Average expected return across all scenarios", 
                           style={'fontSize': '10px', 'color': '#95a5a6', 'margin': '0', 'lineHeight': '1.3'}),
                ], className='stat-card', style={
                    'flex': '1',
                    'minWidth': '150px',
                    'padding': '15px',
                    'backgroundColor': 'white',
                    'borderRadius': '8px',
                    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
                    'textAlign': 'center',
                    'marginRight': '15px'
                }),
                
                html.Div([
                    html.H4("Median IRR", style={'color': '#7f8c8d', 'marginBottom': '5px', 'fontSize': '13px'}),
                    html.H2(id='median-irr-card', children="--",
                           style={'color': '#27ae60', 'marginTop': '0', 'marginBottom': '5px', 'fontSize': '24px'}),
                    html.P("Most likely return (50% chance above/below)", 
                           style={'fontSize': '10px', 'color': '#95a5a6', 'margin': '0', 'lineHeight': '1.3'}),
                ], className='stat-card', style={
                    'flex': '1',
                    'minWidth': '150px',
                    'padding': '15px',
                    'backgroundColor': 'white',
                    'borderRadius': '8px',
                    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
                    'textAlign': 'center',
                    'marginRight': '15px'
                }),
                
                html.Div([
                    html.H4("5th Percentile", style={'color': '#7f8c8d', 'marginBottom': '5px', 'fontSize': '13px'}),
                    html.H2(id='percentile5-card', children="--",
                           style={'color': '#c0392b', 'marginTop': '0', 'marginBottom': '5px', 'fontSize': '24px'}),
                    html.P("Worst-case return (95% confidence level)", 
                           style={'fontSize': '10px', 'color': '#95a5a6', 'margin': '0', 'lineHeight': '1.3'}),
                ], className='stat-card', style={
                    'flex': '1',
                    'minWidth': '150px',
                    'padding': '15px',
                    'backgroundColor': 'white',
                    'borderRadius': '8px',
                    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
                    'textAlign': 'center',
                    'marginRight': '15px'
                }),
                
                html.Div([
                    html.H4("Default Rate", style={'color': '#7f8c8d', 'marginBottom': '5px', 'fontSize': '13px'}),
                    html.H2(id='default-rate-card', children="--",
                           style={'color': '#e67e22', 'marginTop': '0', 'marginBottom': '5px', 'fontSize': '24px'}),
                    html.P("% of loans that fail to repay fully", 
                           style={'fontSize': '10px', 'color': '#95a5a6', 'margin': '0', 'lineHeight': '1.3'}),
                ], className='stat-card', style={
                    'flex': '1',
                    'minWidth': '150px',
                    'padding': '15px',
                    'backgroundColor': 'white',
                    'borderRadius': '8px',
                    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
                    'textAlign': 'center'
                }),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '15px', 'gap': '0'}),
            
            # Charts row 1
            html.Div([
                html.Div([
                    dcc.Graph(id='irr-histogram', style={'height': '380px'}),
                    html.P("This histogram shows the distribution of investor returns across all simulated scenarios. A wider spread indicates higher uncertainty in returns, while clustering around a specific value suggests more predictable outcomes. Use this to assess the probability of achieving different return levels.",
                          style={'fontSize': '11px', 'color': '#7f8c8d', 'marginTop': '10px', 'lineHeight': '1.5'})
                ], className='chart-container', style={
                    'flex': '1',
                    'minWidth': '400px',
                    'marginRight': '15px',
                    'backgroundColor': 'white',
                    'borderRadius': '8px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'padding': '10px'
                }),
                
                html.Div([
                    dcc.Graph(id='irr-cdf', style={'height': '380px'}),
                    html.P("The cumulative distribution function (CDF) shows the probability of achieving returns below any given IRR level. The steeper the curve, the more concentrated the returns. Use this to answer questions like 'What's the probability of earning less than 8%?' or 'What return can I expect with 90% confidence?'",
                          style={'fontSize': '11px', 'color': '#7f8c8d', 'marginTop': '10px', 'lineHeight': '1.5'})
                ], className='chart-container', style={
                    'flex': '1',
                    'minWidth': '400px',
                    'backgroundColor': 'white',
                    'borderRadius': '8px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'padding': '10px'
                }),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '15px', 'gap': '0'}),
            
            # Charts row 2
            html.Div([
                html.Div([
                    dcc.Graph(id='collateral-paths', style={'height': '380px'}),
                    html.P("Sample trajectories of real estate collateral values over the 10-year period. These paths incorporate both systematic market risk (affecting all properties) and idiosyncratic risk (property-specific factors). Diverging paths indicate high volatility, while parallel paths suggest more stable market conditions. This helps visualize the range of possible collateral outcomes.",
                          style={'fontSize': '11px', 'color': '#7f8c8d', 'marginTop': '10px', 'lineHeight': '1.5'})
                ], className='chart-container', style={
                    'flex': '1',
                    'minWidth': '400px',
                    'marginRight': '15px',
                    'backgroundColor': 'white',
                    'borderRadius': '8px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'padding': '10px'
                }),
                
                html.Div([
                    dcc.Graph(id='cash-flows', style={'height': '380px'}),
                    html.P("Average expected cash flows to investors over time across all scenarios. This shows the timing and magnitude of interest payments and principal repayments. Early spikes may indicate prepayments, while declining flows could signal defaults. Use this to understand liquidity patterns and when to expect returns on your investment.",
                          style={'fontSize': '11px', 'color': '#7f8c8d', 'marginTop': '10px', 'lineHeight': '1.5'})
                ], className='chart-container', style={
                    'flex': '1',
                    'minWidth': '400px',
                    'backgroundColor': 'white',
                    'borderRadius': '8px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'padding': '10px'
                }),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '15px', 'gap': '0'}),
            
            # Charts row 3
            html.Div([
                html.Div([
                    dcc.Graph(id='irr-boxplot', style={'height': '380px'}),
                    html.P("Box plot summary of IRR distribution showing median (center line), interquartile range (box edges representing 25th-75th percentiles), and outliers. A taller box indicates more variability in returns. Whiskers extend to show the full range excluding extreme outliers. This provides a compact statistical summary of return dispersion and helps identify the most likely return range.",
                          style={'fontSize': '11px', 'color': '#7f8c8d', 'marginTop': '10px', 'lineHeight': '1.5'})
                ], className='chart-container', style={
                    'flex': '1',
                    'minWidth': '400px',
                    'marginRight': '15px',
                    'backgroundColor': 'white',
                    'borderRadius': '8px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'padding': '10px'
                }),
                
                html.Div([
                    dcc.Graph(id='scenario-comparison', style={'height': '380px'}),
                    html.P("Probability distribution of returns grouped into performance buckets. This answers the practical question: 'What are my chances of different outcome levels?' Higher bars in favorable categories (e.g., >10% returns) indicate better investment prospects. Compare bucket heights to understand the likelihood of achieving your target returns versus experiencing losses.",
                          style={'fontSize': '11px', 'color': '#7f8c8d', 'marginTop': '10px', 'lineHeight': '1.5'})
                ], className='chart-container', style={
                    'flex': '1',
                    'minWidth': '400px',
                    'backgroundColor': 'white',
                    'borderRadius': '8px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'padding': '10px'
                }),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '15px', 'gap': '0'}),
            
            # Disclaimer Footer
            html.Div([
                html.H3("⚠️ Important Disclaimer & Model Limitations", 
                       style={'color': "#212020", 'marginBottom': '15px', 'fontSize': '18px', 'fontWeight': 'bold'}),
                
                html.Div([
                    html.H4("🔴 Key Limitations & Red Flags:", 
                           style={'color': '#e74c3c', 'marginTop': '0', 'marginBottom': '10px', 'fontSize': '15px'}),
                    html.Ul([
                        html.Li([html.Strong("Model Simplifications: "), "Assumes independent defaults, ignores contagion effects, and uses simplified correlation structures that may not capture real-world systemic risks. ", html.Span("(In simple terms: If one borrower defaults, it might trigger others - this model doesn't account for that domino effect.)", style={'fontStyle': 'italic', 'color': '#5a6c7d'})]),
                        html.Li([html.Strong("Interest Rate Model: "), "Single-factor BGM model cannot capture complex yield curve dynamics, regime changes, or central bank policy impacts. ", html.Span("(In simple terms: Real interest rate movements are more complex than what this model predicts.)", style={'fontStyle': 'italic', 'color': '#5a6c7d'})]),
                        html.Li([html.Strong("Collateral Assumptions: "), "Geometric Brownian Motion (GBM) for real estate may not reflect actual property market behavior, especially during crises with fat-tailed distributions and momentum effects. ", html.Span("(In simple terms: Property prices can crash suddenly in a crisis - this model may underestimate such extreme drops.)", style={'fontStyle': 'italic', 'color': '#5a6c7d'})]),
                        html.Li([html.Strong("Liquidity Risk Ignored: "), "Model assumes frictionless liquidation of collateral at market prices; real distressed sales involve significant haircuts and timing delays. ", html.Span("(In simple terms: Selling property in a crisis takes time and you'll get less than market value - not factored here.)", style={'fontStyle': 'italic', 'color': '#5a6c7d'})]),
                        html.Li([html.Strong("No Macroeconomic Factors: "), "Excludes unemployment, GDP growth, inflation, regulatory changes, and other macro variables that significantly impact credit risk. ", html.Span("(In simple terms: Economic recessions increase defaults - this isn't modeled directly.)", style={'fontStyle': 'italic', 'color': '#5a6c7d'})]),
                        html.Li([html.Strong("Static Recovery Rate: "), "Assumes constant 70% recovery; in reality, this varies dramatically with market conditions and can collapse during systemic stress. ", html.Span("(In simple terms: You might recover less than 70% in a bad market or more in a good one - this model uses a fixed rate.)", style={'fontStyle': 'italic', 'color': '#5a6c7d'})]),
                        html.Li([html.Strong("Calibration Risk: "), "Parameters are illustrative, not calibrated to actual market data. Real-world calibration requires extensive historical analysis. ", html.Span("(In simple terms: The numbers used here are examples, not based on real market data.)", style={'fontStyle': 'italic', 'color': '#5a6c7d'})]),
                        html.Li([html.Strong("No Legal/Operational Risk: "), "Ignores foreclosure timelines, legal costs, servicing complexities, fraud risk, and documentation issues. ", html.Span("(In simple terms: Legal battles and paperwork delays can reduce your actual returns - not included here.)", style={'fontStyle': 'italic', 'color': '#5a6c7d'})]),
                    ], style={'fontSize': '12px', 'lineHeight': '1.8', 'color': '#2c3e50', 'marginLeft': '20px'}),
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.H4("✅ Model Strengths:", 
                           style={'color': '#27ae60', 'marginTop': '0', 'marginBottom': '10px', 'fontSize': '15px'}),
                    html.Ul([
                        html.Li([html.Strong("Comprehensive Framework: "), "Integrates interest rate risk, collateral dynamics, and credit risk in a unified Monte Carlo framework. ", html.Span("(In simple terms: Considers multiple risk factors together, giving you a more complete picture.)", style={'fontStyle': 'italic', 'color': '#5a6c7d'})]),
                        html.Li([html.Strong("Transparent Methodology: "), "All assumptions and calculations are explicit and auditable in the source code. ", html.Span("(In simple terms: You can see exactly how the numbers are calculated - nothing hidden.)", style={'fontStyle': 'italic', 'color': '#5a6c7d'})]),
                        html.Li([html.Strong("Scenario Analysis: "), "Allows rapid testing of different parameter assumptions to understand sensitivity to key drivers. ", html.Span("(In simple terms: Change sliders to see how different conditions affect your returns instantly.)", style={'fontStyle': 'italic', 'color': '#5a6c7d'})]),
                        html.Li([html.Strong("Educational Value: "), "Demonstrates proper structure for pricing credit products with stochastic modeling techniques. ", html.Span("(In simple terms: Great learning tool to understand how risk modeling works in finance.)", style={'fontStyle': 'italic', 'color': '#5a6c7d'})]),
                        html.Li([html.Strong("Customizable: "), "Code can be extended to incorporate additional risk factors, more sophisticated models, or real market data. ", html.Span("(In simple terms: You can enhance this model with more features as needed.)", style={'fontStyle': 'italic', 'color': '#5a6c7d'})]),
                    ], style={'fontSize': '12px', 'lineHeight': '1.8', 'color': '#2c3e50', 'marginLeft': '20px'}),
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.H4("⚠️ What to Watch Out For:", 
                           style={'color': '#f39c12', 'marginTop': '0', 'marginBottom': '10px', 'fontSize': '15px'}),
                    html.Ul([
                        html.Li([html.Strong("Tail Risk Underestimation: "), "Lognormal models typically underestimate extreme events (2008-style crises). Real losses can far exceed 5th percentile estimates. ", html.Span("(In simple terms: Worst-case scenarios might be worse than shown - think housing crisis 2008.)", style={'fontStyle': 'italic', 'color': '#5a6c7d'})]),
                        html.Li([html.Strong("Parameter Sensitivity: "), "Small changes in volatility, correlation, or default assumptions can drastically alter results. Always run sensitivity analysis. ", html.Span("(In simple terms: Tiny changes in inputs can lead to big changes in results - test different scenarios.)", style={'fontStyle': 'italic', 'color': '#5a6c7d'})]),
                        html.Li([html.Strong("Concentration Risk: "), "Real portfolios may have geographic, borrower, or property-type concentrations not captured here. ", html.Span("(In simple terms: If all properties are in one city and that market crashes, you're in trouble.)", style={'fontStyle': 'italic', 'color': '#5a6c7d'})]),
                        html.Li([html.Strong("Behavioral Assumptions: "), "Model assumes rational borrower behavior; strategic defaults and prepayment clustering during refinancing waves are ignored. ", html.Span("(In simple terms: Borrowers might walk away even when they can pay, or all refinance at once - not modeled.)", style={'fontStyle': 'italic', 'color': '#5a6c7d'})]),
                        html.Li([html.Strong("Market Regime Changes: "), "Model uses constant parameters; real markets undergo structural breaks (regulatory changes, financial crises, technology disruption). ", html.Span("(In simple terms: Major events can change the rules of the game completely - this model assumes stable conditions.)", style={'fontStyle': 'italic', 'color': '#5a6c7d'})]),
                        html.Li([html.Strong("Survivorship Bias: "), "Historical real estate data often excludes failed projects, leading to overly optimistic parameter estimates. ", html.Span("(In simple terms: Past data might look better than reality because failures aren't always recorded.)", style={'fontStyle': 'italic', 'color': '#5a6c7d'})]),
                    ], style={'fontSize': '12px', 'lineHeight': '1.8', 'color': '#2c3e50', 'marginLeft': '20px'}),
                ], style={'marginBottom': '20px'}),
                
                html.Div([                                                  
                    html.P([
                        html.Strong("THIS IS NOT INVESTMENT ADVICE. NO RESPONSIBILITY OR LIABILITY: "),
                        "The authors and distributors of this model assume no responsibility or liability for any losses, damages, or adverse outcomes ",
                        "resulting from the use or misuse of this tool. Users bear full responsibility for validating assumptions, verifying results, ",
                        "and seeking appropriate professional advice before making any financial decisions. Past performance and simulated results ",
                        "do not guarantee future outcomes."
                    ], style={'fontSize': '12px', 'lineHeight': '1.7', 'color': '#2c3e50', 'marginBottom': '10px'}),
                            html.P([
                                html.Strong("Recommended Next Steps for Serious Analysis: "),
                                "(1) Collect and analyze historical loan performance data, (2) Calibrate parameters using maximum likelihood or Bayesian methods, ", 
                                "(3) Implement more sophisticated models (e.g., copulas for dependency, regime-switching for market states, stochastic volatility), ",
                                "(4) Perform extensive stress testing and scenario analysis, (5) Validate against out-of-sample data, ",
                                "(6) Consult with legal, tax, and financial advisors, (7) Review regulatory requirements and compliance issues. ",
                                html.Strong( "write to us krsna.nandula@gmail.com if you need help with custom implementations."),
                            ], style={'fontSize': '12px', 'lineHeight': '1.7', 'color': '#2c3e50', 'marginBottom': '0'}),
                        ], style={'backgroundColor': '#fff3cd', 'padding': '15px', 'borderRadius': '5px', 'border': '1px solid #ffc107'}),
                    ], style={
                'backgroundColor': '#f8f9fa',
                'padding': '25px',
                'borderRadius': '10px',
                'border': '2px solid #dcdde1',
                'marginTop': '30px',
                'marginBottom': '30px',
                'boxShadow': '0 4px 8px rgba(0,0,0,0.15)'
            }),
            
        ], style={
            'width': '80%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'marginLeft': '20%',
            'padding': '20px',
            'overflowY': 'auto',
            'minHeight': '100vh',
            'backgroundColor': '#f5f6fa',
            'boxSizing': 'border-box'
        }),
        
    ], style={'width': '100%'}),
    
], style={'fontFamily': 'Arial, sans-serif', 'position': 'relative'})


# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    [
        Output('mean-irr-card', 'children'),
        Output('median-irr-card', 'children'),
        Output('percentile5-card', 'children'),
        Output('default-rate-card', 'children'),
        Output('irr-histogram', 'figure'),
        Output('irr-cdf', 'figure'),
        Output('collateral-paths', 'figure'),
        Output('cash-flows', 'figure'),
        Output('irr-boxplot', 'figure'),
        Output('scenario-comparison', 'figure'),
        Output('loading-output', 'children'),
    ],
    Input('run-button', 'n_clicks'),
    [
        State('num-sims-slider', 'value'),
        State('loan-coupon-slider', 'value'),
        State('re-drift-slider', 'value'),
        State('re-vol-slider', 'value'),
        State('default-prob-slider', 'value'),
        State('recovery-rate-slider', 'value'),
        State('collateral-slider', 'value'),
        State('ir-vol-slider', 'value'),
    ]
)
def update_dashboard(n_clicks, num_sims, loan_coupon, re_drift, re_vol, 
                    default_prob, recovery_rate, collateral, ir_vol):
    """Main callback to run simulation and update all visualizations."""
    
    if n_clicks == 0:
        # Initial state - show empty charts
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Click 'Run Simulation' to start",
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{
                'text': 'Awaiting simulation...',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 20, 'color': '#7f8c8d'}
            }]
        )
        return ["--"] * 4 + [empty_fig] * 6 + [""]
    
    # Create config with user parameters
    config = SimulationConfig()
    config.NUM_SIMULATIONS = int(num_sims)
    config.LOAN_COUPON = loan_coupon / 100
    config.RE_DRIFT = re_drift / 100
    config.RE_VOL = re_vol / 100
    config.BASE_DEFAULT_PROB = default_prob / 100
    config.RECOVERY_RATE = recovery_rate / 100
    config.INITIAL_COLLATERAL_PER_PROJECT = collateral * 10_000_000
    config.FORWARD_RATE_VOL = ir_vol / 100
    
    # Run simulation
    num_sims = config.NUM_SIMULATIONS
    
    # Step 1: Simulate interest rates
    forward_rates = simulate_forward_rates(config, num_sims)
    discount_factors = compute_discount_factors(forward_rates, config)
    
    # Step 2: Simulate collateral
    collateral_values = simulate_collateral_paths(config, num_sims)
    
    # Step 3: Simulate defaults
    default_indicator, default_time = simulate_defaults(collateral_values, config, num_sims)
    
    # Step 4: Generate cash flows
    pool_cash_flows = generate_loan_cash_flows(
        collateral_values, default_indicator, default_time, config, num_sims
    )
    
    # Step 5: Compute metrics
    irr_array = compute_investor_irr(pool_cash_flows, config, num_sims)
    npv_array = compute_investor_npv(pool_cash_flows, discount_factors, config, num_sims)
    
    # Calculate statistics
    total_defaults = np.sum(default_time <= config.NUM_STEPS)
    default_rate = total_defaults / (num_sims * config.NUM_PROJECTS)
    
    mean_irr = np.mean(irr_array)
    median_irr = np.median(irr_array)
    p5_irr = np.percentile(irr_array, 5)
    
    # Create figures
    
    # 1. IRR Histogram
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=irr_array * 100,
        nbinsx=50,
        name='IRR Distribution',
        marker_color='rgba(52, 152, 219, 0.7)',
        marker_line_color='rgb(41, 128, 185)',
        marker_line_width=1.5
    ))
    fig_hist.add_vline(x=mean_irr * 100, line_dash="dash", line_color="red", 
                      annotation_text=f"Mean: {mean_irr:.1%}")
    fig_hist.add_vline(x=14, line_dash="dot", line_color="green", 
                      annotation_text="Target: 14%", annotation_position="top")
    fig_hist.add_vline(x=16, line_dash="dot", line_color="green", 
                      annotation_text="Target: 16%", annotation_position="top")
    fig_hist.update_layout(
        title={'text': "Investor IRR Distribution", 'font': {'size': 16, 'color': '#2c3e50'}},
        xaxis_title="IRR (%)",
        yaxis_title="Frequency",
        showlegend=False,
        hovermode='x unified',
        height=380,
        margin=dict(l=50, r=30, t=50, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # 2. IRR CDF
    sorted_irr = np.sort(irr_array)
    cdf = np.arange(1, len(sorted_irr) + 1) / len(sorted_irr)
    
    fig_cdf = go.Figure()
    fig_cdf.add_trace(go.Scatter(
        x=sorted_irr * 100,
        y=cdf * 100,
        mode='lines',
        line=dict(color='rgb(46, 204, 113)', width=3),
        name='CDF'
    ))
    fig_cdf.add_vline(x=14, line_dash="dot", line_color="red")
    fig_cdf.add_vline(x=16, line_dash="dot", line_color="red")
    fig_cdf.update_layout(
        title={'text': "Cumulative Distribution of IRR", 'font': {'size': 16, 'color': '#2c3e50'}},
        xaxis_title="IRR (%)",
        yaxis_title="Cumulative Probability (%)",
        showlegend=False,
        hovermode='x unified',
        height=380,
        margin=dict(l=50, r=30, t=50, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # 3. Sample Collateral Paths
    fig_collateral = go.Figure()
    sample_path = 0
    years = np.arange(config.NUM_STEPS + 1) * config.DT
    
    colors = px.colors.qualitative.Set2
    for proj in range(min(5, config.NUM_PROJECTS)):
        fig_collateral.add_trace(go.Scatter(
            x=years,
            y=collateral_values[sample_path, :, proj] / 10_000_000,
            mode='lines',
            name=f'Project {proj+1}',
            line=dict(width=2, color=colors[proj % len(colors)])
        ))
    
    fig_collateral.add_hline(
        y=config.LOAN_PER_PROJECT / 10_000_000,
        line_dash="dash",
        line_color="red",
        annotation_text="Loan Amount"
    )
    fig_collateral.update_layout(
        title={'text': "Collateral Value Evolution (Sample Path)", 'font': {'size': 16, 'color': '#2c3e50'}},
        xaxis_title="Year",
        yaxis_title="Collateral Value (₹ Crore)",
        hovermode='x unified',
        height=380,
        margin=dict(l=50, r=30, t=50, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # 4. Expected Cash Flows
    expected_cf = np.mean(pool_cash_flows[:, 1:] / config.NUM_INVESTORS, axis=0) / 10_000_000
    std_cf = np.std(pool_cash_flows[:, 1:] / config.NUM_INVESTORS, axis=0) / 10_000_000
    years_cf = np.arange(1, config.NUM_STEPS + 1) * config.DT
    
    fig_cf = go.Figure()
    fig_cf.add_trace(go.Bar(
        x=years_cf,
        y=expected_cf,
        name='Expected CF',
        marker_color='rgba(52, 152, 219, 0.7)',
        error_y=dict(type='data', array=std_cf, visible=True)
    ))
    fig_cf.update_layout(
        title={'text': "Expected Annual Cash Flow to Investor", 'font': {'size': 16, 'color': '#2c3e50'}},
        xaxis_title="Year",
        yaxis_title="Cash Flow (₹ Crore)",
        hovermode='x unified',
        height=380,
        margin=dict(l=50, r=30, t=50, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # 5. IRR Box Plot by Percentile
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=irr_array * 100,
        name='IRR',
        marker_color='rgb(52, 152, 219)',
        boxmean='sd'
    ))
    fig_box.add_hline(y=14, line_dash="dot", line_color="green", annotation_text="Target 14%")
    fig_box.add_hline(y=16, line_dash="dot", line_color="green", annotation_text="Target 16%")
    fig_box.update_layout(
        title={'text': "IRR Distribution Box Plot", 'font': {'size': 16, 'color': '#2c3e50'}},
        yaxis_title="IRR (%)",
        showlegend=False,
        height=380,
        margin=dict(l=50, r=30, t=50, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # 6. Scenario Comparison (Probability Buckets)
    buckets = {
        'IRR < 10%': np.mean(irr_array < 0.10) * 100,
        '10-14%': np.mean((irr_array >= 0.10) & (irr_array < 0.14)) * 100,
        '14-16%': np.mean((irr_array >= 0.14) & (irr_array < 0.16)) * 100,
        'IRR > 16%': np.mean(irr_array >= 0.16) * 100,
    }
    
    fig_buckets = go.Figure()
    fig_buckets.add_trace(go.Bar(
        x=list(buckets.keys()),
        y=list(buckets.values()),
        marker_color=['#e74c3c', '#f39c12', '#2ecc71', '#3498db'],
        text=[f"{v:.1f}%" for v in buckets.values()],
        textposition='auto'
    ))
    fig_buckets.update_layout(
        title={'text': "IRR Probability Buckets", 'font': {'size': 16, 'color': '#2c3e50'}},
        xaxis_title="IRR Range",
        yaxis_title="Probability (%)",
        showlegend=False,
        height=380,
        margin=dict(l=50, r=30, t=50, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Update cards
    mean_irr_text = f"{mean_irr:.2%}"
    median_irr_text = f"{median_irr:.2%}"
    p5_irr_text = f"{p5_irr:.2%}"
    default_rate_text = f"{default_rate:.1%}"
    
    return [
        mean_irr_text,
        median_irr_text,
        p5_irr_text,
        default_rate_text,
        fig_hist,
        fig_cdf,
        fig_collateral,
        fig_cf,
        fig_box,
        fig_buckets,
        ""
    ]


# ============================================================================
# RUN SERVER
# ============================================================================

# For Vercel deployment
server = app.server

if __name__ == '__main__':
    print("=" * 80)
    print("REAL ESTATE POOL MONTE CARLO DASHBOARD")
    print("=" * 80)
    print("\nStarting dashboard server...")
    print("\nOpen your browser and navigate to:")
    print("    http://127.0.0.1:8050/")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 80 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=8050)
