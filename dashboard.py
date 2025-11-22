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
    generate_cash_flows_with_exit,
    compute_investor_irr,
    compute_investor_npv,
    calculate_capital_company_fees,
    calculate_debt_balance_at_exit,
    apply_waterfall_distribution,
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
            
            # Summary Statistics Cards - Row 1 (Investor Metrics)
            html.Div([
                html.Div([
                    html.H4("Investor Mean IRR", style={'color': '#7f8c8d', 'marginBottom': '5px', 'fontSize': '13px'}),
                    html.H2(id='mean-irr-card', children="--", 
                           style={'color': '#2980b9', 'marginTop': '0', 'marginBottom': '5px', 'fontSize': '24px'}),
                    html.P("Average investor return across all scenarios", 
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
            
            # Summary Statistics Cards - Row 2 (Capital Company & Sponsor Metrics)
            html.Div([
                html.Div([
                    html.H4("Capital Co IRR", style={'color': '#7f8c8d', 'marginBottom': '5px', 'fontSize': '13px'}),
                    html.H2(id='cc-irr-card', children="--", 
                           style={'color': '#8e44ad', 'marginTop': '0', 'marginBottom': '5px', 'fontSize': '24px'}),
                    html.P("Capital company return from fees & carry", 
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
                    html.H4("Sponsor IRR", style={'color': '#7f8c8d', 'marginBottom': '5px', 'fontSize': '13px'}),
                    html.H2(id='sponsor-irr-card', children="--",
                           style={'color': '#16a085', 'marginTop': '0', 'marginBottom': '5px', 'fontSize': '24px'}),
                    html.P("Equity sponsor return on 30% investment", 
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
                    html.H4("Avg CC Fees", style={'color': '#7f8c8d', 'marginBottom': '5px', 'fontSize': '13px'}),
                    html.H2(id='cc-fees-card', children="--",
                           style={'color': '#d35400', 'marginTop': '0', 'marginBottom': '5px', 'fontSize': '24px'}),
                    html.P("Mgmt + performance + origination fees", 
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
            
            # Charts row 4 - Iteration 2 Metrics
            html.Div([
                html.Div([
                    dcc.Graph(id='waterfall-chart', style={'height': '380px'}),
                    html.P("Waterfall distribution showing how exit sale proceeds are allocated across four tiers: (1) Debt repayment to investors, (2) Return of equity capital to sponsor, (3) Preferred return to investors/sponsor, (4) Profit split between investors (80%) and capital company (20%). This visualizes the priority structure of cash flows at property exit.",
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
                    dcc.Graph(id='stakeholder-comparison', style={'height': '380px'}),
                    html.P("Comparison of returns across all three stakeholders in the deal structure. Investors (70% debt) earn coupon + exit proceeds, Capital Company earns management fees + performance carry, Sponsor (30% equity) earns residual profits after debt repayment. This shows how value is created and distributed across the ecosystem.",
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
        Output('cc-irr-card', 'children'),
        Output('sponsor-irr-card', 'children'),
        Output('cc-fees-card', 'children'),
        Output('irr-histogram', 'figure'),
        Output('irr-cdf', 'figure'),
        Output('collateral-paths', 'figure'),
        Output('cash-flows', 'figure'),
        Output('irr-boxplot', 'figure'),
        Output('scenario-comparison', 'figure'),
        Output('waterfall-chart', 'figure'),
        Output('stakeholder-comparison', 'figure'),
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
        # Return: 7 card values + 8 figures + 1 loading message = 16 values
        return ["--"] * 7 + [empty_fig] * 8 + [""]
    
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
    
    # Step 4: Generate cash flows WITH EXIT SALE (Iteration 2)
    investor_cf, capital_co_cf, sponsor_cf = generate_cash_flows_with_exit(
        collateral_values, default_indicator, default_time, config, num_sims
    )
    
    # Step 5: Compute metrics for all stakeholders
    investor_irr = compute_investor_irr(investor_cf, config, num_sims)
    capital_co_irr = compute_investor_irr(capital_co_cf, config, num_sims)
    sponsor_irr = compute_investor_irr(sponsor_cf, config, num_sims)
    
    # Calculate statistics
    total_defaults = np.sum(default_time <= config.NUM_STEPS)
    default_rate = total_defaults / (num_sims * config.NUM_PROJECTS)
    
    # Compute average capital company fees
    mean_investor_irr = np.mean(investor_irr)
    total_profits = np.sum(investor_cf[:, -1]) - config.TOTAL_CORPUS * config.INVESTOR_DEBT_PCT
    avg_cc_fees = calculate_capital_company_fees(
        aum=config.TOTAL_CORPUS,
        years=config.PROJECT_EXIT_YEAR,
        total_profits=max(0, total_profits / num_sims),
        investor_irr=mean_investor_irr,
        config=config
    )
    
    mean_irr = np.mean(investor_irr)
    median_irr = np.median(investor_irr)
    p5_irr = np.percentile(investor_irr, 5)
    
    # Capital company and sponsor stats (with safety checks for empty arrays)
    valid_cc_irr = capital_co_irr[capital_co_irr > -0.99]
    valid_sponsor_irr = sponsor_irr[sponsor_irr > -0.99]
    mean_cc_irr = np.mean(valid_cc_irr) if len(valid_cc_irr) > 0 else 0.0
    mean_sponsor_irr = np.mean(valid_sponsor_irr) if len(valid_sponsor_irr) > 0 else 0.0
    
    # Create figures
    
    # 1. IRR Histogram
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=investor_irr * 100,
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
    sorted_irr = np.sort(investor_irr)
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
    
    # 4. Expected Cash Flows (using investor_cf from Iteration 2)
    expected_cf = np.mean(investor_cf[:, 1:] / config.NUM_INVESTORS, axis=0) / 10_000_000
    std_cf = np.std(investor_cf[:, 1:] / config.NUM_INVESTORS, axis=0) / 10_000_000
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
        y=investor_irr * 100,
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
        'IRR < 10%': np.mean(investor_irr < 0.10) * 100,
        '10-14%': np.mean((investor_irr >= 0.10) & (investor_irr < 0.14)) * 100,
        '14-16%': np.mean((investor_irr >= 0.14) & (investor_irr < 0.16)) * 100,
        'IRR > 16%': np.mean(investor_irr >= 0.16) * 100,
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
    
    # 7. Waterfall Distribution Chart (Average Scenario)
    # Simulate one average scenario to show waterfall breakdown
    avg_sale_price = config.INITIAL_COLLATERAL_PER_PROJECT * config.NUM_PROJECTS * (1 + config.RE_DRIFT) ** config.PROJECT_EXIT_YEAR
    debt_owed = calculate_debt_balance_at_exit(config)
    equity_capital = config.TOTAL_CORPUS * config.SPONSOR_EQUITY_PCT
    waterfall = apply_waterfall_distribution(avg_sale_price, debt_owed, equity_capital, config)
    
    # Calculate waterfall components
    tier1_debt = waterfall['to_debt_holders']
    tier2_equity = waterfall['to_equity_sponsor']
    tier3_preferred = max(0, waterfall['to_equity_sponsor'] - equity_capital) if waterfall['to_equity_sponsor'] > equity_capital else 0
    tier4_profit = waterfall['to_capital_company']
    
    # Use bar chart instead of waterfall for clearer visualization
    fig_waterfall = go.Figure()
    
    categories = ['Debt Holders', 'Equity Sponsor', 'Capital Co (Carry)']
    values = [tier1_debt / 10_000_000, tier2_equity / 10_000_000, tier4_profit / 10_000_000]
    colors = ['#3498db', '#16a085', '#8e44ad']
    
    fig_waterfall.add_trace(go.Bar(
        x=categories,
        y=values,
        text=[f"₹{v:.2f}Cr" for v in values],
        textposition='auto',
        marker_color=colors
    ))
    
    fig_waterfall.update_layout(
        title={'text': "Exit Sale Distribution (Average Scenario)", 'font': {'size': 16, 'color': '#2c3e50'}},
        xaxis_title="Stakeholder",
        yaxis_title="Amount (₹ Crore)",
        showlegend=False,
        height=380,
        margin=dict(l=50, r=30, t=50, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Add total sale price as annotation
    fig_waterfall.add_annotation(
        x=1,
        y=max(values) * 1.1,
        text=f"Total Exit Sale: ₹{avg_sale_price/10_000_000:.1f}Cr",
        showarrow=False,
        font=dict(size=12, color='#2c3e50', weight='bold')
    )
    
    # 8. Stakeholder IRR Comparison
    stakeholder_data = {
        'Stakeholder': ['Investors (Debt)', 'Capital Company', 'Sponsor (Equity)'],
        'Mean IRR': [mean_irr * 100, mean_cc_irr * 100, mean_sponsor_irr * 100],
        'Investment': [
            config.TOTAL_CORPUS * config.INVESTOR_DEBT_PCT / 10_000_000,
            0,  # Capital company invests time/effort, not capital
            config.TOTAL_CORPUS * config.SPONSOR_EQUITY_PCT / 10_000_000
        ]
    }
    
    fig_stakeholder = go.Figure()
    fig_stakeholder.add_trace(go.Bar(
        name='Mean IRR',
        x=stakeholder_data['Stakeholder'],
        y=stakeholder_data['Mean IRR'],
        text=[f"{v:.1f}%" for v in stakeholder_data['Mean IRR']],
        textposition='auto',
        marker_color=['#2980b9', '#8e44ad', '#16a085']
    ))
    fig_stakeholder.update_layout(
        title={'text': "Stakeholder Returns Comparison", 'font': {'size': 16, 'color': '#2c3e50'}},
        xaxis_title="Stakeholder",
        yaxis_title="IRR (%)",
        showlegend=False,
        height=380,
        margin=dict(l=50, r=30, t=50, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Add investment size annotations
    for i, (stakeholder, investment) in enumerate(zip(stakeholder_data['Stakeholder'], stakeholder_data['Investment'])):
        if investment > 0:
            fig_stakeholder.add_annotation(
                x=i,
                y=-5,
                text=f"₹{investment:.1f}Cr",
                showarrow=False,
                font=dict(size=10, color='#7f8c8d')
            )
    
    # Update cards (with safety checks for valid values)
    mean_irr_text = f"{mean_irr:.2%}" if not np.isnan(mean_irr) else "N/A"
    median_irr_text = f"{median_irr:.2%}" if not np.isnan(median_irr) else "N/A"
    p5_irr_text = f"{p5_irr:.2%}" if not np.isnan(p5_irr) else "N/A"
    default_rate_text = f"{default_rate:.1%}" if not np.isnan(default_rate) else "N/A"
    cc_irr_text = f"{mean_cc_irr:.2%}" if mean_cc_irr > 0 else "0.00%"
    sponsor_irr_text = f"{mean_sponsor_irr:.2%}" if mean_sponsor_irr > 0 else "0.00%"
    cc_fees_text = f"₹{avg_cc_fees['total_revenue'] / 10_000_000:.2f}Cr"
    
    return [
        mean_irr_text,
        median_irr_text,
        p5_irr_text,
        default_rate_text,
        cc_irr_text,
        sponsor_irr_text,
        cc_fees_text,
        fig_hist,
        fig_cdf,
        fig_collateral,
        fig_cf,
        fig_box,
        fig_buckets,
        fig_waterfall,
        fig_stakeholder,
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
