#!/usr/bin/env python3
"""
Options Strategy Analyzer
Analyze and visualize options strategies with profit/loss charts and Greeks
High-value tool for options traders - very marketable
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

class OptionsAnalyzer:
    def __init__(self):
        self.strategies = {}
        
    def black_scholes_call(self, S, K, T, r, sigma):
        """Black-Scholes call option price"""
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        return call_price
    
    def black_scholes_put(self, S, K, T, r, sigma):
        """Black-Scholes put option price"""
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        return put_price
    
    def calculate_greeks(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option Greeks"""
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
            
        # Gamma
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta (per day)
        if option_type == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                    r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                    r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        # Vega (per 1% change in volatility)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Rho (per 1% change in interest rate)
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            
        return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}
    
    def calculate_strategy_payoff(self, spot_prices, legs):
        """Calculate strategy payoff across different spot prices"""
        total_payoff = np.zeros_like(spot_prices)
        total_cost = 0
        
        for leg in legs:
            option_type = leg['type']
            strike = leg['strike']
            position = leg['position']  # 'long' or 'short'
            quantity = leg['quantity']
            premium = leg['premium']
            
            # Calculate intrinsic value at expiration
            if option_type == 'call':
                intrinsic = np.maximum(spot_prices - strike, 0)
            else:  # put
                intrinsic = np.maximum(strike - spot_prices, 0)
            
            # Apply position and quantity
            if position == 'long':
                leg_payoff = quantity * intrinsic - quantity * premium
                total_cost += quantity * premium
            else:  # short
                leg_payoff = -quantity * intrinsic + quantity * premium
                total_cost -= quantity * premium
            
            total_payoff += leg_payoff
            
        return total_payoff, total_cost
    
    def get_strategy_template(self, strategy_name, current_price):
        """Get predefined strategy templates"""
        templates = {
            'Long Call': [{
                'type': 'call',
                'strike': current_price * 1.05,
                'position': 'long',
                'quantity': 1,
                'premium': current_price * 0.03
            }],
            
            'Long Put': [{
                'type': 'put',
                'strike': current_price * 0.95,
                'position': 'long',
                'quantity': 1,
                'premium': current_price * 0.03
            }],
            
            'Bull Call Spread': [{
                'type': 'call',
                'strike': current_price,
                'position': 'long',
                'quantity': 1,
                'premium': current_price * 0.04
            }, {
                'type': 'call',
                'strike': current_price * 1.10,
                'position': 'short',
                'quantity': 1,
                'premium': current_price * 0.02
            }],
            
            'Bear Put Spread': [{
                'type': 'put',
                'strike': current_price,
                'position': 'long',
                'quantity': 1,
                'premium': current_price * 0.04
            }, {
                'type': 'put',
                'strike': current_price * 0.90,
                'position': 'short',
                'quantity': 1,
                'premium': current_price * 0.02
            }],
            
            'Iron Condor': [{
                'type': 'put',
                'strike': current_price * 0.90,
                'position': 'short',
                'quantity': 1,
                'premium': current_price * 0.02
            }, {
                'type': 'put',
                'strike': current_price * 0.95,
                'position': 'long',
                'quantity': 1,
                'premium': current_price * 0.03
            }, {
                'type': 'call',
                'strike': current_price * 1.05,
                'position': 'long',
                'quantity': 1,
                'premium': current_price * 0.03
            }, {
                'type': 'call',
                'strike': current_price * 1.10,
                'position': 'short',
                'quantity': 1,
                'premium': current_price * 0.02
            }],
            
            'Straddle': [{
                'type': 'call',
                'strike': current_price,
                'position': 'long',
                'quantity': 1,
                'premium': current_price * 0.05
            }, {
                'type': 'put',
                'strike': current_price,
                'position': 'long',
                'quantity': 1,
                'premium': current_price * 0.05
            }],
            
            'Strangle': [{
                'type': 'call',
                'strike': current_price * 1.05,
                'position': 'long',
                'quantity': 1,
                'premium': current_price * 0.03
            }, {
                'type': 'put',
                'strike': current_price * 0.95,
                'position': 'long',
                'quantity': 1,
                'premium': current_price * 0.03
            }]
        }
        
        return templates.get(strategy_name, [])

def main():
    st.set_page_config(
        page_title="Options Strategy Analyzer",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Options Strategy Analyzer")
    st.markdown("*Professional options strategy analysis with P&L visualization and risk metrics*")
    
    analyzer = OptionsAnalyzer()
    
    # Sidebar - Strategy Configuration
    st.sidebar.header("🎯 Strategy Configuration")
    
    # Market parameters
    st.sidebar.subheader("Market Parameters")
    current_price = st.sidebar.number_input("Current Stock Price ($)", min_value=1.0, value=100.0, step=1.0)
    volatility = st.sidebar.slider("Implied Volatility (%)", 5, 100, 25) / 100
    risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100
    days_to_expiry = st.sidebar.number_input("Days to Expiration", min_value=1, value=30, step=1)
    
    time_to_expiry = days_to_expiry / 365
    
    # Strategy selection
    st.sidebar.subheader("Strategy Setup")
    
    # Quick strategy templates
    strategy_templates = [
        "Custom", "Long Call", "Long Put", "Bull Call Spread", "Bear Put Spread", 
        "Iron Condor", "Straddle", "Strangle"
    ]
    
    selected_template = st.sidebar.selectbox("Strategy Template", strategy_templates)
    
    if selected_template != "Custom":
        legs = analyzer.get_strategy_template(selected_template, current_price)
        st.sidebar.success(f"Loaded {selected_template} template")
    else:
        legs = []
        num_legs = st.sidebar.number_input("Number of Legs", min_value=1, max_value=6, value=1)
        
        st.sidebar.subheader("Strategy Legs")
        for i in range(num_legs):
            st.sidebar.markdown(f"**Leg {i+1}**")
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                option_type = st.selectbox(f"Type", ["call", "put"], key=f"type_{i}")
                position = st.selectbox(f"Position", ["long", "short"], key=f"position_{i}")
            
            with col2:
                strike = st.number_input(f"Strike ($)", min_value=1.0, value=current_price, step=1.0, key=f"strike_{i}")
                quantity = st.number_input(f"Quantity", min_value=1, value=1, step=1, key=f"quantity_{i}")
            
            premium = st.sidebar.number_input(f"Premium ($)", min_value=0.01, value=current_price * 0.03, step=0.01, key=f"premium_{i}")
            
            legs.append({
                'type': option_type,
                'strike': strike,
                'position': position,
                'quantity': quantity,
                'premium': premium
            })
    
    # Analysis
    if legs:
        # Calculate payoff
        spot_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)
        payoff, net_cost = analyzer.calculate_strategy_payoff(spot_range, legs)
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # P&L Chart
            fig = go.Figure()
            
            # Payoff line
            fig.add_trace(go.Scatter(
                x=spot_range,
                y=payoff,
                mode='lines',
                name='P&L at Expiration',
                line=dict(color='blue', width=3)
            ))
            
            # Break-even lines
            breakeven_points = spot_range[np.abs(payoff) < 0.1]
            if len(breakeven_points) > 0:
                for be in breakeven_points:
                    fig.add_vline(x=be, line_dash="dash", line_color="red", 
                                 annotation_text=f"BE: ${be:.2f}")
            
            # Current price line
            fig.add_vline(x=current_price, line_dash="dot", line_color="green",
                         annotation_text=f"Current: ${current_price:.2f}")
            
            # Zero line
            fig.add_hline(y=0, line_color="gray", opacity=0.5)
            
            fig.update_layout(
                title="Strategy Profit & Loss at Expiration",
                xaxis_title="Stock Price ($)",
                yaxis_title="Profit/Loss ($)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Strategy summary
            st.subheader("📊 Strategy Summary")
            
            max_profit = np.max(payoff)
            max_loss = np.min(payoff)
            
            st.metric("Net Cost/Credit", f"${net_cost:.2f}")
            st.metric("Max Profit", f"${max_profit:.2f}" if max_profit < 1e6 else "Unlimited")
            st.metric("Max Loss", f"${max_loss:.2f}" if max_loss > -1e6 else "Unlimited")
            
            # Risk/Reward ratio
            if max_loss < 0 and max_profit > 0:
                risk_reward = abs(max_profit / max_loss)
                st.metric("Risk/Reward", f"{risk_reward:.2f}")
            
            # Breakeven points
            be_points = []
            for i in range(len(payoff)-1):
                if payoff[i] * payoff[i+1] <= 0:  # Sign change indicates breakeven
                    be_price = spot_range[i] + (spot_range[i+1] - spot_range[i]) * (-payoff[i] / (payoff[i+1] - payoff[i]))
                    be_points.append(be_price)
            
            if be_points:
                st.subheader("🎯 Breakeven Points")
                for i, be in enumerate(be_points):
                    st.write(f"BE {i+1}: ${be:.2f}")
        
        # Strategy details table
        st.subheader("📋 Strategy Details")
        
        legs_df = pd.DataFrame(legs)
        legs_df['Strike'] = legs_df['strike'].apply(lambda x: f"${x:.2f}")
        legs_df['Premium'] = legs_df['premium'].apply(lambda x: f"${x:.2f}")
        legs_df['Type'] = legs_df['type'].str.title()
        legs_df['Position'] = legs_df['position'].str.title()
        legs_df['Quantity'] = legs_df['quantity']
        
        display_df = legs_df[['Type', 'Position', 'Strike', 'Quantity', 'Premium']]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Greeks analysis for individual legs
        st.subheader("📐 Greeks Analysis")
        
        greeks_data = []
        for i, leg in enumerate(legs):
            greeks = analyzer.calculate_greeks(
                current_price, leg['strike'], time_to_expiry, 
                risk_free_rate, volatility, leg['type']
            )
            
            # Adjust for position and quantity
            multiplier = leg['quantity'] if leg['position'] == 'long' else -leg['quantity']
            
            greeks_data.append({
                'Leg': f"{leg['position'].title()} {leg['type'].title()}",
                'Delta': f"{greeks['delta'] * multiplier:.3f}",
                'Gamma': f"{greeks['gamma'] * multiplier:.3f}",
                'Theta': f"${greeks['theta'] * multiplier:.2f}",
                'Vega': f"${greeks['vega'] * multiplier:.2f}",
                'Rho': f"${greeks['rho'] * multiplier:.2f}"
            })
        
        # Calculate total portfolio Greeks
        total_greeks = {
            'delta': sum([analyzer.calculate_greeks(current_price, leg['strike'], time_to_expiry, risk_free_rate, volatility, leg['type'])['delta'] * (leg['quantity'] if leg['position'] == 'long' else -leg['quantity']) for leg in legs]),
            'gamma': sum([analyzer.calculate_greeks(current_price, leg['strike'], time_to_expiry, risk_free_rate, volatility, leg['type'])['gamma'] * (leg['quantity'] if leg['position'] == 'long' else -leg['quantity']) for leg in legs]),
            'theta': sum([analyzer.calculate_greeks(current_price, leg['strike'], time_to_expiry, risk_free_rate, volatility, leg['type'])['theta'] * (leg['quantity'] if leg['position'] == 'long' else -leg['quantity']) for leg in legs]),
            'vega': sum([analyzer.calculate_greeks(current_price, leg['strike'], time_to_expiry, risk_free_rate, volatility, leg['type'])['vega'] * (leg['quantity'] if leg['position'] == 'long' else -leg['quantity']) for leg in legs]),
            'rho': sum([analyzer.calculate_greeks(current_price, leg['strike'], time_to_expiry, risk_free_rate, volatility, leg['type'])['rho'] * (leg['quantity'] if leg['position'] == 'long' else -leg['quantity']) for leg in legs])
        }
        
        greeks_data.append({
            'Leg': 'TOTAL PORTFOLIO',
            'Delta': f"{total_greeks['delta']:.3f}",
            'Gamma': f"{total_greeks['gamma']:.3f}",
            'Theta': f"${total_greeks['theta']:.2f}",
            'Vega': f"${total_greeks['vega']:.2f}",
            'Rho': f"${total_greeks['rho']:.2f}"
        })
        
        greeks_df = pd.DataFrame(greeks_data)
        st.dataframe(greeks_df, use_container_width=True, hide_index=True)
        
        # Sensitivity analysis
        st.subheader("🔍 Sensitivity Analysis")
        
        # Price sensitivity
        col1, col2 = st.columns(2)
        
        with col1:
            price_range = np.linspace(current_price * 0.8, current_price * 1.2, 50)
            current_payoff = []
            
            for price in price_range:
                # Calculate current value (not just expiration value)
                leg_values = []
                for leg in legs:
                    if leg['type'] == 'call':
                        value = analyzer.black_scholes_call(price, leg['strike'], time_to_expiry, risk_free_rate, volatility)
                    else:
                        value = analyzer.black_scholes_put(price, leg['strike'], time_to_expiry, risk_free_rate, volatility)
                    
                    if leg['position'] == 'long':
                        leg_values.append(leg['quantity'] * value - leg['quantity'] * leg['premium'])
                    else:
                        leg_values.append(-leg['quantity'] * value + leg['quantity'] * leg['premium'])
                
                current_payoff.append(sum(leg_values))
            
            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(
                x=price_range,
                y=current_payoff,
                mode='lines',
                name='Current P&L',
                line=dict(color='green', width=2)
            ))
            
            # Add expiration P&L for comparison
            exp_payoff, _ = analyzer.calculate_strategy_payoff(price_range, legs)
            fig_sens.add_trace(go.Scatter(
                x=price_range,
                y=exp_payoff,
                mode='lines',
                name='At Expiration',
                line=dict(color='blue', width=2, dash='dash')
            ))
            
            fig_sens.add_vline(x=current_price, line_color="red", line_dash="dot")
            fig_sens.add_hline(y=0, line_color="gray", opacity=0.5)
            
            fig_sens.update_layout(
                title="Price Sensitivity",
                xaxis_title="Stock Price ($)",
                yaxis_title="P&L ($)"
            )
            
            st.plotly_chart(fig_sens, use_container_width=True)
        
        with col2:
            # Time decay analysis
            time_points = np.linspace(time_to_expiry, 0.001, 20)
            time_decay_pnl = []
            
            for t in time_points:
                leg_values = []
                for leg in legs:
                    if leg['type'] == 'call':
                        value = analyzer.black_scholes_call(current_price, leg['strike'], t, risk_free_rate, volatility)
                    else:
                        value = analyzer.black_scholes_put(current_price, leg['strike'], t, risk_free_rate, volatility)
                    
                    if leg['position'] == 'long':
                        leg_values.append(leg['quantity'] * value - leg['quantity'] * leg['premium'])
                    else:
                        leg_values.append(-leg['quantity'] * value + leg['quantity'] * leg['premium'])
                
                time_decay_pnl.append(sum(leg_values))
            
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=time_points * 365,
                y=time_decay_pnl,
                mode='lines',
                name='P&L',
                line=dict(color='red', width=2)
            ))
            
            fig_time.add_hline(y=0, line_color="gray", opacity=0.5)
            
            fig_time.update_layout(
                title="Time Decay (at current price)",
                xaxis_title="Days to Expiration",
                yaxis_title="P&L ($)"
            )
            
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Export functionality
        st.subheader("📥 Export Analysis")
        
        # Create summary report
        report = f"""
Options Strategy Analysis Report
===============================

Strategy: {selected_template if selected_template != 'Custom' else 'Custom Strategy'}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Market Parameters:
- Current Price: ${current_price:.2f}
- Volatility: {volatility*100:.1f}%
- Risk-Free Rate: {risk_free_rate*100:.1f}%
- Days to Expiration: {days_to_expiry}

Strategy Details:
{display_df.to_string(index=False)}

Risk Metrics:
- Net Cost/Credit: ${net_cost:.2f}
- Maximum Profit: ${max_profit:.2f}
- Maximum Loss: ${max_loss:.2f}

Total Portfolio Greeks:
- Delta: {total_greeks['delta']:.3f}
- Gamma: {total_greeks['gamma']:.3f}
- Theta: ${total_greeks['theta']:.2f}
- Vega: ${total_greeks['vega']:.2f}
- Rho: ${total_greeks['rho']:.2f}
        """
        
        st.download_button(
            label="📄 Download Analysis Report",
            data=report,
            file_name=f"options_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )
    
    # Instructions
    with st.expander("📖 How to Use This Tool"):
        st.markdown("""
        ### Options Strategy Analyzer Guide
        
        1. **Set Market Parameters**: Enter current stock price, volatility, risk-free rate, and time to expiration
        2. **Choose Strategy**: Select a template or build custom strategy
        3. **Analyze Results**: Review P&L charts, Greeks, and sensitivity analysis
        
        ### Key Concepts:
        - **Delta**: Price sensitivity (how much option price changes per $1 stock move)
        - **Gamma**: Rate of change of delta
        - **Theta**: Time decay (how much value lost per day)
        - **Vega**: Volatility sensitivity
        - **Rho**: Interest rate sensitivity
        
        ### Popular Strategies:
        - **Bull Call Spread**: Limited upside, limited risk
        - **Iron Condor**: Profit from low volatility
        - **Straddle**: Profit from high volatility
        - **Covered Call**: Income generation
        
        *Note: This tool uses Black-Scholes pricing for demonstration purposes.*
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p><strong>Developed by:</strong> <a href="https://algoscripts.dev" target="_blank" style="color: #1f77b4; text-decoration: none;">Algoscripts.dev</a></p>
            <p>🐦 <strong>X:</strong> <a href="https://x.com/devsec_ai" target="_blank" style="color: #1DA1F2; text-decoration: none;">@devsec_ai</a> | 
            <a href="https://x.com/devsec_ai" target="_blank" style="color: #1DA1F2; text-decoration: none;">x.com/devsec_ai</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
