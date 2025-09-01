#!/usr/bin/env python3
"""
Portfolio Risk Analyzer
Advanced risk analysis tool for investment portfolios
Calculates VaR, correlation analysis, diversification metrics, and stress testing
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PortfolioRiskAnalyzer:
    def __init__(self):
        self.portfolio = {}
        self.returns = pd.DataFrame()
        self.prices = pd.DataFrame()
        
    def add_position(self, symbol, weight):
        """Add a position to the portfolio"""
        self.portfolio[symbol] = weight
        
    def fetch_portfolio_data(self, period="2y"):
        """Fetch historical data for all portfolio positions"""
        symbols = list(self.portfolio.keys())
        data = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(symbols):
            try:
                status_text.text(f'Fetching data for {symbol}... ({i+1}/{len(symbols)})')
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if len(hist) > 0:
                    data[symbol] = hist['Close']
                    
                progress_bar.progress((i + 1) / len(symbols))
                
            except Exception as e:
                st.error(f"Error fetching {symbol}: {e}")
                continue
        
        status_text.text('Data fetching complete!')
        progress_bar.empty()
        status_text.empty()
        
        if data:
            self.prices = pd.DataFrame(data)
            self.returns = self.prices.pct_change().dropna()
            return True
        return False
    
    def calculate_portfolio_returns(self):
        """Calculate portfolio returns based on weights"""
        weights = np.array([self.portfolio[symbol] for symbol in self.prices.columns])
        portfolio_returns = (self.returns * weights).sum(axis=1)
        return portfolio_returns
    
    def calculate_var(self, returns, confidence_level=0.05):
        """Calculate Value at Risk (VaR)"""
        var = np.percentile(returns, confidence_level * 100)
        return var
    
    def calculate_cvar(self, returns, confidence_level=0.05):
        """Calculate Conditional Value at Risk (CVaR/Expected Shortfall)"""
        var = self.calculate_var(returns, confidence_level)
        cvar = returns[returns <= var].mean()
        return cvar
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        return max_drawdown, drawdown
    
    def monte_carlo_simulation(self, num_simulations=1000, time_horizon=252):
        """Run Monte Carlo simulation for portfolio"""
        portfolio_returns = self.calculate_portfolio_returns()
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # Generate random returns
        simulated_returns = np.random.normal(
            mean_return, std_return, 
            (num_simulations, time_horizon)
        )
        
        # Calculate cumulative returns for each simulation
        cumulative_returns = (1 + simulated_returns).cumprod(axis=1)
        
        return cumulative_returns
    
    def calculate_correlation_matrix(self):
        """Calculate correlation matrix of returns"""
        return self.returns.corr()
    
    def calculate_risk_metrics(self):
        """Calculate comprehensive risk metrics"""
        portfolio_returns = self.calculate_portfolio_returns()
        
        # Basic statistics
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility
        
        # Risk metrics
        var_95 = self.calculate_var(portfolio_returns, 0.05)
        var_99 = self.calculate_var(portfolio_returns, 0.01)
        cvar_95 = self.calculate_cvar(portfolio_returns, 0.05)
        cvar_99 = self.calculate_cvar(portfolio_returns, 0.01)
        
        max_dd, _ = self.calculate_max_drawdown(portfolio_returns)
        
        # Additional metrics
        skewness = stats.skew(portfolio_returns)
        kurtosis = stats.kurtosis(portfolio_returns)
        
        return {
            'Annual Return': annual_return,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'VaR (95%)': var_95,
            'VaR (99%)': var_99,
            'CVaR (95%)': cvar_95,
            'CVaR (99%)': cvar_99,
            'Max Drawdown': max_dd,
            'Skewness': skewness,
            'Kurtosis': kurtosis
        }
    
    def stress_test(self, scenarios):
        """Perform stress testing under different scenarios"""
        portfolio_returns = self.calculate_portfolio_returns()
        results = {}
        
        for scenario_name, shock in scenarios.items():
            # Apply shock to returns
            stressed_returns = portfolio_returns + shock
            stressed_cumulative = (1 + stressed_returns).cumprod()
            
            results[scenario_name] = {
                'Total Return': stressed_cumulative.iloc[-1] - 1,
                'Max Drawdown': self.calculate_max_drawdown(stressed_returns)[0],
                'VaR (95%)': self.calculate_var(stressed_returns, 0.05)
            }
        
        return results

def main():
    st.set_page_config(
        page_title="Portfolio Risk Analyzer",
        page_icon="⚠️",
        layout="wide"
    )
    
    st.title("⚠️ Portfolio Risk Analyzer")
    st.markdown("*Professional portfolio risk analysis with VaR, stress testing, and Monte Carlo simulation*")
    
    # Initialize analyzer
    analyzer = PortfolioRiskAnalyzer()
    
    # Sidebar - Portfolio Input
    st.sidebar.header("📊 Portfolio Configuration")
    
    # Number of positions
    num_positions = st.sidebar.number_input("Number of positions", min_value=2, max_value=20, value=5)
    
    # Input positions
    st.sidebar.subheader("Portfolio Positions")
    positions = {}
    total_weight = 0
    
    for i in range(num_positions):
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            symbol = st.text_input(f"Symbol {i+1}", value=f"{'AAPL MSFT GOOGL AMZN TSLA'.split()[i] if i < 5 else ''}", key=f"symbol_{i}")
        with col2:
            weight = st.number_input(f"Weight", min_value=0.0, max_value=100.0, value=20.0, step=1.0, key=f"weight_{i}") / 100
        
        if symbol and weight > 0:
            positions[symbol] = weight
            total_weight += weight
    
    # Validate weights
    if total_weight > 0:
        # Normalize weights
        for symbol in positions:
            analyzer.add_position(symbol, positions[symbol] / total_weight)
        
        st.sidebar.success(f"Total weight: {total_weight:.1%} (normalized)")
        
        # Analysis parameters
        st.sidebar.subheader("Analysis Parameters")
        data_period = st.sidebar.selectbox("Data Period", ["1y", "2y", "3y", "5y"], index=1)
        confidence_levels = st.sidebar.multiselect("VaR Confidence Levels", [90, 95, 99], default=[95, 99])
        
        # Run analysis button
        if st.sidebar.button("🔍 Analyze Portfolio", type="primary"):
            with st.spinner("Fetching data and calculating risk metrics..."):
                success = analyzer.fetch_portfolio_data(period=data_period)
                
                if success:
                    # Calculate metrics
                    risk_metrics = analyzer.calculate_risk_metrics()
                    portfolio_returns = analyzer.calculate_portfolio_returns()
                    
                    # Display key metrics
                    st.subheader("📈 Portfolio Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Annual Return", f"{risk_metrics['Annual Return']:.2%}")
                    with col2:
                        st.metric("Annual Volatility", f"{risk_metrics['Annual Volatility']:.2%}")
                    with col3:
                        st.metric("Sharpe Ratio", f"{risk_metrics['Sharpe Ratio']:.2f}")
                    with col4:
                        st.metric("Max Drawdown", f"{risk_metrics['Max Drawdown']:.2%}")
                    
                    # Risk metrics table
                    st.subheader("⚠️ Risk Metrics")
                    risk_df = pd.DataFrame({
                        'Metric': ['VaR (95%)', 'VaR (99%)', 'CVaR (95%)', 'CVaR (99%)', 'Skewness', 'Kurtosis'],
                        'Value': [
                            f"{risk_metrics['VaR (95%)']:.2%}",
                            f"{risk_metrics['VaR (99%)']:.2%}",
                            f"{risk_metrics['CVaR (95%)']:.2%}",
                            f"{risk_metrics['CVaR (99%)']:.2%}",
                            f"{risk_metrics['Skewness']:.3f}",
                            f"{risk_metrics['Kurtosis']:.3f}"
                        ]
                    })
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.dataframe(risk_df, use_container_width=True, hide_index=True)
                    
                    # Portfolio composition pie chart
                    with col2:
                        fig = px.pie(
                            values=list(analyzer.portfolio.values()),
                            names=list(analyzer.portfolio.keys()),
                            title="Portfolio Composition"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Returns distribution
                    st.subheader("📊 Returns Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Returns histogram
                        fig = px.histogram(
                            x=portfolio_returns,
                            nbins=50,
                            title="Portfolio Returns Distribution",
                            labels={'x': 'Daily Returns', 'y': 'Frequency'}
                        )
                        
                        # Add VaR lines
                        fig.add_vline(x=risk_metrics['VaR (95%)'], line_dash="dash", 
                                     line_color="red", annotation_text="VaR 95%")
                        fig.add_vline(x=risk_metrics['VaR (99%)'], line_dash="dash", 
                                     line_color="darkred", annotation_text="VaR 99%")
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Cumulative returns
                        cumulative_returns = (1 + portfolio_returns).cumprod()
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=cumulative_returns.index,
                            y=cumulative_returns.values,
                            mode='lines',
                            name='Portfolio',
                            line=dict(width=2)
                        ))
                        
                        fig.update_layout(
                            title="Cumulative Portfolio Returns",
                            xaxis_title="Date",
                            yaxis_title="Cumulative Return"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Correlation analysis
                    st.subheader("🔗 Correlation Analysis")
                    correlation_matrix = analyzer.calculate_correlation_matrix()
                    
                    fig = px.imshow(
                        correlation_matrix,
                        text_auto=True,
                        aspect="auto",
                        title="Asset Correlation Matrix",
                        color_continuous_scale='RdBu'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Monte Carlo simulation
                    st.subheader("🎲 Monte Carlo Simulation")
                    
                    with st.spinner("Running Monte Carlo simulation..."):
                        num_sims = st.selectbox("Number of simulations", [100, 500, 1000], index=1)
                        time_horizon = st.selectbox("Time horizon (days)", [30, 90, 252], index=2)
                        
                        simulations = analyzer.monte_carlo_simulation(num_sims, time_horizon)
                        
                        # Plot simulation results
                        fig = go.Figure()
                        
                        # Plot individual simulation paths (sample)
                        sample_size = min(50, num_sims)
                        for i in range(sample_size):
                            fig.add_trace(go.Scatter(
                                x=list(range(time_horizon)),
                                y=simulations[i],
                                mode='lines',
                                line=dict(color='lightblue', width=0.5),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                        
                        # Add percentiles
                        percentiles = [5, 50, 95]
                        colors = ['red', 'blue', 'red']
                        names = ['5th percentile', 'Median', '95th percentile']
                        
                        for p, color, name in zip(percentiles, colors, names):
                            values = np.percentile(simulations, p, axis=0)
                            fig.add_trace(go.Scatter(
                                x=list(range(time_horizon)),
                                y=values,
                                mode='lines',
                                line=dict(color=color, width=3),
                                name=name
                            ))
                        
                        fig.update_layout(
                            title=f"Monte Carlo Simulation ({num_sims} paths, {time_horizon} days)",
                            xaxis_title="Days",
                            yaxis_title="Portfolio Value"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Simulation statistics
                        final_values = simulations[:, -1]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Expected Return", f"{(final_values.mean() - 1):.2%}")
                        with col2:
                            st.metric("Probability of Loss", f"{(final_values < 1).mean():.2%}")
                        with col3:
                            st.metric("5th Percentile", f"{(np.percentile(final_values, 5) - 1):.2%}")
                        with col4:
                            st.metric("95th Percentile", f"{(np.percentile(final_values, 95) - 1):.2%}")
                    
                    # Stress testing
                    st.subheader("🚨 Stress Testing")
                    
                    stress_scenarios = {
                        'Market Crash (-20%)': -0.20,
                        'Severe Recession (-10%)': -0.10,
                        'Moderate Downturn (-5%)': -0.05,
                        'Black Swan (-30%)': -0.30
                    }
                    
                    stress_results = analyzer.stress_test(stress_scenarios)
                    
                    stress_df = pd.DataFrame(stress_results).T
                    stress_df = stress_df.round(4)
                    
                    st.dataframe(stress_df, use_container_width=True)
                    
                    # Download results
                    st.subheader("📥 Export Results")
                    
                    # Prepare export data
                    export_data = {
                        'Portfolio Composition': analyzer.portfolio,
                        'Risk Metrics': risk_metrics,
                        'Stress Test Results': stress_results
                    }
                    
                    # Create downloadable report
                    report = f"""
# Portfolio Risk Analysis Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Portfolio Composition
{pd.DataFrame(list(analyzer.portfolio.items()), columns=['Symbol', 'Weight']).to_string(index=False)}

## Risk Metrics
{pd.DataFrame(list(risk_metrics.items()), columns=['Metric', 'Value']).to_string(index=False)}

## Stress Test Results
{pd.DataFrame(stress_results).T.to_string()}
                    """
                    
                    st.download_button(
                        label="📄 Download Risk Report",
                        data=report,
                        file_name=f"portfolio_risk_report_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
                    
                else:
                    st.error("Failed to fetch portfolio data. Please check your symbols and try again.")
    else:
        st.sidebar.warning("Please add at least 2 positions with positive weights.")
    
    # Instructions
    with st.expander("📖 How to Use This Tool"):
        st.markdown("""
        ### Portfolio Risk Analyzer Guide
        
        1. **Add Positions**: Enter stock symbols and their weights (as percentages)
        2. **Configure Analysis**: Choose data period and confidence levels
        3. **Review Results**: Analyze risk metrics, correlations, and stress tests
        
        ### Key Metrics Explained:
        - **VaR (Value at Risk)**: Maximum expected loss over a given time period at a specified confidence level
        - **CVaR (Conditional VaR)**: Average loss beyond the VaR threshold
        - **Sharpe Ratio**: Risk-adjusted return (higher is better)
        - **Max Drawdown**: Largest peak-to-trough decline
        - **Skewness**: Asymmetry of return distribution (negative = more left tail risk)
        - **Kurtosis**: Tail heaviness (higher = more extreme events)
        
        ### Stress Testing:
        Tests portfolio performance under adverse market conditions to identify vulnerabilities.
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
