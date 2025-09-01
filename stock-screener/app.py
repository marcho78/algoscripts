#!/usr/bin/env python3
"""
Advanced Stock Screener
A comprehensive tool for screening stocks based on technical and fundamental criteria
Perfect for portfolio demonstration - shows data analysis, visualization, and practical finance knowledge
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockScreener:
    def __init__(self):
        self.sp500_symbols = self._get_sp500_symbols()
        
    def _get_sp500_symbols(self):
        """Get S&P 500 symbols (simplified list for demo)"""
        # In production, you'd fetch this dynamically
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
            'UNH', 'JNJ', 'V', 'WMT', 'JPM', 'MA', 'PG', 'CVX', 'HD', 'ABBV',
            'BAC', 'ORCL', 'KO', 'AVGO', 'PEP', 'COST', 'TMO', 'MRK', 'NFLX',
            'ABT', 'ACN', 'LLY', 'VZ', 'ADBE', 'NKE', 'DHR', 'TXN', 'NEE',
            'QCOM', 'PM', 'RTX', 'UPS', 'LOW', 'SBUX', 'HON', 'AMD', 'INTC'
        ]
    
    def fetch_stock_data(self, symbols, period="3mo"):
        """Fetch stock data for multiple symbols"""
        data = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(symbols):
            try:
                status_text.text(f'Fetching data for {symbol}... ({i+1}/{len(symbols)})')
                ticker = yf.Ticker(symbol)
                
                # Get price data
                hist = ticker.history(period=period)
                if len(hist) < 20:  # Need minimum data
                    continue
                    
                # Get basic info
                info = ticker.info
                
                # Calculate technical indicators
                current_price = hist['Close'][-1]
                sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
                sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
                volume_avg = hist['Volume'].rolling(20).mean().iloc[-1]
                
                # Calculate returns
                returns_1d = ((hist['Close'][-1] / hist['Close'][-2]) - 1) * 100
                returns_1w = ((hist['Close'][-1] / hist['Close'][-5]) - 1) * 100
                returns_1m = ((hist['Close'][-1] / hist['Close'][-20]) - 1) * 100
                
                # Calculate RSI
                rsi = self._calculate_rsi(hist['Close'])
                
                # Calculate volatility
                volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100
                
                data[symbol] = {
                    'Symbol': symbol,
                    'Price': current_price,
                    'SMA_20': sma_20,
                    'SMA_50': sma_50,
                    'Volume_Avg': volume_avg,
                    'Volume_Current': hist['Volume'][-1],
                    'Returns_1D': returns_1d,
                    'Returns_1W': returns_1w,
                    'Returns_1M': returns_1m,
                    'RSI': rsi,
                    'Volatility': volatility,
                    'Market_Cap': info.get('marketCap', 0),
                    'PE_Ratio': info.get('trailingPE', 0),
                    'PB_Ratio': info.get('priceToBook', 0),
                    'Dividend_Yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                    'ROE': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
                    'Debt_to_Equity': info.get('debtToEquity', 0),
                    'Revenue_Growth': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0,
                    'Sector': info.get('sector', 'Unknown'),
                    'Industry': info.get('industry', 'Unknown')
                }
                
            except Exception as e:
                st.warning(f"Could not fetch data for {symbol}: {str(e)}")
                continue
                
            progress_bar.progress((i + 1) / len(symbols))
        
        status_text.text('Data fetching complete!')
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(data).T
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) > 0 else 50
    
    def apply_filters(self, df, filters):
        """Apply screening filters to the dataset"""
        filtered_df = df.copy()
        
        # Technical filters
        if filters['price_min'] > 0:
            filtered_df = filtered_df[filtered_df['Price'] >= filters['price_min']]
        if filters['price_max'] > 0:
            filtered_df = filtered_df[filtered_df['Price'] <= filters['price_max']]
            
        if filters['volume_min'] > 0:
            filtered_df = filtered_df[filtered_df['Volume_Current'] >= filters['volume_min']]
            
        if filters['rsi_min'] < filters['rsi_max']:
            filtered_df = filtered_df[
                (filtered_df['RSI'] >= filters['rsi_min']) & 
                (filtered_df['RSI'] <= filters['rsi_max'])
            ]
        
        # Fundamental filters
        if filters['pe_max'] > 0:
            filtered_df = filtered_df[
                (filtered_df['PE_Ratio'] > 0) & 
                (filtered_df['PE_Ratio'] <= filters['pe_max'])
            ]
            
        if filters['market_cap_min'] > 0:
            filtered_df = filtered_df[filtered_df['Market_Cap'] >= filters['market_cap_min']]
            
        if filters['dividend_yield_min'] > 0:
            filtered_df = filtered_df[filtered_df['Dividend_Yield'] >= filters['dividend_yield_min']]
            
        if filters['roe_min'] > 0:
            filtered_df = filtered_df[filtered_df['ROE'] >= filters['roe_min']]
        
        # Performance filters
        if filters['returns_1m_min'] != 0:
            filtered_df = filtered_df[filtered_df['Returns_1M'] >= filters['returns_1m_min']]
            
        return filtered_df

def main():
    st.set_page_config(
        page_title="Advanced Stock Screener",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Advanced Stock Screener")
    st.markdown("*Professional stock screening tool with technical and fundamental analysis*")
    
    # Initialize screener
    screener = StockScreener()
    
    # Sidebar filters
    st.sidebar.header("🎯 Screening Filters")
    
    # Preset strategies for beginners
    st.sidebar.subheader("🚀 Quick Start Presets")
    
    preset_strategies = {
        "Custom (Manual)": {
            "description": "Set your own criteria",
            "filters": {}
        },
        "Value Investing": {
            "description": "Undervalued companies with strong fundamentals",
            "filters": {
                "pe_max": 15.0,
                "price_min": 10.0,
                "market_cap_min": 1e9,
                "dividend_yield_min": 2.0,
                "roe_min": 15.0,
                "returns_1m_min": -10.0,
                "rsi_min": 20,
                "rsi_max": 50
            }
        },
        "Growth Stocks": {
            "description": "High-growth companies with momentum",
            "filters": {
                "returns_1m_min": 5.0,
                "market_cap_min": 5e9,
                "volume_min": 1000000,
                "rsi_min": 50,
                "rsi_max": 80,
                "pe_max": 35.0
            }
        },
        "Dividend Champions": {
            "description": "Reliable dividend-paying companies",
            "filters": {
                "dividend_yield_min": 3.0,
                "market_cap_min": 10e9,
                "pe_max": 25.0,
                "roe_min": 12.0,
                "price_min": 20.0
            }
        },
        "Momentum Trading": {
            "description": "Strong recent performance with high volume",
            "filters": {
                "returns_1m_min": 10.0,
                "volume_min": 2000000,
                "rsi_min": 60,
                "rsi_max": 85,
                "market_cap_min": 1e9
            }
        },
        "Oversold Bargains": {
            "description": "Potentially oversold quality companies",
            "filters": {
                "rsi_min": 10,
                "rsi_max": 30,
                "pe_max": 20.0,
                "market_cap_min": 5e9,
                "roe_min": 10.0
            }
        },
        "Blue Chip Stocks": {
            "description": "Large, stable, established companies",
            "filters": {
                "market_cap_min": 50e9,
                "dividend_yield_min": 1.0,
                "pe_max": 25.0,
                "price_min": 50.0
            }
        }
    }
    
    selected_preset = st.sidebar.selectbox(
        "Choose Strategy",
        list(preset_strategies.keys()),
        help="Select a preset strategy or choose 'Custom' to set your own filters"
    )
    
    # Display strategy description
    if selected_preset != "Custom (Manual)":
        st.sidebar.info(f"💡 **{selected_preset}**: {preset_strategies[selected_preset]['description']}")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Manual Filters")
    st.sidebar.markdown("*Customize the criteria below or use presets above*")
    
    # Initialize session state for filters
    filter_defaults = {
        "price_min": 0.0,
        "price_max": 0.0,
        "volume_min": 0,
        "rsi_min": 20,
        "rsi_max": 80,
        "market_cap_min": 0,
        "pe_max": 0.0,
        "dividend_yield_min": 0.0,
        "roe_min": 0.0,
        "returns_1m_min": 0.0
    }
    
    # Apply session state or defaults
    for key, default_value in filter_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Auto-apply preset values when preset is selected (but not Custom)
    if selected_preset != "Custom (Manual)":
        preset_filters = preset_strategies[selected_preset]['filters']
        for key, value in preset_filters.items():
            if key in filter_defaults:
                st.session_state[key] = value
    
    st.sidebar.subheader("Technical Filters")
    price_min = st.sidebar.number_input("Min Price ($)", min_value=0.0, value=float(st.session_state.get("price_min", 0.0)), step=1.0)
    price_max = st.sidebar.number_input("Max Price ($)", min_value=0.0, value=float(st.session_state.get("price_max", 0.0)), step=1.0)
    volume_min = st.sidebar.number_input("Min Volume", min_value=0, value=int(st.session_state.get("volume_min", 0)), step=100000)
    rsi_min = st.sidebar.slider("Min RSI", 0, 100, int(st.session_state.get("rsi_min", 20)))
    rsi_max = st.sidebar.slider("Max RSI", 0, 100, int(st.session_state.get("rsi_max", 80)))
    
    st.sidebar.subheader("Fundamental Filters")
    market_cap_options = [0, 1e9, 5e9, 10e9, 50e9]
    market_cap_value = st.session_state.get("market_cap_min", 0)
    market_cap_index = market_cap_options.index(market_cap_value) if market_cap_value in market_cap_options else 0
    market_cap_min = st.sidebar.selectbox(
        "Min Market Cap",
        market_cap_options,
        index=market_cap_index,
        format_func=lambda x: f"${x/1e9:.0f}B" if x > 0 else "No minimum"
    )
    pe_max = st.sidebar.number_input("Max P/E Ratio", min_value=0.0, value=float(st.session_state.get("pe_max", 0.0)), step=1.0)
    dividend_yield_min = st.sidebar.number_input("Min Dividend Yield (%)", min_value=0.0, value=float(st.session_state.get("dividend_yield_min", 0.0)), step=0.1)
    roe_min = st.sidebar.number_input("Min ROE (%)", min_value=0.0, value=float(st.session_state.get("roe_min", 0.0)), step=1.0)
    
    st.sidebar.subheader("Performance Filters")
    returns_1m_min = st.sidebar.number_input("Min 1M Return (%)", value=float(st.session_state.get("returns_1m_min", 0.0)), step=1.0)
    
    # Update session state with current values
    st.session_state.update({
        "price_min": price_min,
        "price_max": price_max,
        "volume_min": volume_min,
        "rsi_min": rsi_min,
        "rsi_max": rsi_max,
        "market_cap_min": market_cap_min,
        "pe_max": pe_max,
        "dividend_yield_min": dividend_yield_min,
        "roe_min": roe_min,
        "returns_1m_min": returns_1m_min
    })
    
    # Compile filters
    filters = {
        'price_min': price_min,
        'price_max': price_max,
        'volume_min': volume_min,
        'rsi_min': rsi_min,
        'rsi_max': rsi_max,
        'market_cap_min': market_cap_min,
        'pe_max': pe_max,
        'dividend_yield_min': dividend_yield_min,
        'roe_min': roe_min,
        'returns_1m_min': returns_1m_min
    }
    
    # Run screening button
    if st.sidebar.button("🔍 Run Screen", type="primary"):
        with st.spinner("Fetching and analyzing stock data..."):
            # Fetch data
            stock_data = screener.fetch_stock_data(screener.sp500_symbols[:30])  # Limit for demo
            
            if not stock_data.empty:
                # Apply filters
                filtered_data = screener.apply_filters(stock_data, filters)
                
                st.success(f"Found {len(filtered_data)} stocks matching your criteria out of {len(stock_data)} analyzed")
                
                # Display results
                if not filtered_data.empty:
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Stocks Found", len(filtered_data))
                    with col2:
                        avg_return = filtered_data['Returns_1M'].mean()
                        st.metric("Avg 1M Return", f"{avg_return:.1f}%")
                    with col3:
                        avg_pe = filtered_data[filtered_data['PE_Ratio'] > 0]['PE_Ratio'].mean()
                        st.metric("Avg P/E Ratio", f"{avg_pe:.1f}")
                    with col4:
                        avg_rsi = filtered_data['RSI'].mean()
                        st.metric("Avg RSI", f"{avg_rsi:.1f}")
                    
                    # Results table
                    st.subheader("📋 Screening Results")
                    display_cols = [
                        'Symbol', 'Price', 'Returns_1D', 'Returns_1W', 'Returns_1M',
                        'RSI', 'PE_Ratio', 'Market_Cap', 'Dividend_Yield', 'Sector'
                    ]
                    
                    display_df = filtered_data[display_cols].copy()
                    display_df['Market_Cap'] = display_df['Market_Cap'].apply(
                        lambda x: f"${x/1e9:.1f}B" if x > 1e9 else f"${x/1e6:.0f}M"
                    )
                    display_df = display_df.round(2)
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Returns distribution
                        fig = px.histogram(
                            filtered_data, 
                            x='Returns_1M',
                            title="1-Month Returns Distribution",
                            nbins=20
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # P/E vs Returns scatter
                        fig = px.scatter(
                            filtered_data[filtered_data['PE_Ratio'] > 0],
                            x='PE_Ratio',
                            y='Returns_1M',
                            color='Sector',
                            title="P/E Ratio vs Returns",
                            hover_data=['Symbol', 'Price']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Sector analysis
                    sector_analysis = filtered_data.groupby('Sector').agg({
                        'Symbol': 'count',
                        'Returns_1M': 'mean',
                        'PE_Ratio': lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0
                    }).round(2)
                    sector_analysis.columns = ['Count', 'Avg Return (%)', 'Avg P/E']
                    
                    st.subheader("📈 Sector Analysis")
                    st.dataframe(sector_analysis, use_container_width=True)
                    
                    # Download button
                    csv = filtered_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"stock_screen_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
                else:
                    st.warning("No stocks match your criteria. Try adjusting the filters.")
            else:
                st.error("Could not fetch stock data. Please try again later.")
    
    # Instructions
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Pro Tips:**")
    st.sidebar.markdown("• Use RSI 20-80 for momentum stocks")
    st.sidebar.markdown("• Low P/E + positive returns = value plays")
    st.sidebar.markdown("• High volume confirms price movements")
    st.sidebar.markdown("• Dividend yield for income strategies")
    
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
