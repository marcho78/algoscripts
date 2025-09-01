#!/usr/bin/env python3
"""
Crypto Portfolio Tracker
Real-time cryptocurrency portfolio tracking with P&L analysis
Very marketable - crypto is hot and this provides immediate value
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import time

class CryptoPortfolioTracker:
    def __init__(self):
        self.portfolio = {}
        self.base_url = "https://api.coingecko.com/api/v3"
        
    def get_crypto_price(self, crypto_id):
        """Get current price from CoinGecko API"""
        try:
            url = f"{self.base_url}/simple/price"
            params = {
                'ids': crypto_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_market_cap': 'true'
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            if crypto_id in data:
                return {
                    'price': data[crypto_id]['usd'],
                    'change_24h': data[crypto_id].get('usd_24h_change', 0),
                    'market_cap': data[crypto_id].get('market_cap', 0)
                }
        except Exception as e:
            st.error(f"Error fetching price for {crypto_id}: {e}")
        
        return None
    
    def get_multiple_prices(self, crypto_ids):
        """Get prices for multiple cryptocurrencies"""
        try:
            url = f"{self.base_url}/simple/price"
            params = {
                'ids': ','.join(crypto_ids),
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_market_cap': 'true'
            }
            response = requests.get(url, params=params)
            return response.json()
        except Exception as e:
            st.error(f"Error fetching prices: {e}")
            return {}
    
    def get_historical_data(self, crypto_id, days=30):
        """Get historical price data"""
        try:
            url = f"{self.base_url}/coins/{crypto_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily' if days > 90 else 'hourly'
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'prices' in data:
                prices = data['prices']
                df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('date', inplace=True)
                return df
        except Exception as e:
            st.error(f"Error fetching historical data for {crypto_id}: {e}")
        
        return pd.DataFrame()
    
    def search_cryptocurrencies(self, query):
        """Search for cryptocurrencies"""
        try:
            url = f"{self.base_url}/search"
            params = {'query': query}
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'coins' in data:
                return [(coin['id'], coin['name'], coin['symbol']) for coin in data['coins'][:10]]
        except Exception as e:
            st.error(f"Error searching cryptocurrencies: {e}")
        
        return []
    
    def calculate_portfolio_metrics(self, holdings):
        """Calculate portfolio metrics"""
        total_value = sum([holding['current_value'] for holding in holdings])
        total_invested = sum([holding['invested'] for holding in holdings])
        total_pnl = total_value - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0
        
        return {
            'total_value': total_value,
            'total_invested': total_invested,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct
        }

def main():
    st.set_page_config(
        page_title="Crypto Portfolio Tracker",
        page_icon="₿",
        layout="wide"
    )
    
    st.title("₿ Crypto Portfolio Tracker")
    st.markdown("*Real-time cryptocurrency portfolio tracking with P&L analysis*")
    
    tracker = CryptoPortfolioTracker()
    
    # Initialize session state
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = []
    
    # Sidebar - Portfolio Management
    st.sidebar.header("💼 Portfolio Management")
    
    # Add new position
    st.sidebar.subheader("Add Position")
    
    # Search for cryptocurrency
    search_query = st.sidebar.text_input("Search Cryptocurrency", placeholder="Bitcoin, Ethereum, etc.")
    
    if search_query:
        search_results = tracker.search_cryptocurrencies(search_query)
        if search_results:
            crypto_options = {f"{name} ({symbol.upper()})": crypto_id for crypto_id, name, symbol in search_results}
            selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", list(crypto_options.keys()))
            crypto_id = crypto_options[selected_crypto]
            
            # Position details
            quantity = st.sidebar.number_input("Quantity", min_value=0.0, value=1.0, step=0.1, format="%.8f")
            avg_price = st.sidebar.number_input("Average Buy Price ($)", min_value=0.0, value=100.0, step=0.01)
            
            if st.sidebar.button("Add to Portfolio"):
                # Check if crypto already exists in portfolio
                existing_position = None
                for i, pos in enumerate(st.session_state.portfolio):
                    if pos['crypto_id'] == crypto_id:
                        existing_position = i
                        break
                
                if existing_position is not None:
                    # Update existing position
                    old_pos = st.session_state.portfolio[existing_position]
                    total_quantity = old_pos['quantity'] + quantity
                    total_invested = old_pos['invested'] + (quantity * avg_price)
                    new_avg_price = total_invested / total_quantity
                    
                    st.session_state.portfolio[existing_position] = {
                        'crypto_id': crypto_id,
                        'name': selected_crypto.split(' (')[0],
                        'symbol': selected_crypto.split('(')[1].replace(')', ''),
                        'quantity': total_quantity,
                        'avg_price': new_avg_price,
                        'invested': total_invested
                    }
                else:
                    # Add new position
                    st.session_state.portfolio.append({
                        'crypto_id': crypto_id,
                        'name': selected_crypto.split(' (')[0],
                        'symbol': selected_crypto.split('(')[1].replace(')', ''),
                        'quantity': quantity,
                        'avg_price': avg_price,
                        'invested': quantity * avg_price
                    })
                
                st.sidebar.success(f"Added {quantity} {selected_crypto} to portfolio!")
                time.sleep(1)
                st.rerun()
    
    # Portfolio display
    if st.session_state.portfolio:
        # Fetch current prices
        crypto_ids = [pos['crypto_id'] for pos in st.session_state.portfolio]
        current_prices = tracker.get_multiple_prices(crypto_ids)
        
        # Calculate portfolio values
        holdings = []
        for position in st.session_state.portfolio:
            crypto_id = position['crypto_id']
            if crypto_id in current_prices:
                current_price = current_prices[crypto_id]['usd']
                current_value = position['quantity'] * current_price
                pnl = current_value - position['invested']
                pnl_pct = (pnl / position['invested'] * 100) if position['invested'] > 0 else 0
                change_24h = current_prices[crypto_id].get('usd_24h_change', 0)
                
                holdings.append({
                    'crypto_id': crypto_id,
                    'name': position['name'],
                    'symbol': position['symbol'],
                    'quantity': position['quantity'],
                    'avg_price': position['avg_price'],
                    'current_price': current_price,
                    'invested': position['invested'],
                    'current_value': current_value,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'change_24h': change_24h
                })
        
        if holdings:
            # Portfolio summary
            portfolio_metrics = tracker.calculate_portfolio_metrics(holdings)
            
            st.subheader("📊 Portfolio Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Value", f"${portfolio_metrics['total_value']:,.2f}")
            with col2:
                st.metric("Total Invested", f"${portfolio_metrics['total_invested']:,.2f}")
            with col3:
                pnl_color = "normal" if portfolio_metrics['total_pnl'] >= 0 else "inverse"
                st.metric("Total P&L", f"${portfolio_metrics['total_pnl']:,.2f}", 
                         f"{portfolio_metrics['total_pnl_pct']:+.2f}%")
            with col4:
                st.metric("Number of Holdings", len(holdings))
            
            # Holdings table
            st.subheader("💰 Current Holdings")
            
            # Create display DataFrame
            df = pd.DataFrame(holdings)
            display_df = df.copy()
            
            # Format columns for display
            display_df['Quantity'] = display_df['quantity'].apply(lambda x: f"{x:.8f}".rstrip('0').rstrip('.'))
            display_df['Avg Price'] = display_df['avg_price'].apply(lambda x: f"${x:,.2f}")
            display_df['Current Price'] = display_df['current_price'].apply(lambda x: f"${x:,.2f}")
            display_df['Invested'] = display_df['invested'].apply(lambda x: f"${x:,.2f}")
            display_df['Current Value'] = display_df['current_value'].apply(lambda x: f"${x:,.2f}")
            display_df['P&L'] = display_df.apply(lambda row: f"${row['pnl']:,.2f} ({row['pnl_pct']:+.2f}%)", axis=1)
            display_df['24h Change'] = display_df['change_24h'].apply(lambda x: f"{x:+.2f}%")
            
            # Select columns for display
            display_cols = ['name', 'symbol', 'Quantity', 'Avg Price', 'Current Price', 
                          'Invested', 'Current Value', 'P&L', '24h Change']
            
            st.dataframe(
                display_df[display_cols].rename(columns={'name': 'Name', 'symbol': 'Symbol'}),
                use_container_width=True,
                hide_index=True
            )
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Portfolio allocation pie chart
                fig_pie = px.pie(
                    values=df['current_value'],
                    names=df['symbol'],
                    title="Portfolio Allocation by Value"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # P&L bar chart
                colors = ['green' if pnl >= 0 else 'red' for pnl in df['pnl']]
                
                fig_bar = go.Figure(data=go.Bar(
                    x=df['symbol'],
                    y=df['pnl'],
                    marker_color=colors,
                    text=df['pnl_pct'].apply(lambda x: f"{x:+.1f}%"),
                    textposition='outside'
                ))
                
                fig_bar.update_layout(
                    title="P&L by Asset",
                    xaxis_title="Cryptocurrency",
                    yaxis_title="P&L ($)",
                    showlegend=False
                )
                
                fig_bar.add_hline(y=0, line_dash="dash", line_color="gray")
                
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Historical performance
            st.subheader("📈 Historical Performance")
            
            selected_crypto = st.selectbox(
                "Select crypto for historical view",
                [(holding['crypto_id'], holding['name']) for holding in holdings],
                format_func=lambda x: x[1]
            )
            
            time_period = st.selectbox("Time Period", [7, 30, 90, 365], format_func=lambda x: f"{x} days")
            
            if selected_crypto:
                crypto_id = selected_crypto[0]
                historical_data = tracker.get_historical_data(crypto_id, time_period)
                
                if not historical_data.empty:
                    fig_hist = go.Figure()
                    
                    fig_hist.add_trace(go.Scatter(
                        x=historical_data.index,
                        y=historical_data['price'],
                        mode='lines',
                        name=selected_crypto[1],
                        line=dict(width=2)
                    ))
                    
                    # Add buy price line
                    holding_info = next(h for h in holdings if h['crypto_id'] == crypto_id)
                    fig_hist.add_hline(
                        y=holding_info['avg_price'],
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Avg Buy Price: ${holding_info['avg_price']:.2f}"
                    )
                    
                    fig_hist.update_layout(
                        title=f"{selected_crypto[1]} Price History ({time_period} days)",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
            
            # Portfolio management
            st.subheader("🛠️ Portfolio Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Refresh Prices"):
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Clear Portfolio", type="secondary"):
                    st.session_state.portfolio = []
                    st.rerun()
            # Create export data
            export_df = df[['name', 'symbol', 'quantity', 'avg_price', 'current_price', 
                           'invested', 'current_value', 'pnl', 'pnl_pct', 'change_24h']]
            
            # Add summary row
            summary_row = pd.DataFrame([{
                'name': 'TOTAL PORTFOLIO',
                'symbol': '',
                'quantity': '',
                'avg_price': '',
                'current_price': '',
                'invested': portfolio_metrics['total_invested'],
                'current_value': portfolio_metrics['total_value'],
                'pnl': portfolio_metrics['total_pnl'],
                'pnl_pct': portfolio_metrics['total_pnl_pct'],
                'change_24h': ''
            }])
            
            export_df = pd.concat([export_df, summary_row], ignore_index=True)
            
            # Download button
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📄 Download Portfolio CSV",
                data=csv,
                file_name=f"crypto_portfolio_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
            
        else:
            st.warning("Unable to fetch current prices. Please try again later.")
    
    else:
        st.info("👈 Add your first cryptocurrency position using the sidebar!")
        
        # Sample portfolio for demo
        with st.expander("🚀 Try with Sample Portfolio"):
            if st.button("Load Sample Portfolio"):
                st.session_state.portfolio = [
                    {
                        'crypto_id': 'bitcoin',
                        'name': 'Bitcoin',
                        'symbol': 'BTC',
                        'quantity': 0.5,
                        'avg_price': 45000,
                        'invested': 22500
                    },
                    {
                        'crypto_id': 'ethereum',
                        'name': 'Ethereum',
                        'symbol': 'ETH',
                        'quantity': 5.0,
                        'avg_price': 2800,
                        'invested': 14000
                    },
                    {
                        'crypto_id': 'cardano',
                        'name': 'Cardano',
                        'symbol': 'ADA',
                        'quantity': 10000,
                        'avg_price': 0.5,
                        'invested': 5000
                    }
                ]
                st.rerun()
    
    # Market overview
    if st.checkbox("📊 Show Market Overview"):
        st.subheader("🌍 Top Cryptocurrencies")
        
        try:
            # Get top 10 cryptocurrencies
            url = f"{tracker.base_url}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': 10,
                'page': 1
            }
            response = requests.get(url, params=params)
            market_data = response.json()
            
            if market_data:
                market_df = pd.DataFrame([{
                    'Rank': coin['market_cap_rank'],
                    'Name': coin['name'],
                    'Symbol': coin['symbol'].upper(),
                    'Price': f"${coin['current_price']:,.2f}",
                    'Market Cap': f"${coin['market_cap']:,}",
                    '24h Change': f"{coin['price_change_percentage_24h']:+.2f}%"
                } for coin in market_data])
                
                st.dataframe(market_df, use_container_width=True, hide_index=True)
        
        except Exception as e:
            st.error(f"Unable to fetch market data: {e}")
    
    # Instructions
    with st.expander("📖 How to Use This Tool"):
        st.markdown("""
        ### Crypto Portfolio Tracker Guide
        
        1. **Add Positions**: Search for cryptocurrencies and add them to your portfolio
        2. **Track Performance**: View real-time P&L and portfolio metrics
        3. **Analyze History**: Review price history and performance over time
        4. **Export Data**: Download your portfolio data for record keeping
        
        ### Features:
        - **Real-time Prices**: Live data from CoinGecko API
        - **Portfolio Metrics**: Total value, P&L, and percentage returns
        - **Visual Analytics**: Pie charts and bar graphs for easy analysis
        - **Historical Data**: Price charts to track performance over time
        - **Market Overview**: Top cryptocurrencies by market cap
        
        ### Tips:
        - Regularly update your positions as you buy/sell
        - Use the historical charts to time your entries/exits
        - Export your data for tax reporting
        - Monitor 24h changes for market sentiment
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
