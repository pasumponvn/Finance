import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title='NSE Nifty 50 Overweight/Underweight Prediction (DCM Model)', layout='centered')
st.title('NSE Nifty 50 Overweight/Underweight Prediction (DCM Model)')

st.write("""
Select one Nifty 50 stock below to predict if it is **Overweight** or **Underweight** based on a Discounted Cashflow Model (DCM) approximation using last 1 year of daily price data.
""")

# Full list of Nifty 50 stocks as of May 2024
nifty50_stocks = [
    'ADANIPORTS', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE',
    'BAJAJFINSV', 'BPCL', 'BHARTIARTL', 'BRITANNIA', 'CIPLA',
    'COALINDIA', 'DIVISLAB', 'DRREDDY', 'EICHERMOT', 'GRASIM',
    'HCLTECH', 'HDFCBANK', 'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO',
    'HINDUNILVR', 'ICICIBANK', 'ITC', 'INDUSINDBK', 'INFY',
    'JSWSTEEL', 'KOTAKBANK', 'LT', 'LTIM', 'M&M',
    'MARUTI', 'NESTLEIND', 'NTPC', 'ONGC', 'POWERGRID',
    'RELIANCE', 'SBILIFE', 'SBIN', 'SHRIRAMFIN', 'SUNPHARMA',
    'TATACONSUM', 'TATAMOTORS', 'TATASTEEL', 'TCS', 'TECHM',
    'TITAN', 'ULTRACEMCO', 'UPL', 'WIPRO', 'HDFCAMC'
]

stock = st.selectbox(
    'Select a Nifty 50 stock to analyze:',
    options=nifty50_stocks,
    index=nifty50_stocks.index('RELIANCE')
)
submit = st.button('Submit')

def dcm_fair_price(prices, risk_free_rate=0.065, risk_premium=0.07):
    n = len(prices)
    if n < 2:
        return np.nan, np.nan, np.nan
    start_price = prices[0]
    end_price = prices[-1]
    cagr = (end_price/start_price)**(1/(n/252)) - 1
    expected_return = cagr if not np.isnan(cagr) else 0.10
    discount_rate = risk_free_rate + risk_premium
    fair_price = end_price / (1 + discount_rate)
    return fair_price, expected_return, discount_rate

def get_nse_symbol(symbol):
    #return symbol.strip().upper() + '.NS'
    return symbol.strip().upper().replace("-", "_") + ".NS"

if submit and stock:
    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365)
        yf_symbol = get_nse_symbol(stock)
        data = yf.download(yf_symbol, start=start_date, end=end_date, progress=False)
        if data.empty or 'Close' not in data:
            st.error('No data found for this stock.')
        else:
            prices = data['Close'].dropna().values
            fair_price, cagr, discount_rate = dcm_fair_price(prices)
            current_price = prices[-1]
            status = "Overweight 🚀" if current_price > fair_price else "Underweight ⚠️"
            st.subheader(f"{stock} Prediction: **{status}**")
            st.write(f"**Current Price:** ₹{current_price:.2f}")
            st.write(f"**DCM Fair Price (approx):** ₹{fair_price:.2f}")
            st.write(f"**1Y CAGR:** {cagr*100:.2f}% | **Discount Rate:** {discount_rate*100:.2f}%")

            # Plot
            fig, ax = plt.subplots(figsize=(9, 4))
            data['Close'].plot(ax=ax, label='Historical Close Price')
            ax.axhline(fair_price, color='green', linestyle='--', label='DCM Fair Price')
            ax.axhline(current_price, color='blue', linestyle=':', label='Current Price')
            ax.set_title(f"{stock} (NSE) Price vs DCM Fair Value")
            ax.set_ylabel("Price (INR)")
            ax.legend()
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error fetching data or calculation: {e}")
