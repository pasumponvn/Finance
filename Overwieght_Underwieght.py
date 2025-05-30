import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title='NSE Nifty 50 Overweight/Underweight Prediction (DCM Model)', layout='wide')
st.title('NSE Nifty 50 Overweight/Underweight Prediction (DCM Model)')

st.write("""
Select multiple Nifty 50 stocks below to predict if each is **Overweight** or **Underweight** based on a Discounted Cashflow Model (DCM) approximation using last 1 year of daily price data.
A comparison chart will be shown for all selected stocks.
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

stocks = st.multiselect(
    'Select Nifty 50 stocks to compare (up to 10 recommended for faster results):',
    options=nifty50_stocks,
    default=['RELIANCE', 'HDFCBANK', 'INFY'],
    max_selections=15
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
    return symbol.strip().upper() + '.NS'

results = []

if submit and stocks:
    progress = st.progress(0)
    for i, stock_symbol in enumerate(stocks):
        try:
            end_date = datetime.today()
            start_date = end_date - timedelta(days=365)
            yf_symbol = get_nse_symbol(stock_symbol)
            data = yf.download(yf_symbol, start=start_date, end=end_date, progress=False)
            if data.empty or 'Close' not in data:
                results.append({
                    "symbol": stock_symbol,
                    "error": "No data found",
                })
            else:
                prices = data['Close'].dropna().values
                fair_price, cagr, discount_rate = dcm_fair_price(prices)
                current_price = prices[-1]
                status = "Overweight" if current_price > fair_price else "Underweight"
                results.append({
                    "symbol": stock_symbol,
                    "current_price": current_price,
                    "fair_price": fair_price,
                    "cagr": cagr,
                    "discount_rate": discount_rate,
                    "status": status,
                    "prices_df": data[['Close']]
                })
        except Exception as e:
            results.append({
                "symbol": stock_symbol,
                "error": str(e),
            })
        progress.progress((i+1)/len(stocks))
    progress.empty()

    # Show summary table
    st.markdown("### Summary Table")
    summary_table = []
    for res in results:
        if 'error' in res:
            summary_table.append([res['symbol'], "Error", "", "", "", "", ""])
        else:
            summary_table.append([
                res['symbol'],
                f"₹{res['current_price']:.2f}",
                f"₹{res['fair_price']:.2f}",
                f"{res['cagr']*100:.2f}%",
                f"{res['discount_rate']*100:.2f}%",
                res['status'],
                "✅" if res['status']=="Overweight" else "⚠️"
            ])
    st.dataframe(
        pd.DataFrame(
            summary_table,
            columns=["Symbol", "Current Price", "DCM Fair Price", "1Y CAGR", "Discount Rate", "Status", "Flag"]
        ).set_index("Symbol"),
        use_container_width=True
    )

    # Comparison plot
    st.markdown("### Comparison Chart: Current Price vs DCM Fair Price")
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        fig, ax = plt.subplots(figsize=(min(15, 2+2*len(valid_results)), 6))
        symbols = [r['symbol'] for r in valid_results]
        current_prices = [r['current_price'] for r in valid_results]
        fair_prices = [r['fair_price'] for r in valid_results]
        bar_width = 0.35
        indices = np.arange(len(symbols))
        rects1 = ax.bar(indices - bar_width/2, current_prices, bar_width, label='Current Price')
        rects2 = ax.bar(indices + bar_width/2, fair_prices, bar_width, label='DCM Fair Price')
        ax.set_xticks(indices)
        ax.set_xticklabels(symbols, rotation=45, ha='right')
        ax.set_ylabel("Price (INR)")
        ax.set_title("Nifty 50 Stock Comparison: Current vs DCM Fair Price")
        ax.legend()
        st.pyplot(fig)

    # Individual stock charts
    st.markdown("### Individual Stock Price Charts")
    for res in valid_results:
        st.subheader(f"{res['symbol']} ({res['status']})")
        st.write(f"**Current Price:** ₹{res['current_price']:.2f} | **DCM Fair Price:** ₹{res['fair_price']:.2f} | **1Y CAGR:** {res['cagr']*100:.2f}%")
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        res['prices_df']['Close'].plot(ax=ax2, label='Historical Close Price')
        ax2.axhline(res['fair_price'], color='green', linestyle='--', label='DCM Fair Price')
        ax2.axhline(res['current_price'], color='blue', linestyle=':', label='Current Price')
        ax2.set_title(f"{res['symbol']} (NSE) Price vs DCM Fair Value")
        ax2.set_ylabel("Price (INR)")
        ax2.legend()
        st.pyplot(fig2)
