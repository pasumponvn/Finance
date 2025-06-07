import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import io
from datetime import datetime

# --- Pre-defined Nifty 50 Symbols ---
NIFTY_50_SYMBOLS = [
    'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS',
    'BAJAJ-AUTO.NS', 'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'BPCL.NS',
    'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS',
    'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS',
    'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS',
    'INFY.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS',
    'LTIM.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS',
    'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'TATACONSUM.NS',
    'TATAMOTORS.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'TITAN.NS',
    'ULTRACEMCO.NS', 'UPL.NS', 'WIPRO.NS'
]

# --- Helper Functions ---

@st.cache_data
def get_nifty50_symbols():
    return NIFTY_50_SYMBOLS

@st.cache_data
def get_financial_data(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        income_stmt = ticker.financials
        balance_sheet = ticker.balance_sheet
        cash_flow = ticker.cashflow
        ticker_info = ticker.info
        return income_stmt, balance_sheet, cash_flow, ticker_info
    except Exception as e:
        st.warning(f"Could not fetch financial data for {ticker_symbol} from Yahoo Finance: {e}")
        st.info("Yahoo Finance typically provides 4-5 years of annual financial statements. Data for some tickers might be unavailable or incomplete.")
        return None, None, None, None

def calculate_fcff(income_stmt, cash_flow):
    # ... (rest of the FCFF calculation function remains the same) ...
    pass

def calculate_fcfe(income_stmt, cash_flow, balance_sheet):
    # ... (rest of the FCFE calculation function remains the same) ...
    pass

def calculate_cost_of_equity(ticker_info, risk_free_rate, market_risk_premium):
    # ... (rest of the Cost of Equity calculation function remains the same) ...
    pass

def calculate_dcf_model(fcf_history, forecast_years, initial_growth_rates, discount_rate, perpetual_growth_rate,
                       current_shares_outstanding, current_cash, total_debt, valuation_type="FCFF"):
    # ... (rest of the DCF model calculation function remains the same) ...
    pass

@st.cache_data
def get_nifty_index_value():
    try:
        nifty_ticker = yf.Ticker("^NSEI")
        nifty_data = nifty_ticker.history(period="1d")
        if not nifty_data.empty:
            return nifty_data['Close'].iloc[-1]
        return None
    except Exception as e:
        st.error(f"Could not fetch Nifty 50 index value (^NSEI) from Yahoo Finance: {e}")
        return None

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="Nifty 50 DCF Analysis")
st.title("Nifty 50 Discounted Cash Flow (DCF) Analysis")
st.info(...) # (rest of the info remains the same)

# Sidebar for controls
st.sidebar.header("Select Stock & DCF Assumptions")
nifty_symbols = get_nifty50_symbols()
if not nifty_symbols:
    st.error(...)
    st.stop()
selected_symbol_yf = st.sidebar.selectbox(...)
if selected_symbol_yf:
    # ... (rest of the sidebar controls remain the same) ...

    # --- Main Content Area ---
    st.header(...)
    income_stmt, balance_sheet, cash_flow, ticker_info = get_financial_data(selected_symbol_yf)
    if income_stmt is None or balance_sheet is None or cash_flow is None:
        st.error(...)
    else:
        try:
            current_market_price = yf.Ticker(selected_symbol_yf).history(period="1d")['Close'].iloc[-1]
            st.metric(...)
            balance_sheet_t_latest = balance_sheet.T.sort_index(ascending=True).iloc[-1]
            current_shares_outstanding = ticker_info.get('sharesOutstanding', balance_sheet_t_latest.get('Shares Outstanding', 0))
            # ... (rest of the data extraction remains the same) ...

            # --- FCFF and FCFE Calculations and Display ---
            # ... (FCFF and FCFE calculation and display remain the same) ...

            st.markdown("---")
            st.subheader("Comparison of Valuation Methods")

            nifty_index_value = get_nifty_index_value()

            labels = ['Current Market Price']
            values = [current_market_price]
            colors = ['blue']

            if intrinsic_value_per_share_fcff is not None:
                labels.append('FCFF Intrinsic Value')
                values.append(intrinsic_value_per_share_fcff)
                colors.append('green')

            if intrinsic_value_per_share_fcfe is not None:
                labels.append('FCFE Intrinsic Value')
                values.append(intrinsic_value_per_share_fcfe)
                colors.append('orange')

            if nifty_index_value is not None:
                labels.append('Nifty 50 Index Value')
                values.append(nifty_index_value)
                colors.append('red')

            if len(values) > 1:
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Scatter(x=labels, y=values, mode='lines+markers', name='Value', marker=dict(color=colors)))
                fig_compare.update_layout(title_text='Comparison of Values (Per Share / Index)',
                                        yaxis_title='Value (Rs.)',
                                        height=400, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_compare, use_container_width=True)
                st.info("Note: Comparing Nifty 50 Index value directly with individual stock per-share values is for contextual reference only, as their scales are different. The Nifty 50 value represents the index level, not a 'per share' value of any specific stock.")
            else:
                st.warning("Not enough valid values to create a comparison chart.")

        except Exception as e:
            st.error(...)
            st.exception(e)
