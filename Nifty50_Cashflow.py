import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import io
from datetime import datetime

# --- Helper Functions ---

@st.cache_data
def get_nifty50_symbols():
    """
    Fetches the current Nifty 50 stock symbols from NSE India website.
    """
    try:
        url = 'https://www.nseindia.com/content/indices/ind_nifty50list.csv'
        s = requests.get(url, timeout=10).content
        df = pd.read_csv(io.StringIO(s.decode('utf-8')))
        # Append .NS for Yahoo Finance compatibility for Indian stocks
        symbols = [s + '.NS' for s in df['Symbol'].tolist()]
        return symbols
    except Exception as e:
        st.error(f"Could not fetch Nifty 50 symbols from NSE: {e}")
        return []

@st.cache_data
def get_financial_data(ticker_symbol):
    """
    Fetches historical financial statements using yfinance.
    Returns income statement, balance sheet, cash flow statement.
    yfinance typically provides 4-5 years of annual financial statements.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        income_stmt = ticker.financials
        balance_sheet = ticker.balance_sheet
        cash_flow = ticker.cashflow
        return income_stmt, balance_sheet, cash_flow
    except Exception as e:
        st.warning(f"Could not fetch financial data for {ticker_symbol} from Yahoo Finance: {e}")
        st.info("Yahoo Finance typically provides 4-5 years of annual financial statements.")
        return None, None, None

def calculate_fcf(income_stmt, cash_flow, balance_sheet):
    """
    Calculates Free Cash Flow (FCF) from financial statements.
    Assumes standard format from yfinance.
    FCF = NOPAT + Depreciation - CAPEX - Change in Working Capital
    """
    if income_stmt is None or cash_flow is None or balance_sheet is None or income_stmt.empty or cash_flow.empty or balance_sheet.empty:
        return pd.Series(dtype='float64') # Return empty series if data is missing

    # Ensure dataframes are transposed and sorted by index (year) for consistent calculation
    income_stmt_t = income_stmt.T.sort_index(ascending=True)
    cash_flow_t = cash_flow.T.sort_index(ascending=True)
    balance_sheet_t = balance_sheet.T.sort_index(ascending=True)

    # Get common years for alignment - crucial for consistent calculations
    common_dates = income_stmt_t.index.intersection(cash_flow_t.index).intersection(balance_sheet_t.index)
    if common_dates.empty:
        st.warning("No common financial statement dates found across Income Statement, Balance Sheet, and Cash Flow.")
        return pd.Series(dtype='float64')

    income_stmt_t = income_stmt_t.loc[common_dates]
    cash_flow_t = cash_flow_t.loc[common_dates]
    balance_sheet_t = balance_sheet_t.loc[common_dates]

    fcf_series = pd.Series(dtype='float64')

    for date_idx in common_dates:
        try:
            current_income = income_stmt_t.loc[date_idx]
            current_cashflow = cash_flow_t.loc[date_idx]
            
            ebit = current_income.get('Operating Income', current_income.get('EBIT'))
            if pd.isna(ebit):
                continue

            tax_provision = current_income.get('Tax Provision')
            pretax_income = current_income.get('Pretax Income')
            tax_rate = tax_provision / pretax_income if not pd.isna(tax_provision) and not pd.isna(pretax_income) and pretax_income != 0 else 0.25
            if tax_rate < 0 or tax_rate > 1: tax_rate = 0.25

            nopat = ebit * (1 - tax_rate)

            capex = current_cashflow.get('Capital Expenditures', 0)
            capex = abs(capex)

            depreciation = current_cashflow.get('Depreciation And Amortization', current_income.get('Depreciation And Amortization', 0))
            depreciation = abs(depreciation)

            delta_wc = current_cashflow.get('Change In Working Capital', 0)
            
            fcf_val = nopat + depreciation - capex - delta_wc
            fcf_series.loc[date_idx] = fcf_val

        except Exception as e:
            fcf_series.loc[date_idx] = np.nan

    fcf_series = fcf_series.dropna()
    fcf_series.name = 'FCF'
    return fcf_series

def calculate_dcf(fcf_history, forecast_years, revenue_growth_rates, wacc, perpetual_growth_rate, current_shares_outstanding, current_cash, total_debt):
    """
    Performs DCF calculation.
    """
    if fcf_history.empty:
        st.error("Historical FCF data is not available or could not be calculated for DCF projections.")
        return None, None, None, None, None, None

    last_fcf = fcf_history.iloc[-1]
    
    projected_fcf = []
    
    if len(fcf_history) >= 2:
        historical_fcf_growth = fcf_history.pct_change().mean()
        if pd.isna(historical_fcf_growth) or historical_fcf_growth < -0.5 or historical_fcf_growth > 0.5:
            historical_fcf_growth = 0.05
    else:
        historical_fcf_growth = 0.05

    current_fcf_proj = last_fcf
    
    for i in range(forecast_years):
        if i < len(revenue_growth_rates):
            growth_rate = revenue_growth_rates[i]
        else:
            if forecast_years > len(revenue_growth_rates):
                taper_factor = (i - len(revenue_growth_rates) + 1) / (forecast_years - len(revenue_growth_rates) + 1)
                growth_rate = historical_fcf_growth * (1 - taper_factor) + perpetual_growth_rate * taper_factor
            else:
                growth_rate = historical_fcf_growth
            
            if growth_rate < perpetual_growth_rate:
                growth_rate = perpetual_growth_rate
        
        current_fcf_proj *= (1 + growth_rate)
        projected_fcf.append(current_fcf_proj)

    if wacc <= perpetual_growth_rate:
        st.error("WACC must be greater than Perpetual Growth Rate for DCF calculation to be valid (WACC > g).")
        return None, None, None, None, None, None
        
    terminal_fcf = projected_fcf[-1] * (1 + perpetual_growth_rate)
    terminal_value = terminal_fcf / (wacc - perpetual_growth_rate)
    
    discounted_fcf = []
    for i, fcf_val in enumerate(projected_fcf):
        discount_factor = (1 + wacc)**(i + 1)
        discounted_fcf.append(fcf_val / discount_factor)
    
    discounted_terminal_value = terminal_value / ((1 + wacc)**forecast_years)
    
    enterprise_value = sum(discounted_fcf) + discounted_terminal_value
    
    equity_value = enterprise_value + current_cash - total_debt
    
    intrinsic_value_per_share = equity_value / current_shares_outstanding if current_shares_outstanding > 0 else 0

    return intrinsic_value_per_share, enterprise_value, projected_fcf, discounted_fcf, terminal_value, discounted_terminal_value

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="Nifty 50 DCF Analysis")

st.title("Nifty 50 Discounted Cash Flow (DCF) Analysis")

st.info("""
This tool performs a simplified DCF analysis for Nifty 50 stocks using publicly available financial data from Yahoo Finance.
**Please note:** Yahoo Finance typically provides **4-5 years** of annual financial statements, not 10 years.
The analysis will use the available historical data for FCF calculation and projections.
This is for educational and illustrative purposes only and should not be used for actual investment decisions without professional advice and more rigorous modeling.
""")

# Sidebar for controls
st.sidebar.header("Select Stock & DCF Assumptions")

nifty_symbols = get_nifty50_symbols()
selected_symbol_yf = st.sidebar.selectbox(
    "Select a Nifty 50 Stock (with .NS suffix)",
    options=nifty_symbols,
    index=nifty_symbols.index('RELIANCE.NS') if 'RELIANCE.NS' in nifty_symbols else 0 # Default to Reliance
)

if selected_symbol_yf:
    st.sidebar.subheader("DCF Model Parameters")
    
    forecast_years = st.sidebar.slider("Forecast Period (Years)", 5, 10, 5)

    st.sidebar.markdown("**Annual Growth Rates for FCF (e.g., 0.10 for 10%)**")
    st.sidebar.markdown("*(These will be applied to the last historical FCF. Beyond these years, growth tapers to perpetual rate.)*")
    
    growth_rates_input = []
    num_initial_growth_years = min(forecast_years, 3)
    for i in range(num_initial_growth_years):
        growth = st.sidebar.number_input(f"Initial Growth Rate Year {i+1}", min_value=-0.10, max_value=0.5, value=0.10 - i*0.02, step=0.01, format="%.2f", key=f"gr_init_{i}")
        growth_rates_input.append(growth)
    
    wacc = st.sidebar.slider("Weighted Average Cost of Capital (WACC) - Discount Rate", 0.05, 0.20, 0.10, step=0.005, format="%.3f")
    perpetual_growth_rate = st.sidebar.slider("Perpetual Growth Rate (Terminal Value)", 0.00, 0.05, 0.025, step=0.001, format="%.3f")
    
    st.sidebar.subheader("Valuation Threshold")
    valuation_threshold_pct = st.sidebar.slider(
        "Threshold for Over/Underweight (%)",
        min_value=1.0, max_value=25.0, value=10.0, step=0.5, format="%.1f"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with ❤️ by Gemini")

    # --- Main Content Area ---
    st.header(f"DCF Analysis for {selected_symbol_yf.replace('.NS', '')}")

    income_stmt, balance_sheet, cash_flow = get_financial_data(selected_symbol_yf)

    if income_stmt is None or balance_sheet is None or cash_flow is None:
        st.error("Failed to retrieve sufficient financial data for DCF analysis. Please try another stock or check if data is available on Yahoo Finance.")
    else:
        try:
            current_market_price = yf.Ticker(selected_symbol_yf).history(period="1d")['Close'].iloc[-1]
            st.metric("Current Market Price", f"Rs. {current_market_price:,.2f}")

            balance_sheet_t_latest = balance_sheet.T.sort_index(ascending=True).iloc[-1]

            current_shares_outstanding = balance_sheet_t_latest.get('Shares Outstanding', balance_sheet_t_latest.get('Common Stock Shares Outstanding', 0))
            if current_shares_outstanding == 0:
                ticker_info = yf.Ticker(selected_symbol_yf).info
                current_shares_outstanding = ticker_info.get('sharesOutstanding', 1_000_000_000)
                st.warning(f"Could not find exact 'Shares Outstanding' in Balance Sheet. Using approximate: {current_shares_outstanding:,.0f}. This may affect per-share value.")

            current_cash = balance_sheet_t_latest.get('Cash And Cash Equivalents', 0)
            total_debt = balance_sheet_t_latest.get('Total Debt', 0)

            st.subheader("Historical Free Cash Flow (FCF)")
            historical_fcf = calculate_fcf(income_stmt, cash_flow, balance_sheet)
            
            if not historical_fcf.empty:
                st.write(f"Showing {len(historical_fcf)} years of historical FCF data available from Yahoo Finance (typically 4-5 years):")
                
                historical_fcf.index = historical_fcf.index.map(lambda x: x.year)
                
                st.dataframe(historical_fcf.to_frame(name='FCF').T.style.format("{:,.0f}"))
                
                # --- Historical FCF Graph ---
                fig_hist_fcf = go.Figure(data=[go.Bar(x=historical_fcf.index.astype(str), y=historical_fcf.values)])
                fig_hist_fcf.update_layout(title_text='Historical Free Cash Flow (FCF)', xaxis_title='Year', yaxis_title='FCF (Rs.)',
                                height=300, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_hist_fcf, use_container_width=True)

                # --- Perform DCF Calculation ---
                intrinsic_value_per_share, enterprise_value, projected_fcf, discounted_fcf, terminal_value, discounted_terminal_value = calculate_dcf(
                    historical_fcf,
                    forecast_years,
                    growth_rates_input,
                    wacc,
                    perpetual_growth_rate,
                    current_shares_outstanding,
                    current_cash,
                    total_debt
                )

                if intrinsic_value_per_share is not None:
                    st.subheader("DCF Valuation Results")
                    col_val1, col_val2, col_val3 = st.columns(3)
                    col_val1.metric("Intrinsic Value per Share", f"Rs. {intrinsic_value_per_share:,.2f}")
                    col_val2.metric("Current Market Price", f"Rs. {current_market_price:,.2f}")
                    
                    upside_downside = ((intrinsic_value_per_share - current_market_price) / current_market_price) * 100
                    col_val3.metric("Upside/Downside (%)", f"{upside_downside:,.2f}%")

                    # --- Suggestion based on valuation ---
                    st.subheader("Valuation Recommendation")
                    if upside_downside >= valuation_threshold_pct:
                        st.success(f"**OVERWEIGHT (Undervalued):** The intrinsic value is {upside_downside:.2f}% higher than the current market price, exceeding your {valuation_threshold_pct:.1f}% threshold. This suggests the stock might be undervalued.")
                    elif upside_downside <= -valuation_threshold_pct:
                        st.warning(f"**UNDERWEIGHT (Overvalued):** The intrinsic value is {abs(upside_downside):.2f}% lower than the current market price, exceeding your {valuation_threshold_pct:.1f}% threshold. This suggests the stock might be overvalued.")
                    else:
                        st.info(f"**FAIRLY VALUED:** The intrinsic value is within +/- {valuation_threshold_pct:.1f}% of the current market price ({upside_downside:.2f}% difference). This suggests the stock is fairly valued.")


                    st.markdown(f"**Enterprise Value:** Rs. {enterprise_value:,.0f}")
                    st.markdown(f"**Equity Value:** Rs. {equity_value:,.0f}") # Corrected to use equity_value directly

                    # --- Detailed Projections Table ---
                    st.subheader("Projected Free Cash Flow (FCF) Details")
                    
                    proj_fcf_years = [f"Proj {i+1}" for i in range(forecast_years)]
                    proj_fcf_df = pd.DataFrame({
                        'Year': proj_fcf_years,
                        'Projected FCF (Rs.)': projected_fcf,
                        'Discounted FCF (Rs.)': discounted_fcf
                    })
                    st.dataframe(proj_fcf_df.set_index('Year').style.format("{:,.0f}"))

                    st.markdown(f"**Terminal Value (at end of forecast period):** Rs. {terminal_value:,.0f}")
                    st.markdown(f"**Discounted Terminal Value:** Rs. {discounted_terminal_value:,.0f}")

                    # --- Projected FCF Graph ---
                    fig_proj_fcf = go.Figure(data=[go.Bar(x=proj_fcf_df['Year'], y=proj_fcf_df['Projected FCF (Rs.)'], name='Projected FCF')])
                    fig_proj_fcf.update_layout(title_text='Projected Free Cash Flow (FCF)', xaxis_title='Year', yaxis_title='FCF (Rs.)',
                                            height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_proj_fcf, use_container_width=True)

                    # --- Discounted FCF Graph ---
                    fig_disc_fcf = go.Figure(data=[go.Bar(x=proj_fcf_df['Year'], y=proj_fcf_df['Discounted FCF (Rs.)'], name='Discounted FCF', marker_color='lightseagreen')])
                    fig_disc_fcf.update_layout(title_text='Discounted Free Cash Flow (FCF)', xaxis_title='Year', yaxis_title='Discounted FCF (Rs.)',
                                            height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_disc_fcf, use_container_width=True)

                else:
                    st.warning("DCF calculation could not be completed with provided parameters. Please check assumptions.")
            else:
                st.warning("Historical FCF could not be calculated. Please check financial data availability for the selected stock. Yahoo Finance usually provides 4-5 years of data.")

        except Exception as e:
            st.error(f"An error occurred during DCF analysis: {e}. Please ensure the selected stock has complete financial data on Yahoo Finance.")
            st.exception(e) # Display full traceback for debugging
