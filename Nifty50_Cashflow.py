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
        st.info("Please ensure you have an active internet connection and that NSE website is accessible.")
        return []

@st.cache_data
def get_financial_data(ticker_symbol):
    """
    Fetches historical financial statements and ticker info using yfinance.
    Returns income statement, balance sheet, cash flow statement, and ticker_info.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        income_stmt = ticker.financials
        balance_sheet = ticker.balance_sheet
        cash_flow = ticker.cashflow
        ticker_info = ticker.info # Get additional info including beta, shares outstanding
        return income_stmt, balance_sheet, cash_flow, ticker_info
    except Exception as e:
        st.warning(f"Could not fetch financial data for {ticker_symbol} from Yahoo Finance: {e}")
        st.info("Yahoo Finance typically provides 4-5 years of annual financial statements. Data for some tickers might be unavailable or incomplete.")
        return None, None, None, None

def calculate_fcff(income_stmt, cash_flow):
    """
    Calculates Free Cash Flow to Firm (FCFF) from financial statements.
    FCFF = NOPAT + Depreciation & Amortization - CAPEX - Change in Working Capital
    Assumes standard format from yfinance.
    """
    if income_stmt is None or cash_flow is None or income_stmt.empty or cash_flow.empty:
        return pd.Series(dtype='float64')

    income_stmt_t = income_stmt.T.sort_index(ascending=True)
    cash_flow_t = cash_flow.T.sort_index(ascending=True)

    common_dates = income_stmt_t.index.intersection(cash_flow_t.index)
    if common_dates.empty:
        # st.warning("No common financial statement dates found for FCFF across Income Statement and Cash Flow.")
        return pd.Series(dtype='float64')

    income_stmt_t = income_stmt_t.loc[common_dates]
    cash_flow_t = cash_flow_t.loc[common_dates]

    fcff_series = pd.Series(dtype='float64')

    for date_idx in common_dates:
        try:
            current_income = income_stmt_t.loc[date_idx]
            current_cashflow = cash_flow_t.loc[date_idx]
            
            # Use 'Operating Income' or 'EBIT' for EBIT
            ebit = current_income.get('Operating Income', current_income.get('EBIT'))
            if pd.isna(ebit): continue

            # Get Tax Rate: Tax Provision / Pretax Income
            tax_provision = current_income.get('Tax Provision')
            pretax_income = current_income.get('Pretax Income')
            tax_rate = tax_provision / pretax_income if not pd.isna(tax_provision) and not pd.isna(pretax_income) and pretax_income != 0 else 0.25
            if tax_rate < 0 or tax_rate > 1: tax_rate = 0.25 # Sanity check for tax rate

            nopat = ebit * (1 - tax_rate)

            # Capital Expenditures (negative in cash flow statement, take absolute)
            capex = current_cashflow.get('Capital Expenditures', 0)
            capex = abs(capex)

            # Depreciation & Amortization (often positive in cash flow for add-back)
            depreciation = current_cashflow.get('Depreciation And Amortization', current_income.get('Depreciation And Amortization', 0))
            depreciation = abs(depreciation) # Should be positive as it's added back

            # Change in Working Capital (can be positive or negative)
            delta_wc = current_cashflow.get('Change In Working Capital', 0)
            
            fcff_val = nopat + depreciation - capex - delta_wc
            fcff_series.loc[date_idx] = fcff_val

        except Exception as e:
            fcff_series.loc[date_idx] = np.nan

    fcff_series = fcff_series.dropna()
    fcff_series.name = 'FCFF'
    return fcff_series

def calculate_fcfe(income_stmt, cash_flow, balance_sheet):
    """
    Calculates Free Cash Flow to Equity (FCFE) from financial statements.
    FCFE = Net Income + D&A - CAPEX - Change in Non-Cash Working Capital + Net Borrowing
    """
    if income_stmt is None or cash_flow is None or balance_sheet is None or income_stmt.empty or cash_flow.empty or balance_sheet.empty:
        return pd.Series(dtype='float64')

    income_stmt_t = income_stmt.T.sort_index(ascending=True)
    cash_flow_t = cash_flow.T.sort_index(ascending=True)
    balance_sheet_t = balance_sheet.T.sort_index(ascending=True)

    common_dates = income_stmt_t.index.intersection(cash_flow_t.index).intersection(balance_sheet_t.index)
    if common_dates.empty:
        # st.warning("No common financial statement dates found for FCFE across Income Statement, Balance Sheet, and Cash Flow.")
        return pd.Series(dtype='float64')

    income_stmt_t = income_stmt_t.loc[common_dates]
    cash_flow_t = cash_flow_t.loc[common_dates]
    balance_sheet_t = balance_sheet_t.loc[common_dates]

    fcfe_series = pd.Series(dtype='float64')

    # To calculate Net Borrowing, we need previous year's total debt
    if len(balance_sheet_t) < 2:
        # st.warning("Insufficient balance sheet history to calculate Net Borrowing for FCFE.")
        return pd.Series(dtype='float64')

    total_debt_series = balance_sheet_t.get('Total Debt')
    if total_debt_series is None:
        # st.warning("Could not find 'Total Debt' in balance sheet for FCFE calculation.")
        return pd.Series(dtype='float64')

    net_borrowing_series = total_debt_series.diff()

    for i, date_idx in enumerate(common_dates):
        if i == 0: # Cannot calculate change in working capital or net borrowing for the first year
            continue

        try:
            current_income = income_stmt_t.loc[date_idx]
            current_cashflow = cash_flow_t.loc[date_idx]
            
            net_income = current_income.get('Net Income', np.nan)
            if pd.isna(net_income): continue

            depreciation = current_cashflow.get('Depreciation And Amortization', current_income.get('Depreciation And Amortization', 0))
            depreciation = abs(depreciation)

            capex = current_cashflow.get('Capital Expenditures', 0)
            capex = abs(capex)

            delta_wc = current_cashflow.get('Change In Working Capital', 0)

            # Net Borrowing: Prioritize Cash Flow Statement's explicit items if available
            net_issuance_debt = current_cashflow.get('Net Issuance Of Debt', 0)
            repurchase_debt = current_cashflow.get('Repurchase Of Debt', 0)
            
            net_borrowing_cf = net_issuance_debt - repurchase_debt
            
            # Fallback to Balance Sheet change if CF items are missing or zero
            net_borrowing_bs_diff = net_borrowing_series.loc[date_idx] if date_idx in net_borrowing_series.index else 0
            
            # Use CF explicit items if available and non-zero, otherwise use BS diff
            net_borrowing = net_borrowing_cf if (net_issuance_debt != 0 or repurchase_debt != 0) else net_borrowing_bs_diff

            fcfe_val = net_income + depreciation - capex - delta_wc + net_borrowing
            fcfe_series.loc[date_idx] = fcfe_val

        except Exception as e:
            fcfe_series.loc[date_idx] = np.nan

    fcfe_series = fcfe_series.dropna()
    fcfe_series.name = 'FCFE'
    return fcfe_series

def calculate_cost_of_equity(ticker_info, risk_free_rate, market_risk_premium):
    """
    Calculates Cost of Equity (Ke) using CAPM.
    Ke = Risk-Free Rate + Beta * Market Risk Premium
    """
    beta = ticker_info.get('beta')
    if beta is None or pd.isna(beta):
        st.warning("Could not retrieve Beta from Yahoo Finance. Using a default Beta of 1.0.")
        beta = 1.0 # Default Beta if not found or NaN

    return risk_free_rate + beta * market_risk_premium

def calculate_dcf_model(fcf_history, forecast_years, initial_growth_rates, discount_rate, perpetual_growth_rate, 
                       current_shares_outstanding, current_cash, total_debt, valuation_type="FCFF"):
    """
    Performs DCF calculation for either FCFF or FCFE.
    valuation_type: "FCFF" or "FCFE"
    """
    if fcf_history.empty:
        # st.error(f"Historical {valuation_type} data is not available or could not be calculated for DCF projections.")
        return None, None, None, None, None, None, None

    last_fcf = fcf_history.iloc[-1]
    
    projected_fcf = []
    
    if len(fcf_history) >= 2:
        historical_fcf_growth = fcf_history.pct_change().mean()
        if pd.isna(historical_fcf_growth) or historical_fcf_growth < -0.5 or historical_fcf_growth > 0.5: # Sanity check for extreme growth
            historical_fcf_growth = 0.05 # Default if historical growth is too volatile/missing
    else:
        historical_fcf_growth = 0.05 # Default if less than 2 historical data points

    current_fcf_proj = last_fcf
    
    for i in range(forecast_years):
        if i < len(initial_growth_rates):
            growth_rate = initial_growth_rates[i]
        else:
            if forecast_years > len(initial_growth_rates):
                taper_factor = (i - len(initial_growth_rates) + 1) / (forecast_years - len(initial_growth_rates) + 1)
                growth_rate = historical_fcf_growth * (1 - taper_factor) + perpetual_growth_rate * taper_factor
            else:
                growth_rate = historical_fcf_growth
            
            if growth_rate < perpetual_growth_rate: # Growth should not go below perpetual rate
                growth_rate = perpetual_growth_rate
        
        current_fcf_proj *= (1 + growth_rate)
        projected_fcf.append(current_fcf_proj)

    if discount_rate <= perpetual_growth_rate:
        st.error(f"Discount Rate ({discount_rate:.2%}) must be greater than Perpetual Growth Rate ({perpetual_growth_rate:.2%}) for {valuation_type} DCF calculation to be valid (D_Rate > g).")
        return None, None, None, None, None, None, None
        
    terminal_fcf = projected_fcf[-1] * (1 + perpetual_growth_rate)
    terminal_value = terminal_fcf / (discount_rate - perpetual_growth_rate)
    
    discounted_fcf = []
    for i, fcf_val in enumerate(projected_fcf):
        discount_factor = (1 + discount_rate)**(i + 1)
        discounted_fcf.append(fcf_val / discount_factor)
    
    discounted_terminal_value = terminal_value / ((1 + discount_rate)**forecast_years)
    
    enterprise_value = None # Initialize
    equity_value = None # Initialize

    if valuation_type == "FCFF":
        enterprise_value = sum(discounted_fcf) + discounted_terminal_value
        equity_value = enterprise_value + current_cash - total_debt
    elif valuation_type == "FCFE":
        equity_value = sum(discounted_fcf) + discounted_terminal_value # FCFE already represents cash to equity holders
    else:
        return None, None, None, None, None, None, None

    intrinsic_value_per_share = equity_value / current_shares_outstanding if current_shares_outstanding > 0 else 0

    return intrinsic_value_per_share, equity_value, projected_fcf, discounted_fcf, terminal_value, discounted_terminal_value, enterprise_value

@st.cache_data
def get_nifty_index_value():
    """Fetches the current Nifty 50 index value."""
    try:
        # Use ^NSEI for Nifty 50 index on Yahoo Finance
        nifty_ticker = yf.Ticker("^NSEI")
        nifty_data = nifty_ticker.history(period="1d")
        if not nifty_data.empty:
            return nifty_data['Close'].iloc[-1]
        return None
    except Exception as e:
        st.error(f"Could not fetch Nifty 50 index value: {e}")
        return None

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="Nifty 50 DCF Analysis")

st.title("Nifty 50 Discounted Cash Flow (DCF) Analysis")

st.info("""
This tool performs a simplified DCF analysis for Nifty 50 stocks using publicly available financial data from Yahoo Finance.
It uses two common Free Cash Flow (FCF) methods:
1.  **Free Cash Flow to Firm (FCFF):** Represents cash available to all capital providers (debt & equity holders) after operating expenses and reinvestment. Discounted by WACC to get Enterprise Value, then adjusted for debt/cash to get Equity Value.
2.  **Free Cash Flow to Equity (FCFE):** Represents cash available to equity holders after all expenses, reinvestments, and debt obligations are met. Discounted by Cost of Equity to directly get Equity Value.

**Please note:** Yahoo Finance typically provides **4-5 years** of annual financial statements. The analysis will use the available historical data for FCF calculation and projections.
This is for educational and illustrative purposes only and should not be used for actual investment decisions without professional advice and more rigorous modeling.
""")

# Sidebar for controls
st.sidebar.header("Select Stock & DCF Assumptions")

nifty_symbols = get_nifty50_symbols()
# Ensure nifty_symbols is not empty before proceeding
if not nifty_symbols:
    st.error("Could not load Nifty 50 symbols. Please check your internet connection or try again later.")
    st.stop() # Stop execution if symbols can't be loaded

selected_symbol_yf = st.sidebar.selectbox(
    "Select a Nifty 50 Stock (with .NS suffix)",
    options=nifty_symbols,
    # Set default to RELIANCE.NS if available, otherwise first symbol in the list
    index=nifty_symbols.index('RELIANCE.NS') if 'RELIANCE.NS' in nifty_symbols else 0
)

if selected_symbol_yf:
    st.sidebar.subheader("DCF Model Parameters")
    
    forecast_years = st.sidebar.slider("Forecast Period (Years)", 4, 10, 5)

    st.sidebar.markdown("**Initial Annual Growth Rates for FCF (e.g., 0.10 for 10%)**")
    st.sidebar.markdown("*(These will be applied to the last historical FCF. Beyond these years, growth tapers to perpetual rate.)*")
    
    initial_growth_rates_input = []
    num_initial_growth_years = min(forecast_years, 3) # Allow up to 3 initial growth rates
    for i in range(num_initial_growth_years):
        growth = st.sidebar.number_input(f"Initial Growth Rate Year {i+1}", min_value=-0.10, max_value=0.5, value=0.10 - i*0.02, step=0.01, format="%.2f", key=f"gr_init_{i}")
        initial_growth_rates_input.append(growth)
    
    st.sidebar.subheader("Discount Rates (for WACC & Cost of Equity)")
    wacc = st.sidebar.slider("Weighted Average Cost of Capital (WACC) - for FCFF", 0.05, 0.20, 0.10, step=0.005, format="%.3f")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Cost of Equity (for FCFE) - CAPM Inputs**")
    # Default values based on search results for India (as of recent data)
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate (e.g., 10-yr G-Sec)", min_value=0.03, max_value=0.10, value=0.062, step=0.001, format="%.3f")
    market_risk_premium = st.sidebar.number_input("Market Risk Premium", min_value=0.04, max_value=0.10, value=0.070, step=0.001, format="%.3f")

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

    income_stmt, balance_sheet, cash_flow, ticker_info = get_financial_data(selected_symbol_yf)

    if income_stmt is None or balance_sheet is None or cash_flow is None:
        st.error("Failed to retrieve sufficient financial data for DCF analysis. Please try another stock or check if data is available on Yahoo Finance.")
    else:
        try:
            current_market_price = yf.Ticker(selected_symbol_yf).history(period="1d")['Close'].iloc[-1]
            st.metric("Current Market Price", f"Rs. {current_market_price:,.2f}")

            # Extracting key data points for DCF models
            balance_sheet_t_latest = balance_sheet.T.sort_index(ascending=True).iloc[-1]
            current_shares_outstanding = ticker_info.get('sharesOutstanding', balance_sheet_t_latest.get('Shares Outstanding', 0))
            if current_shares_outstanding == 0 or pd.isna(current_shares_outstanding):
                st.warning(f"Could not find exact 'Shares Outstanding' in Yahoo Finance info or Balance Sheet. Using a default of 1 Billion. This may significantly affect per-share value.")
                current_shares_outstanding = 1_000_000_000 
            else:
                 st.info(f"Shares Outstanding: {current_shares_outstanding:,.0f}")

            current_cash = balance_sheet_t_latest.get('Cash And Cash Equivalents', 0)
            total_debt = balance_sheet_t_latest.get('Total Debt', 0)
            
            st.markdown("---")
            st.subheader("1. Free Cash Flow to Firm (FCFF) Based Valuation")
            
            historical_fcff = calculate_fcff(income_stmt, cash_flow)
            
            intrinsic_value_per_share_fcff = None
            if not historical_fcff.empty:
                st.write(f"Showing {len(historical_fcff)} years of historical FCFF data available:")
                historical_fcff.index = historical_fcff.index.map(lambda x: x.year)
                st.dataframe(historical_fcff.to_frame(name='FCFF').T.style.format("{:,.0f}"))
                
                intrinsic_value_per_share_fcff, equity_value_fcff, projected_fcf_fcff, discounted_fcf_fcff, terminal_value_fcff, discounted_terminal_value_fcff, enterprise_value_fcff = calculate_dcf_model(
                    historical_fcff,
                    forecast_years,
                    initial_growth_rates_input,
                    wacc,
                    perpetual_growth_rate,
                    current_shares_outstanding,
                    current_cash,
                    total_debt,
                    valuation_type="FCFF"
                )

                if intrinsic_value_per_share_fcff is not None:
                    col_fcff_val1, col_fcff_val2, col_fcff_val3 = st.columns(3)
                    col_fcff_val1.metric("FCFF Intrinsic Value per Share", f"Rs. {intrinsic_value_per_share_fcff:,.2f}")
                    
                    upside_downside_fcff = ((intrinsic_value_per_share_fcff - current_market_price) / current_market_price) * 100
                    col_fcff_val2.metric("Upside/Downside (FCFF) (%)", f"{upside_downside_fcff:,.2f}%")
                    
                    if upside_downside_fcff >= valuation_threshold_pct:
                        col_fcff_val3.success(f"**OVERWEIGHT (FCFF):** > {valuation_threshold_pct:.1f}%")
                    elif upside_downside_fcff <= -valuation_threshold_pct:
                        col_fcff_val3.warning(f"**UNDERWEIGHT (FCFF):** < -{valuation_threshold_pct:.1f}%")
                    else:
                        col_fcff_val3.info(f"**FAIRLY VALUED (FCFF):** within +/- {valuation_threshold_pct:.1f}%")

                    with st.expander("Show FCFF Detailed Projections"):
                        st.markdown(f"**FCFF Enterprise Value:** Rs. {enterprise_value_fcff:,.0f}")
                        st.markdown(f"**FCFF Equity Value:** Rs. {equity_value_fcff:,.0f}")
                        
                        proj_fcf_years = [f"Proj {i+1}" for i in range(forecast_years)]
                        proj_fcf_df_fcff = pd.DataFrame({
                            'Year': proj_fcf_years,
                            'Projected FCFF (Rs.)': projected_fcf_fcff,
                            'Discounted FCFF (Rs.)': discounted_fcf_fcff
                        })
                        st.dataframe(proj_fcf_df_fcff.set_index('Year').style.format("{:,.0f}"))
                        st.markdown(f"**Terminal Value (FCFF):** Rs. {terminal_value_fcff:,.0f}")
                        st.markdown(f"**Discounted Terminal Value (FCFF):** Rs. {discounted_terminal_value_fcff:,.0f}")
                else:
                    st.warning("FCFF DCF calculation could not be completed with provided parameters.")
            else:
                st.warning("Historical FCFF could not be calculated. Check financial data availability.")
            
            st.markdown("---")
            st.subheader("2. Free Cash Flow to Equity (FCFE) Based Valuation")
            
            cost_of_equity = calculate_cost_of_equity(ticker_info, risk_free_rate, market_risk_premium)
            st.info(f"Calculated Cost of Equity (Ke) using CAPM: {cost_of_equity:.3%}")

            historical_fcfe = calculate_fcfe(income_stmt, cash_flow, balance_sheet)

            intrinsic_value_per_share_fcfe = None
            if not historical_fcfe.empty:
                st.write(f"Showing {len(historical_fcfe)} years of historical F
