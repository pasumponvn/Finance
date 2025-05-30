import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px # Import plotly for interactive plots

# --- Configuration ---
# Set page configuration for better aesthetics
st.set_page_config(
    page_title="NSE Stock Valuation (DDM)",
    page_icon="�",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Helper Functions ---

def calculate_gordon_growth_ddm(d0, r, g):
    """
    Calculates the intrinsic value of a stock using the Gordon Growth Model.

    Args:
        d0 (float): Last paid annual dividend per share.
        r (float): Required rate of return (cost of equity) as a decimal.
        g (float): Constant growth rate of dividends as a decimal.

    Returns:
        float: Intrinsic value of the stock.
        str: Error message if calculation is not possible, otherwise None.
    """
    if r <= g:
        return None, "Error: Required rate of return (r) must be greater than dividend growth rate (g)."
    if d0 <= 0:
        return None, "Error: Last paid dividend (D0) must be positive for DDM."

    d1 = d0 * (1 + g) # Expected dividend next year
    intrinsic_value = d1 / (r - g)
    return intrinsic_value, None

def get_stock_data(ticker_symbol, benchmark_ticker="^NSEI"): # Added benchmark_ticker
    """
    Fetches current price, historical dividend data, and historical price data for a given ticker,
    and also fetches historical data for a benchmark ticker.

    Args:
        ticker_symbol (str): The stock ticker symbol (e.g., "RELIANCE.NS").
        benchmark_ticker (str): The benchmark ticker symbol (default: "^NSEI" for NIFTY 50).

    Returns:
        tuple: (current_price, latest_annual_dividend, historical_prices_df, benchmark_prices_df, error_message)
               latest_annual_dividend is sum of dividends over last 12 months.
               historical_prices_df is a pandas DataFrame with 'Date' and 'Close' columns.
               benchmark_prices_df is a pandas DataFrame with 'Date' and 'Close' columns for the benchmark.
    """
    try:
        stock = yf.Ticker(ticker_symbol)

        # Fetch historical data for charting (last 2 years)
        hist_data = stock.history(period="2y")
        if hist_data.empty:
            return None, None, None, None, "No historical data found for this stock. Check ticker symbol."

        current_price = hist_data['Close'].iloc[-1]
        historical_prices_df = hist_data[['Close']].reset_index()
        historical_prices_df.columns = ['Date', 'Close'] # Rename columns for clarity

        # Fetch benchmark data
        benchmark = yf.Ticker(benchmark_ticker)
        benchmark_hist_data = benchmark.history(period="2y")
        if benchmark_hist_data.empty:
            return None, None, None, None, f"No historical data found for benchmark {benchmark_ticker}. Check ticker symbol."

        benchmark_prices_df = benchmark_hist_data[['Close']].reset_index()
        benchmark_prices_df.columns = ['Date', 'Close']

        # Get historical dividends for the last 12 months
        dividends = stock.dividends
        if dividends.empty:
            return current_price, 0, historical_prices_df, benchmark_prices_df, "No historical dividend data found for this stock."

        # Sum dividends for the last 12 months (approx. annual dividend)
        latest_dividend_date = dividends.index.max()
        if pd.isna(latest_dividend_date):
            return current_price, 0, historical_prices_df, benchmark_prices_df, "Could not determine latest dividend date."

        one_year_ago = latest_dividend_date - pd.Timedelta(days=365)
        recent_dividends = dividends[dividends.index >= one_year_ago]
        latest_annual_dividend = recent_dividends.sum()

        if latest_annual_dividend == 0:
            return current_price, 0, historical_prices_df, benchmark_prices_df, "No dividends paid in the last 12 months."

        return current_price, latest_annual_dividend, historical_prices_df, benchmark_prices_df, None
    except Exception as e:
        return None, None, None, None, f"Could not fetch data for {ticker_symbol} or {benchmark_ticker}. Error: {e}"

# --- Streamlit UI ---

st.title("📈 NSE Stock Valuation using DDM")
st.markdown("""
This application helps you estimate the intrinsic value of an NSE stock using the Gordon Growth Dividend Discount Model (DDM).
Compare the intrinsic value with the current market price to determine if a stock is "overweight" (undervalued) or "underweight" (overvalued).
""")

st.info("💡 **Note:** For NSE stocks, remember to add `.NS` to the ticker symbol (e.g., `RELIANCE.NS`, `TCS.NS`). The benchmark used is NIFTY 50 (`^NSEI`).")

# Input fields
ticker_input = st.text_input("Enter NSE Stock Ticker (e.g., RELIANCE.NS)", "RELIANCE.NS").strip().upper()

st.subheader("DDM Parameters")

# Required Rate of Return (r)
st.markdown("### Required Rate of Return (r)")
st.markdown("""
This represents the minimum rate of return an investor expects from an investment.
It is often estimated using the Capital Asset Pricing Model (CAPM):
$r = R_f + \\beta \\times (R_m - R_f)$
Where:
- $R_f$: Risk-free rate (e.g., yield on a 10-year Indian Government Bond).
- $\\beta$: Beta of the stock (measures volatility relative to the market).
- $R_m - R_f$: Market risk premium (expected market return minus risk-free rate).
""")

# Example values for India:
# Risk-free rate (Rf): ~7.2% (as of late 2024/early 2025 for 10-year G-Sec)
# Market Risk Premium (Rm - Rf): ~6-8% (historical average for India)
# Beta: Varies by stock (can be found on financial websites)

col1, col2 = st.columns(2)
with col1:
    risk_free_rate = st.number_input("Risk-Free Rate ($R_f$, e.g., 0.072 for 7.2%)", min_value=0.0, max_value=1.0, value=0.072, step=0.001, format="%.3f")
with col2:
    market_risk_premium = st.number_input("Market Risk Premium ($R_m - R_f$, e.g., 0.07 for 7%)", min_value=0.0, max_value=1.0, value=0.07, step=0.001, format="%.3f")

beta_input = st.number_input("Stock Beta ($\beta$, e.g., 1.0)", min_value=0.0, value=1.0, step=0.01, format="%.2f")

# Calculate 'r' using CAPM
required_rate_of_return = risk_free_rate + beta_input * market_risk_premium
st.write(f"Calculated Required Rate of Return ($r$): **{required_rate_of_return:.2%}**")

st.markdown("### Dividend Growth Rate (g)")
st.markdown("""
This is the expected constant annual growth rate of the company's dividends.
It can be estimated from historical dividend growth, analyst forecasts, or using the formula:
$g = ROE \\times (1 - Payout Ratio)$
Where:
- $ROE$: Return on Equity
- Payout Ratio: Dividends / Net Income
""")
dividend_growth_rate = st.number_input("Dividend Growth Rate ($g$, e.g., 0.05 for 5%)", min_value=0.0, max_value=required_rate_of_return - 0.001, value=0.05, step=0.001, format="%.3f", help="Must be less than Required Rate of Return (r)")


if st.button("Analyze Stock"):
    if not ticker_input:
        st.error("Please enter a stock ticker symbol.")
    else:
        with st.spinner(f"Fetching data for {ticker_input} and benchmark..."):
            current_price, d0, historical_prices_df, benchmark_prices_df, fetch_error = get_stock_data(ticker_input)

        if fetch_error:
            st.error(fetch_error)
        elif current_price is None or d0 is None or historical_prices_df is None or benchmark_prices_df is None:
            st.error("Failed to retrieve essential stock data. Please check the ticker symbol and internet connection.")
        else:
            st.subheader(f"Analysis for {ticker_input}")

            # Display the price chart with benchmark comparison
            st.markdown("---")
            st.subheader("Stock Price Performance vs. NIFTY 50 (Last 2 Years)")

            # Normalize prices to their starting point for percentage comparison
            initial_stock_price = historical_prices_df['Close'].iloc[0]
            initial_benchmark_price = benchmark_prices_df['Close'].iloc[0]

            historical_prices_df['Normalized Close'] = (historical_prices_df['Close'] / initial_stock_price - 1) * 100
            benchmark_prices_df['Normalized Close'] = (benchmark_prices_df['Close'] / initial_benchmark_price - 1) * 100

            # Merge dataframes for plotting
            merged_df = pd.merge(historical_prices_df[['Date', 'Normalized Close']],
                                 benchmark_prices_df[['Date', 'Normalized Close']],
                                 on='Date',
                                 suffixes=(f'_{ticker_input}', '_Benchmark'))

            # Melt the dataframe for Plotly Express
            plot_df = merged_df.melt(id_vars=['Date'],
                                     value_vars=[f'Normalized Close_{ticker_input}', 'Normalized Close_Benchmark'],
                                     var_name='Asset',
                                     value_name='Percentage Change')

            # Rename assets for legend
            plot_df['Asset'] = plot_df['Asset'].replace({
                f'Normalized Close_{ticker_input}': ticker_input,
                'Normalized Close_Benchmark': 'NIFTY 50'
            })

            fig = px.line(plot_df, x='Date', y='Percentage Change', color='Asset',
                          title=f'{ticker_input} vs. NIFTY 50 Performance (Percentage Change)')
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Percentage Change (%)")
            st.plotly_chart(fig, use_container_width=True)


            st.markdown("---")
            st.metric("Current Market Price", f"₹{current_price:,.2f}")
            st.metric("Last 12-Month Dividends (D0)", f"₹{d0:,.2f}")

            if d0 == 0:
                st.warning("Warning: No dividends paid in the last 12 months. DDM is not suitable for non-dividend paying stocks.")
            else:
                intrinsic_value, ddm_error = calculate_gordon_growth_ddm(d0, required_rate_of_return, dividend_growth_rate)

                if ddm_error:
                    st.error(ddm_error)
                else:
                    st.metric("Calculated Intrinsic Value (DDM)", f"₹{intrinsic_value:,.2f}")

                    # Determine if overweight or underweight
                    price_difference = intrinsic_value - current_price
                    percentage_difference = (price_difference / current_price) * 100 if current_price != 0 else 0

                    st.markdown("---")
                    st.subheader("Valuation Result")

                    if percentage_difference > 10: # More than 10% undervalued
                        st.success(f"**Underweight by Market** (Undervalued): The intrinsic value is {percentage_difference:.2f}% higher than the current price. This stock might be a **BUY** and could be **Overweight** in your portfolio.")
                    elif percentage_difference < -10: # More than 10% overvalued
                        st.error(f"**Overweight by Market** (Overvalued): The intrinsic value is {abs(percentage_difference):.2f}% lower than the current price. This stock might be a **SELL** and could be **Underweight** in your portfolio.")
                    else:
                        st.info(f"**Fairly Valued**: The intrinsic value is within ±10% of the current price ({percentage_difference:.2f}% difference). This stock might be a **HOLD**.")

                    st.markdown("""
                    <small>
                    *Disclaimer: This tool provides a simplified valuation based on the Gordon Growth Model.
                    It should not be used as the sole basis for investment decisions.
                    Actual stock performance can vary significantly.
                    Always conduct thorough research and consult with a financial advisor.*
                    </small>
                    """, unsafe_allow_html=True)


