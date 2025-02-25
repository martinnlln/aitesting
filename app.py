import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ollama
import tempfile
import base64
import os
from datetime import datetime
import threading

def fetch_stock_data(ticker, start_date, end_date):
    """Fetch historical stock data from yfinance."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("No data found for the provided ticker. Please check your input.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def create_candlestick_chart(data):
    """Create a base candlestick chart using Plotly."""
    fig = go.Figure(data=[
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Candlestick"
        )
    ])
    return fig

def add_indicator(fig, data, indicator):
    """Add the selected technical indicator to the chart."""
    try:
        if indicator == "20-Day SMA":
            sma = data['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)', line=dict(color='blue')))
        elif indicator == "20-Day EMA":
            ema = data['Close'].ewm(span=20, adjust=False).mean()
            fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)', line=dict(color='orange')))
        elif indicator == "20-Day Bollinger Bands":
            sma = data['Close'].rolling(window=20).mean()
            std = data['Close'].rolling(window=20).std()
            bb_upper = sma + 2 * std
            bb_lower = sma - 2 * std
            fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower', line=dict(color='red')))
        elif indicator == "VWAP":
            data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
            fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP', line=dict(color='purple')))
        elif indicator == "RSI (14)":
            delta = data['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            fig.add_trace(go.Scatter(x=data.index, y=rsi, mode='lines', name='RSI (14)', line=dict(color='cyan')))
        elif indicator == "MACD":
            ema12 = data['Close'].ewm(span=12, adjust=False).mean()
            ema26 = data['Close'].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            fig.add_trace(go.Scatter(x=data.index, y=macd_line, mode='lines', name='MACD', line=dict(color='brown')))
            fig.add_trace(go.Scatter(x=data.index, y=signal_line, mode='lines', name='Signal', line=dict(color='pink')))
        elif indicator == "ATR (14)":
            high_low = data['High'] - data['Low']
            high_close = (data['High'] - data['Close'].shift()).abs()
            low_close = (data['Low'] - data['Close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            fig.add_trace(go.Scatter(x=data.index, y=atr, mode='lines', name='ATR (14)', line=dict(color='magenta')))
        elif indicator == "OBV":
            delta = data['Close'].diff()
            direction = delta.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            obv = (direction * data['Volume']).cumsum()
            fig.add_trace(go.Scatter(x=data.index, y=obv, mode='lines', name='OBV', line=dict(color='yellow')))
        elif indicator == "Stochastic Oscillator":
            low_min = data['Low'].rolling(window=14).min()
            high_max = data['High'].rolling(window=14).max()
            stochastic_k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
            stochastic_d = stochastic_k.rolling(window=3).mean()
            fig.add_trace(go.Scatter(x=data.index, y=stochastic_k, mode='lines', name='%K', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=data.index, y=stochastic_d, mode='lines', name='%D', line=dict(color='purple')))
    except Exception as e:
        st.error(f"Error adding {indicator}: {e}")

def run_ai_analysis(fig):
    """Save the chart as an image, encode it, send it for AI analysis, and return the result."""
    try:
        # Save chart as a temporary image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            fig.write_image(tmpfile.name)
            tmpfile_path = tmpfile.name

        # Encode the image to Base64
        with open(tmpfile_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

        # Prepare the AI analysis request
        messages = [{
            'role': 'user',
            'content': (
                "You are a Stock Trader specializing in Technical Analysis at a top financial institution. "
                "Analyze the stock chart's technical indicators and provide a buy/hold/sell recommendation. "
                "Base your recommendation only on the candlestick chart and the displayed technical indicators. "
                "First, provide the recommendation, then, provide your detailed reasoning."
            ),
            'images': [image_data]
        }]

        # This function will run the analysis
        def analysis_thread():
            response = ollama.chat(model='llama3.2-vision', messages=messages)
            result = response.get("message", {}).get("content", "No response from AI analysis.")
            st.session_state['ai_analysis_result'] = result

            # Clean up the temporary image file
            os.remove(tmpfile_path)

        # Start the analysis in a separate thread
        analysis = threading.Thread(target=analysis_thread)
        analysis.start()

        # Wait for the analysis to finish with a timeout
        analysis.join(timeout=30)  # 30 seconds timeout
        if analysis.is_alive():
            st.error("AI analysis is taking too long. Please try again later.")
            # Clean up if necessary (not shown here)

        # If analysis completed successfully
        if 'ai_analysis_result' in st.session_state:
            return st.session_state['ai_analysis_result']
        
    except Exception as e:
        st.error(f"Error during AI analysis: {e}")
        return None

# Set up Streamlit app
st.set_page_config(layout="wide")
st.title("AI-Powered Technical Stock Analysis Dashboard")
st.sidebar.header("Configuration")

# Inputs for stock ticker and date range
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime(datetime.today().strftime('%Y-%m-%d')))

# Fetch stock data on button click
if st.sidebar.button("Fetch Data"):
    with st.spinner("Fetching stock data..."):
        data = fetch_stock_data(ticker, start_date, end_date)
        if data is not None:
            st.session_state["stock_data"] = data
            st.success("Stock data loaded successfully!")

# Check if data is available in session state
if "stock_data" in st.session_state:
    data = st.session_state["stock_data"]

    # Create base candlestick chart
    fig = create_candlestick_chart(data)

    # Sidebar: Select technical indicators
    st.sidebar.subheader("Technical Indicators")
    indicators = st.sidebar.multiselect(
        "Select Indicators:",
        [
            "20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP",
            "RSI (14)", "MACD", "ATR (14)", "OBV", "Stochastic Oscillator"
        ],
        default=["20-Day SMA"]
    )

    # Add each selected indicator to the chart
    for indicator in indicators:
        add_indicator(fig, data, indicator)

    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # AI-Powered Analysis Section
    st.subheader("AI-Powered Analysis")
    if st.button("Run AI Analysis"):
        with st.spinner("Analyzing the chart, please wait..."):
            result = run_ai_analysis(fig)
            if result:
                st.write("**AI Analysis Results:**")
                st.write(result)
