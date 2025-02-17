# Commented out IPython magic to ensure Python compatibility.

from pyngrok import ngrok
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np

st.title("Stock Prediction Dashboard")
st.write("Enter the details below:")

# Paste your actual ngrok auth token here
ngrok.set_auth_token("2tASRt9PdowYUZHRa9vSceICoC8_892R3ymymYPByDGaXffca")

# User Inputs
ticker = st.text_input("Enter Stock Ticker", "TATAMOTORS.NS")

model_choice = st.selectbox("Select Prediction Model", ["Linear Regression", "Random Forest"])

time_period = st.slider("Select Time Period (Days)", 1, 720, 30)  # Now allows up to 10 years

if st.button("Submit"):
    st.write(f"Fetching data for **{ticker}** and predicting using **{model_choice}** for **{time_period}** days...")

    # Fetch historical stock data
    stock_data = yf.download(ticker, period="10y", interval="1d")  # Increased to 10 years
    stock_data = stock_data.reset_index()

    if stock_data.empty:
        st.error("Failed to fetch stock data. Check the ticker symbol and try again.")
    else:
        st.success("Data successfully loaded!")

        # Prepare data for prediction
        stock_data["Days"] = np.arange(len(stock_data))
        X = stock_data[["Days"]]
        y = stock_data["Close"]

        # Train the selected model
        if model_choice == "Linear Regression":
            model = LinearRegression()
            model.fit(X, y)
            st.write("Trained Linear Regression Model")

        elif model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            st.write("Trained Random Forest Model")

        # Predict for the future
        future_days = np.arange(len(stock_data), len(stock_data) + time_period).reshape(-1, 1)
        if model_choice == "LSTM":
            future_days_lstm = future_days.reshape(-1, 1, 1)
            predicted_prices = model.predict(future_days_lstm)
        else:
            predicted_prices = model.predict(future_days)

        # Plot the results
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(stock_data["Date"], stock_data["Close"], label="Historical Prices", color="blue")
        ax.plot(pd.date_range(start=stock_data["Date"].iloc[-1], periods=time_period), predicted_prices, label="Predicted Prices", color="red", linestyle="dashed")
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price")
        ax.set_title(f"{ticker} Stock Price Prediction")
        ax.legend()

        st.pyplot(fig)



