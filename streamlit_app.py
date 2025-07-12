import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import datetime
import smtplib
from email.mime.text import MIMEText
import plotly.graph_objects as go  # Added for interactive chart

# Load dataset
df = pd.read_csv("LSTM_ARIMA_ActualPrice.csv", parse_dates=["Prediction_Date"])
df = df.dropna(subset=["Actual_Price", "Predicted_LSTM_Price", "Predicted_ARIMA_Price"])

st.set_page_config(page_title="Forecast Evaluation Tool", layout="centered")

st.title("🔍 Steel Price Forecast Evaluation")
st.markdown("Welcome! Please select a prediction date to compare actual steel prices with forecasts from LSTM and ARIMA models.")

# Date selection
dates = df["Prediction_Date"].dt.strftime("%Y-%m").unique()
selected_date_str = st.selectbox("Choose a prediction month:", dates)
selected_date = pd.to_datetime(selected_date_str)

# Filter for selected date
row = df[df["Prediction_Date"] == selected_date].iloc[0]

# Calculate errors
lstm_abs_error = abs(row["Predicted_LSTM_Price"] - row["Actual_Price"])
arima_abs_error = abs(row["Predicted_ARIMA_Price"] - row["Actual_Price"])
lstm_pct_error = (lstm_abs_error / row["Actual_Price"]) * 100
arima_pct_error = (arima_abs_error / row["Actual_Price"]) * 100

# Display comparison
table_data = {
    "Model": ["LSTM", "ARIMA"],
    "Forecasted Price (USD)": [row["Predicted_LSTM_Price"], row["Predicted_ARIMA_Price"]],
    "Actual Price (USD)": [row["Actual_Price"], row["Actual_Price"]],
    "Absolute Error (USD)": [lstm_abs_error, arima_abs_error],
    "% Error": [round(lstm_pct_error, 2), round(arima_pct_error, 2)]
}
st.subheader("📊 Forecast Comparison")
st.dataframe(pd.DataFrame(table_data).set_index("Model"))

# 📈 Plotly Chart: Forecast vs Actual Over Time
st.subheader("📈 Forecast vs Actual Prices (Interactive)")

# Prepare data for plotting
trend_df = df.sort_values("Prediction_Date").tail(12)
trend_df["Prediction_Date_str"] = trend_df["Prediction_Date"].dt.strftime("%Y-%m")
trend_df["LSTM_%_Error"] = abs(trend_df["Predicted_LSTM_Price"] - trend_df["Actual_Price"]) / trend_df["Actual_Price"] * 100
trend_df["ARIMA_%_Error"] = abs(trend_df["Predicted_ARIMA_Price"] - trend_df["Actual_Price"]) / trend_df["Actual_Price"] * 100

# Create Plotly figure
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=trend_df["Prediction_Date_str"],
    y=trend_df["Actual_Price"],
    mode='lines+markers',
    name='Actual',
    line=dict(color='black')
))

fig.add_trace(go.Scatter(
    x=trend_df["Prediction_Date_str"],
    y=trend_df["Predicted_LSTM_Price"],
    mode='lines+markers',
    name='LSTM Prediction',
    text=[f"LSTM Error: {e:.2f}%" for e in trend_df["LSTM]()_]()
