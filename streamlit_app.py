import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import streamlit as st

try:
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials = Credentials.from_service_account_info(st.secrets["gspread"], scopes=scope)
    client = gspread.authorize(credentials)

    st.success("✅ Google Sheets authentication successful!")

    # Try opening your target sheet
    spreadsheet = client.open("LSTM_ARIMA_Feedback")
    st.info(f"📄 Connected to spreadsheet: {spreadsheet.title}")

    # List sheet names
    sheet_names = [ws.title for ws in spreadsheet.worksheets()]
    st.write("Available worksheets:", sheet_names)

except Exception as e:
    st.error(f"❌ Failed to connect to Google Sheets: {e}")

from google.oauth2.service_account import Credentials
import datetime

# Load dataset
df = pd.read_csv("LSTM_ARIMA_ActualPrice.csv", parse_dates=["Prediction_Date"])

# Drop rows with missing actual or predicted values
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

# Show comparison
table_data = {
    "Model": ["LSTM", "ARIMA"],
    "Forecasted Price (USD)": [row["Predicted_LSTM_Price"], row["Predicted_ARIMA_Price"]],
    "Actual Price (USD)": [row["Actual_Price"], row["Actual_Price"]],
    "Absolute Error (USD)": [lstm_abs_error, arima_abs_error],
    "% Error": [round(lstm_pct_error, 2), round(arima_pct_error, 2)]
}
st.subheader("📊 Forecast Comparison")
st.dataframe(pd.DataFrame(table_data).set_index("Model"))

# Model explanation
with st.expander("ℹ️ How do these models work?"):
    st.markdown("**LSTM** (Long Short-Term Memory) is a type of deep learning model that learns from historical sequences of data. It is designed to detect complex, nonlinear patterns in time series, making it well-suited for forecasting tasks in volatile markets.")
    st.markdown("**ARIMA** (AutoRegressive Integrated Moving Average) is a classical statistical model that uses past values and error terms to predict future points. It is known for its transparency and interpretability but can struggle with rapidly changing trends.")

# Feedback form
st.subheader("🗣️ Expert Feedback")
model_choice = st.radio("Which model's prediction do you trust more?", ["LSTM", "ARIMA", "Both equally", "Neither"])
confidence = st.slider("How confident would you be using this forecast in a real procurement decision?", 0, 100, 50)
comment = st.text_area("Additional comments (optional):")

import datetime
import os

if st.button("Submit Feedback"):
    try:
        # Connect to Google Sheets using Streamlit secrets
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        credentials = Credentials.from_service_account_info(st.secrets["gspread"], scopes=scope)
        client = gspread.authorize(credentials)
        st.success("✅ Google Sheets authentication successful!")  # 👈 This confirms your secret works
        # Open sheet by name
        sheet = client.open("LSTM_ARIMA_Feedback").worksheet("responses")

        # Prepare the row
        new_row = [
            datetime.datetime.now().isoformat(),
            selected_date_str,
            model_choice,
            confidence,
            comment
        ]

        # Append to the sheet
        sheet.append_row(new_row)
        st.success("✅ Thank you! Your feedback has been saved.")

    except Exception as e:
        st.error(f"⚠️ Something went wrong while saving your feedback: {e}")


