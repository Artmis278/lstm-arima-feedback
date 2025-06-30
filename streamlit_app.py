import streamlit as st
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_excel("LSTM_ARIMA_ActualPrice.xlsx")

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

if st.button("Submit Feedback"):
    st.success("✅ Thank you! Your feedback has been recorded.")
    # You can later extend this to save feedback to a file or database

