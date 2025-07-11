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
    text=[f"LSTM Error: {e:.2f}%" for e in trend_df["LSTM_%_Error"]],
    hoverinfo='text+y'
))

fig.add_trace(go.Scatter(
    x=trend_df["Prediction_Date_str"],
    y=trend_df["Predicted_ARIMA_Price"],
    mode='lines+markers',
    name='ARIMA Prediction',
    text=[f"ARIMA Error: {e:.2f}%" for e in trend_df["ARIMA_%_Error"]],
    hoverinfo='text+y'
))

# Highlight selected date
highlight = trend_df[trend_df["Prediction_Date_str"] == selected_date_str]
if not highlight.empty:
    fig.add_trace(go.Scatter(
        x=highlight["Prediction_Date_str"],
        y=highlight["Actual_Price"],
        mode='markers',
        name='Selected Date',
        marker=dict(size=12, color='gold', symbol='star'),
        showlegend=True
    ))

fig.update_layout(
    title="📈 Actual vs Predicted Prices Over Time",
    xaxis_title="Prediction Month",
    yaxis_title="Price (USD)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# Model explanation
with st.expander("ℹ️ How do these models work?"):
    st.markdown("**LSTM** (Long Short-Term Memory) is a type of deep learning model that learns from historical sequences of data. It is designed to detect complex, nonlinear patterns in time series, making it well-suited for forecasting tasks in volatile markets.")
    st.markdown("**ARIMA** (AutoRegressive Integrated Moving Average) is a classical statistical model that uses past values and error terms to predict future points. It is known for its transparency and interpretability but can struggle with rapidly changing trends.")

# Feedback form
st.subheader("🗣️ Expert Feedback")
model_choice = st.radio("Which model's prediction do you trust more?", ["LSTM", "ARIMA", "Both equally", "Neither"])
confidence = st.slider("How confident would you be using this forecast in a real procurement decision?", 0, 100, 50)
comment = st.text_area("Additional comments (optional):")

# Email fallback function
def send_feedback_email(subject, body):
    try:
        sender = st.secrets["email"]["address"]
        password = st.secrets["email"]["app_password"]
        receiver = sender

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = receiver

        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"📧 Failed to send feedback via email: {e}")
        return False

# Submit feedback logic
if st.button("Submit Feedback"):
    subject = f"🗣️ Forecast Feedback - {model_choice}"
    body = f"""Date: {selected_date_str}
Model: {model_choice}
Confidence: {confidence}
Comment: {comment}
Timestamp: {datetime.datetime.now().isoformat()}"""

    if send_feedback_email(subject, body):
        st.success("📧 Thank you! Your feedback has been emailed successfully.")
    else:
        st.error("❌ Feedback could not be sent.")
