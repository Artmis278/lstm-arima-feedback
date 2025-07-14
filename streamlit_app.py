import streamlit as st
import pandas as pd
import numpy as np
import datetime
import smtplib
from email.mime.text import MIMEText
import plotly.graph_objects as go
import openai

# Load dataset
df = pd.read_csv("LSTM_ARIMA_ActualPrice.csv", parse_dates=["Prediction_Date"])
df = df.dropna(subset=["Actual_Price", "Predicted_LSTM_Price", "Predicted_ARIMA_Price"])

st.set_page_config(page_title="Forecast Evaluation Tool", layout="centered")

# Session state init
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(datetime.datetime.now().timestamp())
if "model_choice" not in st.session_state:
    st.session_state["model_choice"] = "Not selected"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

session_id = st.session_state["session_id"]

# Title and intro
st.title("Steel Price Forecast Evaluation")
st.markdown("Welcome! Please select a prediction date to compare actual steel prices with forecasts from LSTM and ARIMA models.")

# Date selection
dates = df["Prediction_Date"].dt.strftime("%Y-%m").unique()
selected_date_str = st.selectbox("Choose a prediction month:", dates)
selected_date = pd.to_datetime(selected_date_str)

# Selected row and errors
row = df[df["Prediction_Date"] == selected_date].iloc[0]
lstm_abs_error = abs(row["Predicted_LSTM_Price"] - row["Actual_Price"])
arima_abs_error = abs(row["Predicted_ARIMA_Price"] - row["Actual_Price"])
lstm_pct_error = (lstm_abs_error / row["Actual_Price"]) * 100
arima_pct_error = (arima_abs_error / row["Actual_Price"]) * 100

# Forecast table
st.subheader("Forecast Comparison")
st.dataframe(pd.DataFrame({
    "Model": ["LSTM", "ARIMA"],
    "Forecasted Price (USD)": [row["Predicted_LSTM_Price"], row["Predicted_ARIMA_Price"]],
    "Actual Price (USD)": [row["Actual_Price"]] * 2,
    "Absolute Error (USD)": [lstm_abs_error, arima_abs_error],
    "% Error": [round(lstm_pct_error, 2), round(arima_pct_error, 2)]
}).set_index("Model"))

# Plot forecast vs actual
st.subheader("Forecast vs Actual Prices (Interactive)")
trend_df = df.sort_values("Prediction_Date").tail(12)
trend_df["Prediction_Date_str"] = trend_df["Prediction_Date"].dt.strftime("%Y-%m")
trend_df["LSTM_%_Error"] = abs(trend_df["Predicted_LSTM_Price"] - trend_df["Actual_Price"]) / trend_df["Actual_Price"] * 100
trend_df["ARIMA_%_Error"] = abs(trend_df["Predicted_ARIMA_Price"] - trend_df["Actual_Price"]) / trend_df["Actual_Price"] * 100

fig = go.Figure()
fig.add_trace(go.Scatter(x=trend_df["Prediction_Date_str"], y=trend_df["Actual_Price"], mode='lines+markers', name='Actual', line=dict(color='black')))
fig.add_trace(go.Scatter(x=trend_df["Prediction_Date_str"], y=trend_df["Predicted_LSTM_Price"], mode='lines+markers', name='LSTM Prediction', text=[f"LSTM Error: {e:.2f}%" for e in trend_df["LSTM_%_Error"]], hoverinfo='text+y'))
fig.add_trace(go.Scatter(x=trend_df["Prediction_Date_str"], y=trend_df["Predicted_ARIMA_Price"], mode='lines+markers', name='ARIMA Prediction', text=[f"ARIMA Error: {e:.2f}%" for e in trend_df["ARIMA_%_Error"]], hoverinfo='text+y'))

highlight = trend_df[trend_df["Prediction_Date_str"] == selected_date_str]
if not highlight.empty:
    fig.add_trace(go.Scatter(x=highlight["Prediction_Date_str"], y=highlight["Actual_Price"], mode='markers', name='Selected Date', marker=dict(size=12, color='gold', symbol='star')))

fig.update_layout(title="Actual vs Predicted Prices Over Time", xaxis_title="Prediction Month", yaxis_title="Price (USD)", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# Model explanation
with st.expander("How do these models work?"):
    st.markdown("**LSTM** is a deep learning model for sequences, great for volatile markets.")
    st.markdown("**ARIMA** is a statistical model using past values, known for transparency.")

# Email sender
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
        st.error(f"Failed to send feedback via email: {e}")
        return False

# -----------------------------
# ForecastPal Chat UI
# -----------------------------
with st.container():
    st.markdown("""
        <div style='border: 1px solid lightgray; border-radius: 12px; padding: 20px; background-color: #f9f9f9;'>
            <h3>\ud83d\udcac Ask ForecastPal \ud83e\uddd9</h3>
            <p>If you have any questions about the forecasts, modeling approach, or why the models differ,<br>
            ask ForecastPal – your steel forecasting sidekick!</p>
    """, unsafe_allow_html=True)

    st.markdown("<div style='max-height: 300px; overflow-y: auto;'>", unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        st.markdown(f"""
            <div style='margin-bottom: 1rem; padding: 10px; background-color: #ffffff; border-radius: 10px; border: 1px solid #ddd;'>
                <p style='margin:0;'><b>\U0001F9D1 You ({chat['timestamp']}):</b><br>{chat['question']}</p>
                <p style='margin:8px 0 0 0;'><b>\u263A\ufe0f ForecastPal:</b><br>{chat['answer']}</p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.form(key="chat_form", clear_on_submit=True):
        user_question = st.text_input("Ask ForecastPal...", placeholder="Type your question here")
        submitted = st.form_submit_button("Send")
        if submitted and user_question.strip():
            st.session_state["pending_question"] = user_question

    if "pending_question" in st.session_state:
        user_question = st.session_state["pending_question"]
        with st.spinner("ForecastPal is thinking..."):
            try:
                openai.api_key = st.secrets["openai"]["api_key"]
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are ForecastPal, a helpful assistant that explains AI models, especially LSTM and ARIMA, used for steel price forecasting."},
                        {"role": "user", "content": user_question}
                    ]
                )
                reply = response.choices[0].message.content
                st.session_state.chat_history.append({
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "question": user_question,
                    "answer": reply
                })
                subject = f"\U0001F4E9 Chatbot Question Logged [Session ID: {session_id}]"
                body = f"""Chatbot Question Submitted\nSession ID: {session_id}\nTimestamp: {datetime.datetime.now().isoformat()}\nPrediction Date: {selected_date_str}\nModel Trusted: {st.session_state['model_choice']}\nUser Question: {user_question}\nAI Response: {reply}"
                send_feedback_email(subject, body)
                del st.session_state["pending_question"]
                st.experimental_rerun()
            except Exception as e:
                st.error(f"\u26a0\ufe0f ForecastPal had a problem: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Feedback Section
# -----------------------------
# Expert Feedback Form
st.subheader("🗣️ Expert Feedback")
model_choice = st.radio("Which model's prediction do you trust more?", ["LSTM", "ARIMA", "Both equally", "Neither"])
st.session_state["model_choice"] = model_choice
confidence = st.slider("How confident would you be using this forecast in a real procurement decision?", 0, 100, 50)
comment = st.text_area("Additional comments (optional):")

if st.button("Submit Feedback"):
    subject = f"🗣️ Forecast Feedback - {model_choice} [Session ID: {session_id}]"
    body = f"""Forecast Feedback Submitted
Session ID: {session_id}
Date: {selected_date_str}
Model: {model_choice}
Confidence: {confidence}
Comment: {comment}
Timestamp: {datetime.datetime.now().isoformat()}
"""
    if send_feedback_email(subject, body):
        st.success("📧 Thank you! Your feedback has been emailed successfully.")
    else:
        st.error("❌ Feedback could not be sent.")
