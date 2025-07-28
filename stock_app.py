import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Predictor (Next 7 Days)")

# User inputs
tickers = st.text_input("Enter Stock Ticker(s) (comma-separated)", "AAPL").upper().split(",")
start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))

for ticker in tickers:
    ticker = ticker.strip()
    if not ticker:
        continue
    if st.button(f"Predict: {ticker}"):
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error(f"No data found for {ticker}")
        else:
            st.success(f"Fetched data for {ticker}")
            df = data.reset_index()
            df['Date_ordinal'] = df['Date'].apply(lambda x: x.toordinal())

            X = df[['Date_ordinal']]
            y = df['Close']
            model = LinearRegression()
            model.fit(X, y)

            last_date = df['Date'].max()
            future = [last_date + timedelta(days=i) for i in range(1, 8)]
            future_ordinal = np.array([d.toordinal() for d in future]).reshape(-1, 1)
            preds = model.predict(future_ordinal)

            forecast_df = pd.DataFrame({'Date': future, 'Predicted Close': np.round(preds, 2)})
            st.subheader(f"📉 {ticker} Forecast – Next 7 Days")
            st.dataframe(forecast_df)

            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", data=csv,
                               file_name=f"{ticker}_7_day_predictions.csv",
                               mime="text/csv")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'],
                                     mode='lines', name='Actual'))
            fig.add_trace(go.Scatter(x=forecast_df['Date'],
                                     y=forecast_df['Predicted Close'],
                                     mode='lines+markers', name='Predicted'))
            fig.update_layout(title=f"{ticker} Closing Price Forecast",
                              xaxis_title='Date', yaxis_title='Price',
                              template='plotly_dark')

            st.plotly_chart(fig, use_container_width=True)
