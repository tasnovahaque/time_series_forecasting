import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import load_model

# Load dataset
def load_data():
    data = pd.read_csv("dataset.csv")   # must exist in repo
    return data.iloc[:, 0].values

series = load_data()

# Load trained LSTM model if available
try:
    lstm_model = load_model("models/lstm_model.h5")
except:
    lstm_model = None

def forecast(model_type):
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    if model_type == "ARIMA":
        history = train.tolist()
        predictions = []
        order = (1,1,1)
        for t in range(len(test)):
            model = ARIMA(history, order=order)
            model_fit = model.fit()
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(test[t])
        rmse = np.sqrt(mean_squared_error(test, predictions))
        mape = mean_absolute_percentage_error(test, predictions)

    elif model_type == "LSTM" and lstm_model:
        scaler = MinMaxScaler()
        ts_scaled = scaler.fit_transform(series.reshape(-1,1))
        train_scaled = ts_scaled[:train_size]
        test_scaled = ts_scaled[train_size:]
        time_steps = 60
        history = ts_scaled[:train_size].tolist()
        predictions = []
        for i in range(len(test_scaled)):
            input_seq = np.array(history[-time_steps:]).reshape(1, time_steps, 1)
            yhat = lstm_model.predict(input_seq, verbose=0)[0,0]
            predictions.append(yhat)
            history.append(test_scaled[i])
        predictions = np.array(predictions).reshape(-1,1)
        predictions = scaler.inverse_transform(predictions)
        test = scaler.inverse_transform(test_scaled)
        rmse = np.sqrt(mean_squared_error(test, predictions))
        mape = mean_absolute_percentage_error(test, predictions)
    else:
        return None

    # Plot
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(test, label="Actual", color="blue")
    ax.plot(predictions, label=f"{model_type} Predictions", color="red")
    ax.set_title(f"{model_type} Forecast vs Actual\nRMSE={rmse:.2f}, MAPE={mape:.2f}")
    ax.legend()
    return fig

iface = gr.Interface(
    fn=forecast,
    inputs=gr.Dropdown(["ARIMA", "LSTM"], label="Select Model"),
    outputs=gr.Plot(),
    title="DataSynthis_ML_JobTask",
    description="Compare ARIMA vs LSTM forecasting"
)

if __name__ == "__main__":
    iface.launch()

