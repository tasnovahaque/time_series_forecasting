import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load dataset and LSTM model from root
data = pd.read_csv("dataset.csv").iloc[:, 0].values
lstm_model = load_model("lstm_model.h5")
time_steps = 60

def forecast_lstm():
    scaler = MinMaxScaler()
    ts_scaled = scaler.fit_transform(data.reshape(-1,1))
    train_size = int(len(ts_scaled) * 0.8)
    train_scaled = ts_scaled[:train_size]
    test_scaled = ts_scaled[train_size:]
    
    history = train_scaled.tolist()
    predictions = []

    for i in range(len(test_scaled)):
        input_seq = np.array(history[-time_steps:]).reshape(1, time_steps, 1)
        yhat = lstm_model.predict(input_seq, verbose=0)[0,0]
        predictions.append(yhat)
        history.append(test_scaled[i])

    predictions_inv = scaler.inverse_transform(np.array(predictions).reshape(-1,1))
    y_test_inv = scaler.inverse_transform(test_scaled)

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(y_test_inv, label="Actual", color="blue")
    ax.plot(predictions_inv, label="LSTM Predictions", color="red")
    ax.set_title("LSTM Forecast vs Actual")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    return fig

iface = gr.Interface(
    fn=forecast_lstm,
    inputs=[],
    outputs=gr.Plot(),
    title="LSTM Forecasting",
    description="Forecast using trained LSTM model"
)

if __name__ == "__main__":
    iface.launch()
