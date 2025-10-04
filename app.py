import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Load LSTM model
lstm_model = load_model("lstm_model.h5")

time_steps = 60  # same as training

def create_sequences(data, time_steps=time_steps):
    X, y = [], []
    for i in range(len(data)-time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

def forecast_lstm(file):
    # Read CSV
    df = pd.read_csv(file)
    
    if df.shape[1] > 1:
        ts = df.iloc[:, 1]  # assume second column is values
    else:
        ts = df.iloc[:, 0]

    ts_values = ts.values.reshape(-1,1)
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0,1))
    ts_scaled = scaler.fit_transform(ts_values)
    
    # Split into train/test (80% train)
    train_size = int(len(ts_scaled)*0.8)
    train_scaled = ts_scaled[:train_size]
    test_scaled = ts_scaled[train_size:]
    
    # Forecast
    history = train_scaled.tolist()
    predictions = []

    for i in range(len(test_scaled)):
        input_seq = np.array(history[-time_steps:]).reshape(1, time_steps, 1)
        yhat = lstm_model.predict(input_seq, verbose=0)[0,0]
        predictions.append(yhat)
        history.append(test_scaled[i])

    # Inverse scale
    predictions_inv = scaler.inverse_transform(np.array(predictions).reshape(-1,1))
    y_test_inv = scaler.inverse_transform(test_scaled)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
    mape = mean_absolute_percentage_error(y_test_inv, predictions_inv)

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(range(train_size, len(ts_scaled)), y_test_inv, label='Actual', color='blue')
    plt.plot(range(train_size, len(ts_scaled)), predictions_inv, label='LSTM Predictions', color='red', linestyle='--')
    plt.title("LSTM Forecast vs Actual")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # Save plot to return
    plt.savefig("/tmp/forecast_plot.png")
    plt.close()

    return "/tmp/forecast_plot.png", f"RMSE: {rmse:.4f}, MAPE: {mape:.4f}"

# Gradio interface
iface = gr.Interface(
    fn=forecast_lstm,
    inputs=gr.File(label="Upload CSV"),
    outputs=[gr.Image(label="Forecast Plot"), gr.Textbox(label="Metrics")],
    title="LSTM Time Series Forecast",
    description="Upload a CSV file with one column of values to forecast using LSTM."
)

iface.launch()
