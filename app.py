import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load the trained LSTM model (from local file)
lstm_model = load_model("models/lstm_model.h5")

time_steps = 60
scaler = MinMaxScaler(feature_range=(0,1))

def forecast_plot(csv_file):
    try:
        # Read uploaded CSV
        df = pd.read_csv(csv_file.name)
        
        # Assume the time series column is 'Value' or first column
        if 'Value' in df.columns:
            ts = df['Value'].values
        else:
            ts = df.iloc[:,0].values
        
        ts_scaled = scaler.fit_transform(ts.reshape(-1,1))
        
        # Prepare sequences for prediction
        X_input = []
        for i in range(len(ts_scaled)-time_steps, len(ts_scaled)):
            X_input.append(ts_scaled[i-time_steps:i])
        X_input = np.array(X_input)
        
        # Predict next values
        preds_scaled = lstm_model.predict(X_input)
        preds = scaler.inverse_transform(preds_scaled)
        
        # Plot actual vs predicted
        plt.figure(figsize=(10,5))
        plt.plot(ts[-len(preds):], label="Actual", color='blue')
        plt.plot(preds.flatten(), label="Predicted", color='red', linestyle='--')
        plt.title("LSTM Forecast vs Actual")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        
        # Save plot to return
        plt.tight_layout()
        plot_path = "forecast_plot.png"
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio interface
iface = gr.Interface(
    fn=forecast_plot,
    inputs=gr.File(label="Upload CSV"),
    outputs=gr.Image(label="Forecast Plot"),
    title="LSTM Time Series Forecast",
    description="Upload a CSV file with a single column of time series values. The app will forecast the next values and plot predicted vs actual."
)

iface.launch()
