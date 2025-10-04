import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load LSTM model
lstm_model = load_model("lstm_model.h5")

time_steps = 60

def create_sequences(data, time_steps=time_steps):
    X = []
    for i in range(len(data)-time_steps):
        X.append(data[i:i+time_steps])
    return np.array(X)

def forecast(csv_file=None):
    # Use default dataset if no file uploaded
    if csv_file is None:
        df = pd.read_csv("dataset.csv")  # your default CSV in repo
    else:
        df = pd.read_csv(csv_file.name)
    
    ts = df.iloc[:, 0].values  # assuming first column is the series
    ts = ts.reshape(-1,1)
    
    scaler = MinMaxScaler(feature_range=(0,1))
    ts_scaled = scaler.fit_transform(ts)
    
    # Prepare sequences for prediction
    X_input = create_sequences(ts_scaled)
    y_pred_scaled = lstm_model.predict(X_input)
    
    # Inverse transform predictions
    y_pred = scaler.inverse_transform(y_pred_scaled)
    
    # Combine with actual values for plotting
    actual = ts[time_steps:]
    
    # Return DataFrame for plotting
    result_df = pd.DataFrame({
        "Actual": actual.flatten(),
        "Predicted": y_pred.flatten()
    })
    return result_df

# Gradio interface
iface = gr.Interface(
    fn=forecast,
    inputs=gr.File(file_types=[".csv"], label="Upload CSV (optional)"),
    outputs=gr.LinePlot(x="index", y=["Actual","Predicted"], labels={"Actual":"Actual","Predicted":"Predicted"}, title="LSTM Forecast vs Actual"),
    live=True,
    description="Upload a CSV to forecast or use default dataset"
)

iface.launch()
