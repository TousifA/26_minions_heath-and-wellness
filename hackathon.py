import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from joblib import load
import streamlit as st

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the pre-trained model
MODEL_PATH = "rnn_fcn_heart_anomaly_modael.keras"
SCALER_PATH = "scaler_filename.joblib"

try:
    model = load_model(MODEL_PATH)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Load the pre-saved scaler
try:
    scaler = load(SCALER_PATH)
except Exception as e:
    st.error(f"Error loading the scaler: {e}")
    st.stop()

# Function to classify ECG data
def classify_ecg(ecg_data, model, scaler):
    try:
        # Normalize the data
        ecg_data_normalized = scaler.transform(ecg_data)
        # Reshape for RNN input
        ecg_data_reshaped = ecg_data_normalized.reshape(1, ecg_data_normalized.shape[1], 1)
        # Predict class probabilities
        prediction = model.predict(ecg_data_reshaped)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None, None

# Streamlit app starts here
st.title("ECG Classification Application")
st.write("Upload ECG data or manually enter values to visualize and classify.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your ECG data file (CSV format)", type=["csv"])

# Input ECG data manually
manual_input = st.text_area("Or enter comma-separated ECG values:")

# Process input data
ecg_values = None
if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file, header=None)
        ecg_values = data.iloc[0, :].values.reshape(1, -1)
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
elif manual_input:
    try:
        ecg_values = np.array([float(x) for x in manual_input.split(",")]).reshape(1, -1)
    except ValueError:
        st.error("Invalid input. Please enter comma-separated numeric values.")

if ecg_values is not None:
    # Visualize the ECG data
    st.write("ECG Waveform")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(ecg_values[0])), ecg_values[0], label="ECG Signal")
    ax.set_title("ECG Signal")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)

    # Classify the ECG data
    predicted_class, confidence = classify_ecg(ecg_values, model, scaler)
    if predicted_class is not None:
        st.write(f"**Predicted Class**: {predicted_class}")
        st.write(f"**Confidence**: {confidence * 100:.2f}%")
        if predicted_class == 0:
            st.success("The ECG is classified as **Normal**.")
        elif predicted_class == 1:
            st.warning("The ECG indicates a **possible anomaly**. Please consult a specialist.")
        else:
            st.error("The ECG indicates a **severe anomaly**. Immediate medical attention is advised.")
else:
    st.info("Please upload an ECG file or enter values manually to proceed.")
