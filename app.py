from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.h5"
SCALER_PATH = BASE_DIR / "scaler.pkl"
ENCODER_PATH = BASE_DIR / "encoder.pkl"
DATA_PATH = BASE_DIR / "customer_churn.csv"

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📞", layout="wide")

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.h5 is missing from the repository.")
    if not SCALER_PATH.exists():
        raise FileNotFoundError("scaler.pkl is missing from the repository.")
    if not ENCODER_PATH.exists():
        raise FileNotFoundError("encoder.pkl is missing from the repository.")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with SCALER_PATH.open("rb") as file:
        scaler = pickle.load(file)
    with ENCODER_PATH.open("rb") as file:
        encoders = pickle.load(file)
    return model, scaler, encoders

@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError("customer_churn.csv is missing from the repository.")
    df = pd.read_csv(DATA_PATH)
    required = {"customerID", "Churn"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    df = df.drop(columns=["customerID"])
    return df

try:
    model, scaler, encoders = load_artifacts()
    df = load_dataset()
except Exception as exc:
    st.error(f"Unable to start the churn predictor: {exc}")
    st.info("Make sure model.h5, scaler.pkl, encoder.pkl, and customer_churn.csv are present in the repository.")
    st.stop()

st.sidebar.title("📞 Customer Churn Predictor")
st.sidebar.caption("Predict telecom customer churn using a trained neural network.")
st.sidebar.divider()
st.sidebar.metric("Training records", f"{len(df):,}")
st.sidebar.metric("Features", f"{len(df.columns) - 1}")

st.title("📞 Customer Churn Prediction")
st.markdown("Enter customer details below to estimate the probability that the customer will churn.")

with st.expander("About this model"):
    st.write(
        "This app uses a trained TensorFlow/Keras artificial neural network (ANN). "
        "Categorical values are transformed with the saved LabelEncoders and the complete feature vector "
        "is standardized with the saved StandardScaler before prediction."
    )

st.subheader("👤 Customer Details")

with st.form("prediction_form"):
    input_data = {}
    input_columns = [col for col in df.columns if col != "Churn"]
    cols = st.columns(3)

    for i, col in enumerate(input_columns):
        with cols[i % 3]:
            if col in encoders:
                options = list(encoders[col].classes_)
                input_data[col] = st.selectbox(col, options, key=f"input_{col}")
            else:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                minimum = float(series.min())
                maximum = float(series.max())
                default = float(series.median())
                if minimum == maximum:
                    input_data[col] = st.number_input(col, value=default, key=f"input_{col}")
                else:
                    input_data[col] = st.number_input(
                        col,
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        key=f"input_{col}",
                    )

    submitted = st.form_submit_button("🔮 Predict Churn", type="primary", use_container_width=True)

if submitted:
    with st.spinner("Analyzing customer data..."):
        time.sleep(0.5)
        user_df = pd.DataFrame([input_data], columns=input_columns)

        try:
            for col, encoder in encoders.items():
                if col != "Churn" and col in user_df.columns:
                    user_df[col] = encoder.transform(user_df[col].astype(str))

            user_scaled = scaler.transform(user_df)
            probability = float(model.predict(user_scaled, verbose=0)[0][0])
        except ValueError as exc:
            st.error(f"Invalid input: {exc}")
            st.stop()
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

    churn = probability >= 0.5

    st.divider()
    st.subheader("📊 Prediction Result")
    result_col, score_col = st.columns(2)

    with result_col:
        if churn:
            st.error("⚠️ Customer is likely to CHURN")
            st.write("Consider targeted retention offers, proactive support, or a plan review.")
        else:
            st.success("✅ Customer is likely to STAY")
            st.write("The model estimates a lower probability of churn for this customer.")

    with score_col:
        st.metric("Churn probability", f"{probability:.1%}")
        st.progress(probability)

    with st.expander("Prediction input"):
        st.dataframe(user_df, use_container_width=True)

st.divider()
st.caption("For educational and portfolio demonstration purposes. Model predictions should not be treated as business decisions without validation.")
