import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import time

# Page Config
st.set_page_config(page_title="Churn Prediction", page_icon="📞", layout="wide")

# Load model & tools
model = tf.keras.models.load_model("model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoder.pkl", "rb"))

# Sidebar UI
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/942/942799.png", width=80)
st.sidebar.title("Customer Churn Predictor")
st.sidebar.markdown("""
🔍 Predict whether a telecom customer will churn or not using AI.  
Fill in the details and get instant results!
""")
st.sidebar.divider()
st.sidebar.markdown("Developed with ❤️ using **Streamlit & TensorFlow**")

# Title
st.title("📞 Customer Churn Prediction")
st.markdown("Enter customer details below to check churn probability")

# Load dataset
df = pd.read_csv("customer_churn.csv")
df = df[df.TotalCharges != ' ']
df.TotalCharges = pd.to_numeric(df.TotalCharges)
df.drop("customerID", axis=1, inplace=True)

# Form UI
with st.form("prediction_form"):
    cols = st.columns(3)
    input_data = {}
    i = 0
    for col in df.columns:
        if col == "Churn":
            continue
        if df[col].dtype == "object":
            options = list(encoders[col].classes_)
            input_data[col] = cols[i % 3].selectbox(col, options)
        else:
            input_data[col] = cols[i % 3].number_input(col, float(df[col].min()), float(df[col].max()))
        i += 1

    submit = st.form_submit_button("🔮 Predict Churn")

# Prediction
if submit:
    with st.spinner("Analyzing customer data..."):
        time.sleep(1.5)

    user_df = pd.DataFrame([input_data])

    # encoding
    for col, encoder in encoders.items():
        if col != "Churn" and col in user_df:
            user_df[col] = encoder.transform(user_df[col])


    # scaling
    user_df = scaler.transform(user_df)

    prediction = model.predict(user_df)[0][0]
    churn = prediction > 0.5

    # Result UI
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Prediction Result")
        if churn:
            st.error("⚠ Customer is likely to CHURN!")
        else:
            st.success("✅ Customer is likely to STAY!")

    with col2:
        st.subheader("📊 Churn Probability")
        st.progress(float(prediction))
        st.write(f"**Score:** `{prediction:.2f}`")

    # Extra message
    if churn:
        st.warning("Consider giving special offers or better customer support to retain this user!")
    else:
        st.balloons()
        st.info("This customer seems satisfied. Keep up the good service! 🎉")
