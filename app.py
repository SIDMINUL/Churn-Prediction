from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "customer_churn.csv"

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📞", layout="wide")


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError("customer_churn.csv is missing from the repository.")

    df = pd.read_csv(DATA_PATH)
    required = {"customerID", "Churn"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(sorted(missing))}"
        )

    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop(columns=["customerID"])
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    df = df.dropna(subset=["Churn"])
    return df


@st.cache_resource
def train_model(data):
    X = data.drop(columns=["Churn"])
    y = data["Churn"].astype(int)

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [
        col for col in X.columns if col not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=2000, solver="liblinear"),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    return model, metrics, numeric_features, categorical_features


try:
    df = load_dataset()
    model, metrics, numeric_features, categorical_features = train_model(df)
except Exception as exc:
    st.error(f"Unable to start the churn predictor: {exc}")
    st.stop()

st.sidebar.title("📞 Customer Churn Predictor")
st.sidebar.caption("Predict telecom customer churn with a machine-learning model.")
st.sidebar.divider()
st.sidebar.metric("Training records", f"{len(df):,}")
st.sidebar.metric("Model accuracy", f"{metrics['accuracy']:.1%}")
st.sidebar.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

st.title("📞 Customer Churn Prediction")
st.markdown(
    "Enter customer details below to estimate the probability that the customer will churn."
)

with st.expander("About this model"):
    st.write(
        "The app trains a Logistic Regression model on the complete customer dataset "
        "at startup. Numeric features are imputed and standardized, while categorical "
        "features are one-hot encoded. This avoids treating categories such as contract "
        "type or internet service as arbitrary numbers."
    )

st.subheader("👤 Customer Details")

input_columns = [col for col in df.columns if col != "Churn"]

with st.form("prediction_form"):
    input_data = {}
    cols = st.columns(3)

    for i, col in enumerate(input_columns):
        with cols[i % 3]:
            if col in categorical_features:
                options = sorted(df[col].dropna().astype(str).unique().tolist())
                input_data[col] = st.selectbox(col, options, key=f"input_{col}")
            else:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                minimum = float(series.min())
                maximum = float(series.max())
                default = float(series.median())

                if minimum == maximum:
                    input_data[col] = st.number_input(
                        col, value=default, key=f"input_{col}"
                    )
                else:
                    input_data[col] = st.number_input(
                        col,
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        key=f"input_{col}",
                    )

    submitted = st.form_submit_button(
        "🔮 Predict Churn", type="primary", use_container_width=True
    )

if submitted:
    with st.spinner("Analyzing customer data..."):
        user_df = pd.DataFrame([input_data], columns=input_columns)

        try:
            probability = float(model.predict_proba(user_df)[0, 1])
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
            st.write(
                "The model estimates a higher churn probability. "
                "Consider targeted retention offers or proactive support."
            )
        else:
            st.success("✅ Customer is likely to STAY")
            st.write(
                "The model estimates a lower churn probability for this customer."
            )

    with score_col:
        st.metric("Churn probability", f"{probability:.1%}")
        st.progress(probability)

    with st.expander("Prediction input"):
        st.dataframe(user_df, use_container_width=True)

st.divider()
st.caption(
    "For educational and portfolio demonstration purposes. Model predictions should not "
    "be treated as business decisions without validation."
)
