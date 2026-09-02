# 📞 Customer Churn Prediction

An interactive **Customer Churn Prediction** web app built with **Python, Streamlit, Pandas, Scikit-learn, and Logistic Regression**.

The application analyzes telecom customer information and estimates the probability that a customer will leave the service.

## 🚀 Live Demo

**Streamlit:** https://churn-prediction0808.streamlit.app/

## ✨ Features

- 📊 Customer churn probability prediction
- 🤖 Logistic Regression machine-learning model
- 🔤 One-hot encoding for categorical variables
- 📏 Standardization of numeric features
- 🧹 Automatic handling of missing `TotalCharges` values
- 📈 Accuracy and ROC-AUC shown in the sidebar
- 🎯 Stratified train/test evaluation
- 🖥️ Interactive Streamlit interface

## 🧠 Why the model was improved

The earlier version encoded every categorical column with `LabelEncoder` and then standardized the resulting integer values. That can incorrectly imply an ordered relationship between categories such as contract type or internet service.

The current version uses a proper preprocessing pipeline:

```text
Raw customer data
       ↓
Missing-value handling
       ↓
Numeric → StandardScaler
Categorical → OneHotEncoder
       ↓
Logistic Regression
       ↓
Churn probability
```

The app trains the model from the available customer dataset at startup, so the deployed application does not depend on an incompatible TensorFlow runtime or an outdated saved neural-network artifact.

## 📊 Dataset

The project uses the **Telco Customer Churn** dataset containing telecom customer information such as:

- Demographics
- Tenure
- Contract type
- Internet service
- Payment method
- Monthly charges
- Total charges
- Churn status

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Logistic Regression**
- **OneHotEncoder**
- **StandardScaler**

## ▶️ Run Locally

```bash
git clone https://github.com/SIDMINUL/Churn-Prediction.git
cd Churn-Prediction
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure

```text
Churn-Prediction/
├── app.py
├── train.py
├── customer_churn.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## 📌 Notes

This project is intended for **educational and portfolio demonstration purposes**. Predictions should be validated against real business data before being used for operational decisions.
