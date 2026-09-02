# 📞 Customer Churn Prediction

An interactive **AI-powered customer churn prediction web app** built with **Python, TensorFlow/Keras, Scikit-learn, Pandas, and Streamlit**.

The application uses a trained **Artificial Neural Network (ANN)** to estimate whether a telecom customer is likely to churn and displays the predicted churn probability.

---

## 🚀 Live Demo

🔗 **[Open the Live App](YOUR_STREAMLIT_APP_URL)**

---

## ✨ Features

- 🧠 Neural Network / ANN churn prediction
- 📊 Churn probability score
- 🎯 Interactive customer input form
- 🔤 Saved categorical encoders for consistent preprocessing
- 📏 Saved StandardScaler for model-compatible feature scaling
- ⚡ Fast cached model and dataset loading
- 🛡️ Startup validation for required model/data files
- 💡 Retention suggestions based on the prediction
- ☁️ Streamlit Community Cloud ready

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| TensorFlow / Keras | ANN model and inference |
| Scikit-learn | Label encoding and feature scaling |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Streamlit | Interactive web application |
| Pickle | Persisting encoders and scaler |

---

## 🧠 Model Architecture

The training script builds a feed-forward ANN with:

```text
Input Features
      ↓
Dense(20) + ReLU
      ↓
Dense(15) + ReLU
      ↓
Dense(1) + Sigmoid
      ↓
Churn Probability
```

The saved model is used directly by the Streamlit application for inference.

---

## 🔄 Prediction Pipeline

```text
Customer Input
      ↓
Categorical Label Encoding
      ↓
Feature Scaling
      ↓
TensorFlow ANN
      ↓
Churn Probability
      ↓
Stay / Churn Result
```

The application uses the same saved encoders and scaler that were created during training, which keeps inference preprocessing aligned with the trained model.

---

## 📊 Dataset

The project uses the **Telco Customer Churn** dataset format. The repository contains `customer_churn.csv` with customer information such as:

- Gender
- Senior citizen status
- Partner/dependents
- Tenure
- Phone service
- Internet service
- Online security/backup
- Device protection
- Technical support
- Streaming services
- Contract type
- Paperless billing
- Payment method
- Monthly charges
- Total charges
- Churn label

The application removes the customer ID from model inputs and converts `TotalCharges` to numeric values before prediction.

---

## 📂 Project Structure

```text
Churn-Prediction/
├── app.py                 # Streamlit application
├── train.py               # Model training script
├── model.h5               # Trained TensorFlow/Keras model
├── scaler.pkl             # Fitted StandardScaler
├── encoder.pkl            # Fitted categorical LabelEncoders
├── customer_churn.csv     # Training/reference dataset
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

---

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/SIDMINUL/Churn-Prediction.git
cd Churn-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate it

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the application

```bash
streamlit run app.py
```

---

## ☁️ Streamlit Deployment

This repository is prepared for **Streamlit Community Cloud** deployment.

Use:

```text
Repository: SIDMINUL/Churn-Prediction
Branch: master
Main file: app.py
```

Make sure these files are committed to the repository before deploying:

```text
app.py
model.h5
scaler.pkl
encoder.pkl
customer_churn.csv
requirements.txt
```

Streamlit Community Cloud installs dependencies from `requirements.txt` and runs the selected Python entrypoint. citeturn0search0turn0search1

> **Note:** TensorFlow is a relatively large dependency, so the first deployment may take longer than a lightweight Streamlit application.

---

## 🧪 Retraining the Model

If you want to retrain the model using the repository dataset:

```bash
python train.py
```

This regenerates:

```text
model.h5
scaler.pkl
encoder.pkl
```

After retraining, test the Streamlit application locally before deploying the updated artifacts.

---

## 📈 Prediction Interpretation

The app uses a probability threshold of **0.50**:

```text
Probability >= 0.50  →  Likely to churn
Probability <  0.50  →  Likely to stay
```

The probability is a model output, not a guarantee of future customer behavior.

---

## ⚠️ Limitations

- The model's performance depends on the original training data and preprocessing.
- Predictions are probabilistic and should not be treated as guaranteed outcomes.
- This is a portfolio/educational implementation rather than a production customer-retention system.
- The saved `.h5` model and preprocessing artifacts must remain compatible with the installed TensorFlow/Scikit-learn versions.

---

## 🔐 Privacy

Do not upload or deploy customer datasets containing personally identifiable information without appropriate authorization and security controls.

---

## 🎯 Future Improvements

- [ ] Add model performance metrics and ROC-AUC dashboard
- [ ] Add SHAP/model explainability
- [ ] Add batch CSV prediction
- [ ] Add retention recommendation scoring
- [ ] Add model versioning
- [ ] Replace legacy `.h5` storage with a modern Keras format
- [ ] Add automated model retraining pipeline
- [ ] Add monitoring for prediction drift

---

## 👨‍💻 Author

**Abdul Momin Siddiqui**

GitHub: **[@SIDMINUL](https://github.com/SIDMINUL)**

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

## 📄 License

This project is available for educational and portfolio purposes.
