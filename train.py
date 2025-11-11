import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# Load data
df = pd.read_csv("customer_churn.csv")
df.drop('customerID',axis='columns',inplace=True)
df = df[df.TotalCharges!=' ']
df.TotalCharges = pd.to_numeric(df.TotalCharges)

# Encode categorical columns
encoders = {}
for col in df.select_dtypes(include='object'):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save encoders
pickle.dump(encoders, open("encoder.pkl","wb"))

# Split data
X = df.drop("Churn", axis=1)
y = df["Churn"]

scaler = StandardScaler()
X = scaler.fit_transform(X)
pickle.dump(scaler, open("scaler.pkl","wb"))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Build ANN model
model = keras.Sequential([
    keras.layers.Dense(20, input_shape=(X_train.shape[1],), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50)

# Save model
model.save("model.h5")
print("✅ Model, scaler, and encoders saved successfully!")
