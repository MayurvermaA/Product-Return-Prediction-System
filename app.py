import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Title
st.title("Product Return Prediction System")

# Read CSV
df = pd.read_csv("product_return.csv")

# First 5 Records
st.subheader("First 5 Records")
st.write(df.head())

# Shape
st.subheader("Dataset Shape")
st.write(df.shape)

# Summary
st.subheader("Dataset Summary")
st.write(df.describe())

# Missing Values
st.subheader("Missing Values")
st.write(df.isnull().sum())

# Features and Target
X = df[['Product_Price','Delivery_Days','Customer_Rating','Purchase_Count']]
y = df['Return_Status']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.success(f"{accuracy*100:.2f}%")

st.subheader("Enter Product Details")

price = st.number_input("Product Price", min_value=500)
days = st.number_input("Delivery Days", min_value=1)
rating = st.number_input("Customer Rating", min_value=1, max_value=5)
purchase = st.number_input("Purchase Count", min_value=1)

if st.button("Predict"):

    prediction = model.predict([[price, days, rating, purchase]])

    if prediction[0] == 0:
        st.success("✅ Product Will Not Be Returned")
    else:
        st.error("🔄 Product Will Be Returned")