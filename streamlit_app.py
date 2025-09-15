import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Beer Servings Predictor", layout="centered")

st.title("Beer Servings Predictor")
st.image(
    "https://images.unsplash.com/photo-1542444459-db3bfb7aeb3b?auto=format&fit=crop&w=1200&q=60",
    use_container_width=True,   # ✅ fixed deprecation warning
)
st.write("Enter features for a country (numeric inputs) to predict beer servings.")

# Load model
model_data = joblib.load("beer_model.joblib")
model = model_data["model"]
features = model_data["features"]

# Sidebar inputs
inputs = {}
st.sidebar.header("Input features")
for feat in features:
    inputs[feat] = st.sidebar.number_input(feat, value=0.0)

X_user = pd.DataFrame([inputs])[features]
pred = model.predict(X_user)[0]

st.subheader("Predicted beer servings")
st.write(f"**{pred:.2f}**")
