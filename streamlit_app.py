import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Beer Servings Predictor", layout="centered")

st.title("Beer Servings Predictor")
st.image("https://images.unsplash.com/photo-1542444459-db3bfb7aeb3b?auto=format&fit=crop&w=1200&q=60", use_column_width=True)
st.write("Enter features for a country (numeric inputs) to predict beer servings.")

model_data = joblib.load("beer_model.joblib")
model = model_data["model"]
features = model_data["features"]

inputs = {}
st.sidebar.header("Input features")
for feat in features:
    default = 0.0
    inputs[feat] = st.sidebar.number_input(feat, value=float(default))

X_user = pd.DataFrame([inputs])[features]
pred = model.predict(X_user)[0]
st.subheader("Predicted beer servings")
st.write(f"**{pred:.2f}**")