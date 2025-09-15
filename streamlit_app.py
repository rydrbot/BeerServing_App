import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Beer Servings Predictor", layout="centered")

st.title("Beer Servings Predictor")
st.image(
    "https://images.unsplash.com/photo-1542444459-db3bfb7aeb3b?auto=format&fit=crop&w=1200&q=60",
    width="stretch",  # ✅ updated per deprecation notice
)
st.write("Enter features (numeric inputs) to predict beer servings.")

# -----------------------
# Train model at runtime
# -----------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("beer-servings.csv")
    numeric = df.select_dtypes(include=[np.number]).dropna()

    if "beer_servings" not in numeric.columns:
        st.error("Dataset must contain 'beer_servings' column.")
        st.stop()

    X = numeric.drop(columns=["beer_servings"])
    y = numeric["beer_servings"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, X.columns.tolist()

model, features = load_model()

# Sidebar inputs
inputs = {}
st.sidebar.header("Input features")
for feat in features:
    inputs[feat] = st.sidebar.number_input(feat, value=0.0)

# Prediction button
if st.sidebar.button("Predict Beer Servings"):
    X_user = pd.DataFrame([inputs])[features]
    pred = model.predict(X_user)[0]
    st.subheader("Predicted beer servings")
    st.write(f"**{pred:.2f}**")
else:
    st.info("👈 Enter values in the sidebar and click **Predict Beer Servings** to see the result.")
