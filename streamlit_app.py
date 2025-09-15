import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Alcohol Consumption Predictor", layout="centered")

st.title("Alcohol Consumption Predictor")

st.image(
    "https://friendsofglass.com/wp-content/uploads/2024/08/why-does-beer-taste-better-in-glass.png",
    width="stretch",
)

st.write("Enter features (numeric inputs) to predict **total litres of pure alcohol**.")

# -----------------------
# Train model at runtime
# -----------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("beer-servings.csv")

    # Drop non-feature columns if present
    drop_cols = [col for col in ["Unnamed: 0", "country", "continent"] if col in df.columns]
    df = df.drop(columns=drop_cols)

    numeric = df.select_dtypes(include=[np.number]).dropna()

    if "total_litres_of_pure_alcohol" not in numeric.columns:
        st.error("Dataset must contain 'total_litres_of_pure_alcohol' column.")
        st.stop()

    # Target is total_litres_of_pure_alcohol
    X = numeric.drop(columns=["total_litres_of_pure_alcohol"])
    y = numeric["total_litres_of_pure_alcohol"]

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
if st.sidebar.button("Predict Total Litres of Pure Alcohol"):
    X_user = pd.DataFrame([inputs])[features]
    pred = model.predict(X_user)[0]
    st.subheader("Predicted total litres of pure alcohol")
    st.write(f"**{pred:.2f} litres**")
else:
    st.info("👈 Enter values in the sidebar and click **Predict** to see the result.")
