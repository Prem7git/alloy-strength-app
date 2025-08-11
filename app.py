import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("alloy_model_multi.pkl")
scaler = joblib.load("scaler_multi.pkl")

# Define feature names (same order as training)
features = joblib.load("features_list.pkl")

st.title("Alloy Strength Prediction App")
st.write("Enter alloy composition (% of each element) to predict Yield Strength & UTS.")

composition_input = st.text_input("Composition (e.g., Fe 98.5 C 0.2 Mn 1.3)")

if st.button("Predict"):
    input_vec = np.zeros(len(features))

    tokens = composition_input.replace(',', ' ').replace('%', '').split()
    if len(tokens) % 2 != 0:
        st.error("⚠️ Wrong format. Must be: Element value Element value ...")
    else:
        for i in range(0, len(tokens), 2):
            elem = tokens[i].capitalize()
            try:
                val = float(tokens[i+1])
            except:
                st.warning(f"Cannot parse value for {elem}")
                continue
            if elem in features:
                idx = features.index(elem)
                input_vec[idx] = val
            else:
                st.warning(f"{elem} not recognized, ignoring.")

        input_df = pd.DataFrame([input_vec], columns=features)
        scaled = scaler.transform(input_df)
        pred = model.predict(scaled)

        ys_pred, uts_pred = float(pred[0][0]), float(pred[0][1])
        st.success(f"Predicted Yield Strength: {ys_pred:.2f} MPa")
        st.success(f"Predicted Ultimate Tensile Strength: {uts_pred:.2f} MPa")
