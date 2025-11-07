import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Patient Readmission Risk Predictor",
    page_icon="⚕️",
    layout="centered"
)

# ✅ Load Model
@st.cache_resource
def load_model():
    saved = joblib.load("models/trained_model.pkl")
    return saved["preprocessor"], saved["model"]

preprocessor, model = load_model()

# ✅ Page Title
st.title("🏥 Patient Readmission Risk Predictor")
st.write("Predict the probability that a patient will be readmitted within 30 days.")

st.divider()

# ✅ Load feature template
@st.cache_data
def load_feature_template():
    df = pd.read_csv("data/readmission.csv").drop(columns=["readmitted"])
    return df

template_df = load_feature_template()

# ✅ Create User Input Form
st.subheader("📝 Enter Patient Details")

cols = st.columns(2)

input_data = {}

for idx, col_name in enumerate(template_df.columns):
    if template_df[col_name].dtype == "object":
        unique_vals = template_df[col_name].dropna().unique().tolist()
        if len(unique_vals) == 0:
            unique_vals = ["Unknown"]
        input_data[col_name] = cols[idx % 2].selectbox(
            col_name.replace("_", " ").title(), unique_vals
        )
    else:
        default_val = (
            float(template_df[col_name].mean()) 
            if not np.isnan(template_df[col_name].mean()) 
            else 0.0
        )
        input_data[col_name] = cols[idx % 2].number_input(
            col_name.replace("_", " ").title(),
            value=default_val
        )

user_df = pd.DataFrame([input_data])

st.divider()

# ✅ Prediction Button
if st.button("Predict Readmission Risk"):
    try:
        # Transform input + predict
        X_processed = preprocessor.transform(user_df)
        pred = model.predict(X_processed)[0]
        proba = model.predict_proba(X_processed)[0][1]

        # ✅ Risk Categorization
        if proba >= 0.70:
            risk = "🔴 HIGH RISK"
            color = "red"
        elif proba >= 0.40:
            risk = "🟠 MEDIUM RISK"
            color = "orange"
        else:
            risk = "🟢 LOW RISK"
            color = "green"

        # ✅ Display Results
        st.subheader("📊 Prediction Result")
        st.markdown(f"### **Risk Category:** <span style='color:{color};'>{risk}</span>", unsafe_allow_html=True)

        st.write(f"### Probability of 30-day Readmission: **{proba:.2%}**")

        st.progress(float(proba))

        st.info(
            "⚕️ *Interpretation:* A higher probability means the patient is more likely to return within 30 days. "
            "Doctors should review medication, follow-up appointments, and discharge instructions."
        )

    except Exception as e:
        st.error("An error occurred during prediction.")
        st.exception(e)

# ✅ Footer
st.divider()
st.caption("Built with ❤️ using Streamlit & XGBoost.")
