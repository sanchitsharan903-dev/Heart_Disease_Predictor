import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import base64
import pandas as pd

# ---------- BACKGROUND IMAGE ---------- #
def get_base64(image_file):
    with open(image_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64("heart.jpg")

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: 
        linear-gradient(rgba(0, 0, 0, 0.45), rgba(0, 0, 0, 0.45)),
        url("data:image/png;base64,{img_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

[data-testid="stSidebar"] {{
    background: rgba(10, 15, 25, 0.9);
}}

.block-container {{
    background: rgba(0, 0, 0, 0.35);
    padding: 2rem;
    border-radius: 15px;
    backdrop-filter: blur(6px);
}}

h1, h2, h3, p, label {{
    color: #f1f5f9 !important;
}}

.stButton > button {{
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    font-weight: bold;
}}

.stNumberInput input, .stSelectbox div {{
    background-color: rgba(0,0,0,0.6) !important;
    color: white !important;
    border-radius: 8px;
}}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ---------- #
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------- TITLE ---------- #
st.title("❤️ Heart Disease Predictor (Advanced AI System)")

# ---------- INPUT SECTION ---------- #
age = st.number_input("Age", 1, 120, 30)

sex_dict = {"Female": 0, "Male": 1}
sex = sex_dict[st.selectbox("Sex", list(sex_dict.keys()))]

cp_dict = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}
cp = cp_dict[st.selectbox("Chest Pain Type", list(cp_dict.keys()))]

trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)

chol = st.number_input("Cholesterol (mg/dl)", 100, 400, 200)

# Cholesterol interpretation
if chol < 200:
    st.success("Cholesterol: Normal ✅")
elif chol < 240:
    st.warning("Cholesterol: Borderline ⚠️")
else:
    st.error("Cholesterol: High ❌")

fbs_dict = {"No": 0, "Yes": 1}
fbs = fbs_dict[st.selectbox("Fasting Blood Sugar > 120 mg/dl", list(fbs_dict.keys()))]

restecg_dict = {
    "Normal": 0,
    "ST-T Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}
restecg = restecg_dict[st.selectbox("Rest ECG", list(restecg_dict.keys()))]

thalach = st.number_input("Max Heart Rate", 60, 220, 150)

exang_dict = {"No": 0, "Yes": 1}
exang = exang_dict[st.selectbox("Exercise Induced Angina", list(exang_dict.keys()))]

oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0)

slope_dict = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
slope = slope_dict[st.selectbox("Slope", list(slope_dict.keys()))]

ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])

thal_dict = {
    "Normal": 0,
    "Fixed Defect": 1,
    "Reversible Defect": 2
}
thal = thal_dict[st.selectbox("Thalassemia", list(thal_dict.keys()))]

# ---------- PREDICTION ---------- #
if st.button("Predict"):

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    input_data = scaler.transform(input_data)

    result = model.predict(input_data)
    prob = model.predict_proba(input_data)

    risk = float(prob[0][1]) * 100
    confidence = float(max(prob[0])) * 100

    # ---------- RESULT ---------- #
    st.subheader(f"🧠 Risk: {risk:.2f}%")
    st.write(f"Model Confidence: {confidence:.2f}%")

    # ---------- PROGRESS ---------- #
    st.progress(float(risk)/100)

    if risk < 30:
        st.success("Low Risk 🟢")
    elif risk < 60:
        st.warning("Moderate Risk 🟡")
    else:
        st.error("High Risk 🔴")

    # ---------- USER GRAPH ---------- #
    st.subheader("📈 Your Health Data")

    input_data_dict = {
        "Age": age,
        "BP": trestbps,
        "Cholesterol": chol,
        "Max HR": thalach,
        "Oldpeak": oldpeak
    }

    df_input = pd.DataFrame.from_dict(input_data_dict, orient='index', columns=['Value'])
    st.bar_chart(df_input)

    # ---------- HEALTH INSIGHTS ---------- #
    st.subheader("📋 Health Insights")

    if trestbps > 140:
        st.warning("High Blood Pressure ⚠️")
    else:
        st.success("Blood Pressure Normal ✅")

    if thalach < 100:
        st.warning("Low Heart Rate ⚠️")
    else:
        st.success("Heart Rate Normal ✅")

    # ---------- SUGGESTIONS ---------- #
    st.subheader("💡 Health Suggestions")

    if chol > 240:
        st.write("➡️ Reduce oily food & exercise regularly")

    if trestbps > 140:
        st.write("➡️ Reduce salt intake")

    if risk > 60:
        st.write("➡️ Consult a cardiologist immediately")

    # ---------- HISTORY ---------- #
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append(round(risk, 2))

    st.subheader("📜 Prediction History")
    st.write(st.session_state.history)

    # ---------- FEATURE IMPORTANCE ---------- #
    st.subheader("📊 Feature Importance")

    features = [
        "age","sex","cp","trestbps","chol","fbs",
        "restecg","thalach","exang","oldpeak",
        "slope","ca","thal"
    ]

    importance = model.feature_importances_

    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(features, importance)
    ax.set_title("Feature Importance")

    st.pyplot(fig)

    # ---------- FOOTER ---------- #
    st.markdown("""
<hr style="margin-top: 50px; border: 1px solid rgba(255,255,255,0.2);">

<p style='text-align: center; font-size: 14px; color: #94a3b8;'>
🚀 Built with Machine Learning | Developed by <b style="color:#00c6ff;">Sanchit Sharan</b>
</p>
""", unsafe_allow_html=True)
