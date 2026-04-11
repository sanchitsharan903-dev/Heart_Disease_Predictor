import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import base64

def get_base64(image_file):
    with open(image_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64("heart.jpg")
st.markdown(f"""
<style>

/* FULL BACKGROUND IMAGE (CLEAR + VISIBLE) */
[data-testid="stAppViewContainer"] {{
    background: 
        linear-gradient(rgba(0, 0, 0, 0.45), rgba(0, 0, 0, 0.45)),
        url("data:image/png;base64,{img_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

/* SIDEBAR */
[data-testid="stSidebar"] {{
    background: rgba(10, 15, 25, 0.9);
}}

/* GLASS EFFECT MAIN PANEL */
.block-container {{
    background: rgba(0, 0, 0, 0.35);
    padding: 2rem;
    border-radius: 15px;
    backdrop-filter: blur(6px);
}}

/* TEXT */
h1, h2, h3, p, label {{
    color: #f1f5f9 !important;
}}

/* BUTTON */
.stButton > button {{
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    font-weight: bold;
}}

/* INPUT */
.stNumberInput input, .stSelectbox div {{
    background-color: rgba(0,0,0,0.6) !important;
    color: white !important;
    border-radius: 8px;
}}

</style>
""", unsafe_allow_html=True)
# load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("❤️ Heart Disease Predictor (Based on Machine Learning)")

# -------- INPUT SECTION -------- #

age = st.number_input("Age", 1, 120, 30)

sex_dict = {"Female": 0, "Male": 1}
sex = st.selectbox("Sex", list(sex_dict.keys()))
sex = sex_dict[sex]

cp_dict = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}
cp = st.selectbox("Chest Pain Type", list(cp_dict.keys()))
cp = cp_dict[cp]

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
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", list(fbs_dict.keys()))
fbs = fbs_dict[fbs]

restecg_dict = {
    "Normal": 0,
    "ST-T Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}
restecg = st.selectbox("Rest ECG", list(restecg_dict.keys()))
restecg = restecg_dict[restecg]

thalach = st.number_input("Max Heart Rate", 60, 220, 150)

exang_dict = {"No": 0, "Yes": 1}
exang = st.selectbox("Exercise Induced Angina", list(exang_dict.keys()))
exang = exang_dict[exang]

oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0)

slope_dict = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
slope = st.selectbox("Slope", list(slope_dict.keys()))
slope = slope_dict[slope]

ca = st.selectbox("Number of Major Vessels", [0,1,2,3])

thal_dict = {
    "Normal": 0,
    "Fixed Defect": 1,
    "Reversible Defect": 2
}
thal = st.selectbox("Thalassemia", list(thal_dict.keys()))
thal = thal_dict[thal]

# -------- PREDICTION -------- #

if st.button("Predict"):

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    # scale
    input_data = scaler.transform(input_data)

    result = model.predict(input_data)
    prob = model.predict_proba(input_data)

    risk = float(prob[0][1]) * 100
    confidence = max(prob[0]) * 100

    st.subheader(f"🧠 Risk: {risk:.2f}%")
    st.write(f"Model Confidence: {confidence:.2f}%")
    

    # risk bar
    st.progress(risk/100)

    # risk category
    if risk < 30:
        st.success("Low Risk 🟢")
    elif risk < 60:
        st.warning("Moderate Risk 🟡")
    else:
        st.error("High Risk 🔴")

    # ---------------- USER INPUT GRAPH ---------------- #
    import pandas as pd

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

    # -------- HEALTH INSIGHTS -------- #

    st.subheader("📋 Health Insights")

    if trestbps > 140:
        st.warning("High Blood Pressure ⚠️")
    else:
        st.success("Blood Pressure Normal ✅")

    if thalach < 100:
        st.warning("Low Heart Rate ⚠️")
    else:
        st.success("Heart Rate Normal ✅")

    # -------- SUGGESTIONS -------- #

    st.subheader("💡 Health Suggestions")

    if chol > 240:
        st.write("➡️ Reduce oily food & exercise regularly")

    if trestbps > 140:
        st.write("➡️ Reduce salt intake")

    if risk > 60:
        st.write("➡️ Consult a cardiologist immediately")

    # -------- HISTORY -------- #

    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append(round(risk,2))

    st.subheader("📜 Prediction History")
    st.write(st.session_state.history)

    # -------- FEATURE IMPORTANCE -------- #

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
    st.markdown("""
<hr style="margin-top: 50px; border: 1px solid rgba(255,255,255,0.2);">

<p style='text-align: center; font-size: 14px; color: #94a3b8;'>
🚀 Built with Machine Learning | Developed by <b style="color:#00c6ff;">Sachin Chaudhary & Sanchit Sharan</b>
</p>
""", unsafe_allow_html=True)
