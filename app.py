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
}}

.block-container {{
    background: rgba(0, 0, 0, 0.35);
    padding: 2rem;
    border-radius: 15px;
}}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ---------- #
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------- TITLE ---------- #
st.title("❤️ Heart Disease Predictor (Advanced AI System)")

# ---------- INPUT ---------- #
age = st.number_input("Age", 1, 120, 30)

sex = {"Female": 0, "Male": 1}[st.selectbox("Sex", ["Female","Male"])]

cp_labels = ["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"]
cp = cp_labels.index(st.selectbox("Chest Pain Type", cp_labels))

trestbps = st.number_input("Resting BP", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)

if chol < 200:
    st.success("Cholesterol Normal ✅")
elif chol < 240:
    st.warning("Borderline ⚠️")
else:
    st.error("High Cholesterol ❌")

fbs = {"No":0,"Yes":1}[st.selectbox("FBS >120",["No","Yes"])]

restecg_labels = ["Normal","ST-T abnormality","LVH"]
restecg = restecg_labels.index(st.selectbox("Rest ECG",restecg_labels))

thalach = st.number_input("Max Heart Rate",60,220,150)

exang = {"No":0,"Yes":1}[st.selectbox("Exercise Angina",["No","Yes"])]

oldpeak = st.number_input("Oldpeak",0.0,6.0,1.0)

slope_labels = ["Upsloping","Flat","Downsloping"]
slope = slope_labels.index(st.selectbox("Slope",slope_labels))

ca = st.selectbox("Major Vessels",[0,1,2,3])

thal_labels = ["Normal","Fixed Defect","Reversible Defect"]
thal = thal_labels.index(st.selectbox("Thal",thal_labels))

# ---------- PREDICT ---------- #
if st.button("Predict"):

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    input_data = scaler.transform(input_data)

    prob = model.predict_proba(input_data)

    risk = float(prob[0][1]) * 100
    confidence = float(max(prob[0])) * 100

    # ---------- RESULT ---------- #
    st.subheader(f"🧠 Risk: {risk:.2f}%")
    st.write(f"Confidence: {confidence:.2f}%")

    st.progress(float(risk)/100)

    if risk < 30:
        st.success("Low Risk 🟢")
    elif risk < 60:
        st.warning("Moderate Risk 🟡")
    else:
        st.error("High Risk 🔴")

    # ---------- COMPARISON GRAPHS ---------- #
    st.subheader("🧪 Health Comparison")

    st.write("### Cholesterol")
    st.bar_chart(pd.DataFrame({"Value":[chol,200]}, index=["You","Normal"]))

    st.write("### Blood Pressure")
    st.bar_chart(pd.DataFrame({"Value":[trestbps,120]}, index=["You","Normal"]))

    st.write("### Heart Rate")
    st.bar_chart(pd.DataFrame({"Value":[thalach,100]}, index=["You","Normal"]))

    st.write("### Oldpeak")
    st.bar_chart(pd.DataFrame({"Value":[oldpeak,1]}, index=["You","Normal"]))

    # ---------- HEALTH SUGGESTIONS ---------- #
    st.subheader("💡 Health Suggestions")

    suggestion_flag = False

    if chol > 240:
        st.warning("High cholesterol → reduce oily food")
        suggestion_flag = True

    if trestbps > 140:
        st.warning("High BP → reduce salt")
        suggestion_flag = True

    if thalach < 100:
        st.warning("Low heart rate → improve fitness")
        suggestion_flag = True

    if oldpeak > 2:
        st.warning("Heart stress detected")
        suggestion_flag = True

    if cp == 3:
        st.warning("Asymptomatic chest pain → serious risk")
        suggestion_flag = True

    if exang == 1:
        st.warning("Exercise angina detected")
        suggestion_flag = True

    if ca > 1:
        st.warning("Blocked vessels detected")
        suggestion_flag = True

    if thal != 0:
        st.warning("Abnormal thal result")
        suggestion_flag = True

    if risk > 60:
        st.error("Consult cardiologist immediately 🚨")
        suggestion_flag = True

    if not suggestion_flag:
        st.success("All parameters look good ✅ Maintain healthy lifestyle")

    # ---------- HISTORY ---------- #
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append(round(risk,2))

    st.subheader("📜 Prediction History")
    st.write(st.session_state.history)

    # ---------- FEATURE IMPORTANCE ---------- #
    st.subheader("📊 Feature Importance")

    features = ["age","sex","cp","trestbps","chol","fbs",
                "restecg","thalach","exang","oldpeak",
                "slope","ca","thal"]

    fig, ax = plt.subplots()
    ax.barh(features, model.feature_importances_)
    st.pyplot(fig)

    # ---------- FOOTER ---------- #
    st.markdown("""
<hr style="margin-top: 50px;">
<p style='text-align:center;'>Developed by <b>Sanchit Sharan</b></p>
""", unsafe_allow_html=True)
