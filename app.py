from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
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
st.title("❤️ Heart Disease Predictor (AI Powered)")
patient_name = st.text_input("👤 Patient Name")

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

    if patient_name.strip() == "":
        st.warning("Please enter patient name")
        st.stop()

    st.subheader(f"👤 Patient: {patient_name}")

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
    st.progress(risk/100)

    if risk < 30:
        category = "LOW"
        st.success("Low Risk 🟢")
    elif risk < 60:
        category = "MODERATE"
        st.warning("Moderate Risk 🟡")
    else:
        category = "HIGH"
        st.error("High Risk 🔴")

    # ---------- RISK BREAKDOWN ---------- #
    st.subheader("📊 Risk Breakdown")

    risk_factors = {
        "Cholesterol": chol / 400,
        "Blood Pressure": trestbps / 200,
        "Heart Rate": 1 - (thalach / 220),
        "Oldpeak": oldpeak / 6,
        "Blocked Vessels": ca / 3
    }

    risk_df = pd.DataFrame(list(risk_factors.items()), columns=["Factor", "Impact"])
    st.bar_chart(risk_df.set_index("Factor"))

    # ---------- COMPARISON GRAPHS ---------- #
    st.subheader("🧪 Health Comparison")

    st.bar_chart(pd.DataFrame({"Value":[chol,200]}, index=["You","Normal Chol"]))
    st.bar_chart(pd.DataFrame({"Value":[trestbps,120]}, index=["You","Normal BP"]))
    st.bar_chart(pd.DataFrame({"Value":[thalach,100]}, index=["You","Normal HR"]))
    st.bar_chart(pd.DataFrame({"Value":[oldpeak,1]}, index=["You","Normal Oldpeak"]))

    # ---------- HEALTH SUGGESTIONS ---------- #
    st.subheader("💡 Health Suggestions")

    suggestions = []

    if chol > 240:
        st.warning("High cholesterol → reduce oily food")
        suggestions.append("Reduce oily and fatty food")

    if trestbps > 140:
        st.warning("High BP → reduce salt")
        suggestions.append("Reduce salt intake")

    if thalach < 100:
        st.warning("Low heart rate → improve fitness")
        suggestions.append("Improve cardiovascular fitness")

    if oldpeak > 2:
        st.warning("Heart stress detected")
        suggestions.append("Possible heart stress")

    if cp == 3:
        st.warning("Asymptomatic chest pain → serious risk")
        suggestions.append("Chest pain risk detected")

    if exang == 1:
        st.warning("Exercise angina detected")
        suggestions.append("Exercise-induced angina")

    if ca > 1:
        st.warning("Blocked vessels detected")
        suggestions.append("Blocked vessels")

    if thal != 0:
        st.warning("Abnormal thal result")
        suggestions.append("Abnormal thalassemia")

    if risk > 60:
        st.error("Consult cardiologist immediately 🚨")
        suggestions.append("Consult cardiologist immediately")

    if len(suggestions) == 0:
        st.success("All parameters look good ✅")
        suggestions.append("All parameters normal")

    # ---------- PDF REPORT ---------- #
    st.subheader("📄 Download Report")

    def create_pdf():
        doc = SimpleDocTemplate("report.pdf")
        styles = getSampleStyleSheet()
        content = []

        content.append(Paragraph("Heart Disease Report", styles["Title"]))
        content.append(Spacer(1,10))

        content.append(Paragraph(f"Patient Name: {patient_name}", styles["Normal"]))
        content.append(Paragraph(f"Risk: {risk:.2f}%", styles["Normal"]))
        content.append(Paragraph(f"Confidence: {confidence:.2f}%", styles["Normal"]))
        content.append(Paragraph(f"Category: {category}", styles["Normal"]))
        content.append(Spacer(1,10))

        content.append(Paragraph("Health Suggestions:", styles["Heading2"]))
        for s in suggestions:
            content.append(Paragraph(f"- {s}", styles["Normal"]))

        doc.build(content)

    create_pdf()

    with open("report.pdf", "rb") as f:
        st.download_button("⬇️ Download Full Report", f, file_name="Heart_Report.pdf")
