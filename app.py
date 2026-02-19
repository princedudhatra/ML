import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os


st.markdown("""
<style>
/* Style ONLY the main button */
div[data-testid="stButton"] > button {
    background: linear-gradient(90deg, #2563eb, #0ea5e9);
    color: white;
    font-size: 1.05rem;
    font-weight: 600;
    padding: 0.75rem 1.2rem;
    border-radius: 14px;
    border: none;
    box-shadow: 0px 6px 18px rgba(37, 99, 235, 0.4);
    transition: all 0.2s ease-in-out;
}

/* Hover effect */
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(90deg, #1e40af, #0284c7);
    box-shadow: 0px 8px 24px rgba(37, 99, 235, 0.55);
    transform: translateY(-1px);
}

/* Active (click) effect */
div[data-testid="stButton"] > button:active {
    transform: scale(0.98);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Section headers */
h3 {
    color: #1e3a8a;
}


/* Prediction emphasis */
.result-box {
    padding: 1.25rem;
    border-radius: 14px;
    font-size: 1.1rem;
    margin-top: 1rem;
}

/* Softer labels */
label {
    font-weight: 500;
    color: #334155;
}
</style>
""", unsafe_allow_html=True)

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Cardio Health Risk Predictor",
    page_icon="🫀",
    layout="wide"
)

# -------------------- LOAD MODEL --------------------

@st.cache_resource(show_spinner="Loading ML model...")
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "cardio_model.pkl")
    saved = joblib.load(model_path)
    return saved

saved = load_model()

model = saved["model"]
columns = saved["columns"]

# Backward compatibility for analysis metrics
has_analysis = "train_accuracy" in saved


# -------------------- HEADER --------------------
st.markdown("<h1>🫀 Cardiovascular Health Risk Assessment</h1>", unsafe_allow_html=True)
st.write(
    "This tool estimates **cardiovascular disease risk** using health and lifestyle information. "
    "It is intended for **educational purposes only**."
)

# -------------------- INPUT SECTION --------------------
st.markdown("## 🧑‍⚕️ Patient Information")

st.markdown("### 🧾 Basic Details")

c1, c2 = st.columns(2)

with c1:
    gender = st.selectbox("Gender", [0, 1], index=1, format_func=lambda x: "Female" if x == 0 else "Male")
    age = st.number_input("Age (years)", 18, 100, 40)

with c2:
    weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
    active = st.selectbox("Physically Active", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")



st.markdown("### ❤️ Vital Signs")

c3, c4 = st.columns(2)

with c3:
    ap_hi = st.number_input("Systolic BP (mmHg)", 80, 200, 120)

with c4:
    ap_lo = st.number_input("Diastolic BP (mmHg)", 50, 130, 80)


st.markdown("### 🧪 Blood & Lifestyle Factors")

c5, c6 = st.columns(2)

with c5:
    cholesterol = st.selectbox(
        "Cholesterol Level", [1, 2, 3],
        format_func=lambda x: ["Normal", "Above Normal", "High"][x - 1]
    )
    gluc = st.selectbox(
        "Glucose Level", [1, 2, 3],
        format_func=lambda x: ["Normal", "Above Normal", "High"][x - 1]
    )

with c6:
    smoke = st.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    alco = st.selectbox("Alcohol Consumption", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")


# -------------------- FEATURE ENGINEERING --------------------
height_m = 1.7
bmi = weight / (height_m ** 2)
pulse_pressure = ap_hi - ap_lo
health_index = (active + (1 - smoke) + (1 - alco)) / 3
cholesterol_gluc_interaction = cholesterol * gluc

input_data = pd.DataFrame([{
    "gender": gender,
    "weight": weight,
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "cholesterol": cholesterol,
    "gluc": gluc,
    "smoke": smoke,
    "alco": alco,
    "active": active,
    "age_years": age,
    "bmi": bmi,
    "pulse_pressure": pulse_pressure,
    "health_index": health_index,
    "cholesterol_gluc_interaction": cholesterol_gluc_interaction
}])[columns]

def doctor_summary(risk_level):
    if risk_level == "low":
        return (
            "### 🩺 Doctor’s Summary\n"
            "Your results suggest a **low risk** of cardiovascular disease at the moment.\n\n"
            "**What this means:**\n"
            "- Your current health indicators are within a generally safe range.\n\n"
            "**What you should do:**\n"
            "- Continue maintaining a healthy lifestyle\n"
            "- Eat a balanced diet and stay physically active\n"
            "- Get routine health checkups once a year\n\n"
            "✅ No immediate medical concern is indicated."
        )

    elif risk_level == "medium":
        return (
            "### 🩺 Doctor’s Summary\n"
            "Your results indicate a **moderate risk** of cardiovascular disease.\n\n"
            "**What this means:**\n"
            "- Some health or lifestyle factors may increase future risk if ignored.\n\n"
            "**What you should do:**\n"
            "- Improve physical activity and diet habits\n"
            "- Reduce smoking or alcohol intake if applicable\n"
            "- Monitor blood pressure and cholesterol regularly\n"
            "- Consider consulting a healthcare professional for advice\n\n"
            "⚠️ Early action can significantly reduce future risk."
        )

    else:
        return (
            "### 🩺 Doctor’s Summary\n"
            "Your results suggest a **high risk** of cardiovascular disease.\n\n"
            "**What this means:**\n"
            "- Multiple factors indicate a higher chance of heart-related issues.\n\n"
            "**What you should do immediately:**\n"
            "- Consult a doctor or healthcare professional as soon as possible\n"
            "- Get a detailed medical evaluation\n"
            "- Follow professional advice on medication, diet, and lifestyle changes\n\n"
            "🚨 This result is **not a diagnosis**, but it should not be ignored."
        )

# -------------------- RISK ASSESSMENT --------------------
st.markdown("### 📊 Risk Assessment")

predict_clicked = st.button(
    "🩺 Predict Cardiovascular Risk",
    use_container_width=True
)

if predict_clicked:
    probability = model.predict_proba(input_data)[0][1]
    risk_percentage = probability * 100

    if risk_percentage < 30:
        risk_level = "low"
        st.success(f"🟢 **Low Risk** — {risk_percentage:.1f}%")
    elif risk_percentage < 60:
        risk_level = "medium"
        st.warning(f"🟠 **Moderate Risk** — {risk_percentage:.1f}%")
    else:
        risk_level = "high"
        st.error(f"🔴 **High Risk** — {risk_percentage:.1f}%")

    st.progress(int(risk_percentage))


# -------------------- DOCTOR SUMMARY --------------------
if predict_clicked:
    st.markdown(doctor_summary(risk_level))

# -------------------- VIEW DETAILED MODEL ANALYSIS --------------------
if has_analysis:
    with st.expander("🔬 View Detailed Model Analysis", expanded=False):
        st.markdown("#### 📈 Key metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Accuracy", f"{saved['train_accuracy']:.2%}")
        with col2:
            st.metric("Testing Accuracy", f"{saved['test_accuracy']:.2%}")
        with col3:
            st.metric("5-Fold CV Score", f"{saved['cv_score']:.2%}")

        if "model_comparison" in saved:
            st.markdown("#### 📊 Model comparison")
            comp = pd.DataFrame.from_dict(saved["model_comparison"], orient="index")
            st.dataframe(comp.style.format("{:.4f}"), use_container_width=True)

        if "feature_importance" in saved:
            st.markdown("#### 📉 Feature importance")
            fi = saved["feature_importance"]
            fig, ax = plt.subplots(figsize=(10, 4))
            names = list(fi.keys())
            vals = list(fi.values())
            ax.barh(range(len(names)), vals, color="steelblue", alpha=0.8)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importance")
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        if "roc_fpr" in saved and "roc_tpr" in saved:
            st.markdown("#### 📈 ROC curve")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(saved["roc_fpr"], saved["roc_tpr"], label="ROC curve", color="darkorange")
            ax.plot([0, 1], [0, 1], "k--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        if "learning_curve" in saved:
            lc = saved["learning_curve"]
            st.markdown("#### 📈 Learning curve")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(lc["train_sizes"], lc["train_scores_mean"], "o-", label="Training score")
            ax.plot(lc["train_sizes"], lc["test_scores_mean"], "o-", label="Cross-validation score")
            ax.set_xlabel("Training set size")
            ax.set_ylabel("Accuracy")
            ax.set_title("Learning Curve")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# -------------------- FOOTER --------------------
st.markdown(
    """
    <div style="
        text-align:center;
        font-size:0.85rem;
        color:#475569;
        margin-top:2rem;
        padding-bottom:1rem;
    ">
        ⚠️ This application is for <b>academic and educational purposes only</b>.
        It does not replace professional medical advice or diagnosis.
    </div>
    """,
    unsafe_allow_html=True
)
