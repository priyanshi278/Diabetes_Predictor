import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Load model
with open('xg3.pkl', 'rb') as file:
    model = joblib.load(file)


#https://www.shutterstock.com/image-photo/doctor-checking-blood-sugar-level-600nw-1439349791.jpg
# Set page config
st.set_page_config(page_title='Diabetes Predictor', layout='wide')
# Subtle pastel gradient background
# Full-page background image with white transparent overlay for readability
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.85)),
                          url("https://t3.ftcdn.net/jpg/01/81/41/36/360_F_181413640_s8Pzq5NxaDHHT57OTwOuUyGE74HYTRBB.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
""", unsafe_allow_html=True)


import random

# Add this at the top of your script
health_tips = [
    "💧 Stay hydrated – drink at least 8 glasses of water daily.",
    "🥦 Eat more fiber-rich foods to manage blood sugar levels.",
    "🚶‍♂️ A 30-minute walk after meals can reduce glucose spikes.",
    "🛌 Get 7-8 hours of quality sleep to balance insulin levels.",
    "🍎 Avoid sugary drinks – choose fresh fruit or lemon water.",
    "🧘‍♀️ Reduce stress through meditation or deep breathing.",
    "🥗 Maintain a balanced plate – protein, carbs, and greens.",
    "💉 Monitor your sugar levels regularly.",
    "🧂 Limit processed food and added salts.",
    "👟 Regular exercise improves insulin sensitivity.",
]

selected_tip = random.choice(health_tips)


# ---- Sidebar Content ----
with st.sidebar:
    st.markdown("""
    <h4 style='text-align: center; color: #333; font-size: 28px;'>
        Diabetes Predictor App
    </h4>
    """, unsafe_allow_html=True)
    st.markdown("### 👋 Welcome")
    st.write("Every number tells a story — your health speaks, and we're here to listen.")
    st.markdown("> 🌟 *“Early awareness is the first step to lifelong wellness.”*")
    st.markdown("## 💡 Health Tip of the Day")
    st.success(selected_tip)

    st.markdown("## 🤖 Do you have any Question ?")

    question = st.selectbox(
        "Select a question to get quick help:",
        [
            "❓ What does this app do?",
            "💬 What factors increase diabetes risk?",
            "🧪 How accurate is this prediction?",
            "📉 How can I reduce my diabetes risk?",
            "🗒️ Can I trust the prediction result?",
        ]
    )

    if question == "❓ What does this app do?":
        st.info("This app uses health parameters to predict your diabetes risk using machine learning.")
    elif question == "💬 What factors increase diabetes risk?":
        st.info("Unhealthy diet, obesity, lack of exercise, high BP, age, and family history are major risk factors.")
    elif question == "🧪 How accurate is this prediction?":
        st.info("The model is trained on real-world data with an accuracy of around 85–90%. Still, consult a doctor.")
    elif question == "📉 How can I reduce my diabetes risk?":
        st.info("Exercise daily, eat a balanced diet, reduce sugar intake, sleep well, and manage stress.")
    elif question == "🗒️ Can I trust the prediction result?":
        st.info("It’s a smart assistant, not a doctor. Use it as a guide and always follow up with a medical expert.")


#title
st.markdown("""
    <div style='display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 10px;'>
        <img src="https://www.shutterstock.com/image-vector/beautiful-vector-diabetic-icon-glucometer-600nw-2204717311.jpg"
             alt="Logo"
             width="75"
             height="75"
             style="border-radius: 50%; border: 2px solid #9bd8b3; box-shadow: 0 4px 10px rgba(0,0,0,0.08);">
        <h1 style='color: #222; font-size: 38px; margin: 0;'>GlucoPilot</h1>
    </div>
    <h4 style='text-align:center; margin-top: 5px; color: #2f4f4f;'>  Let’s take a closer look at your health.</h4>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* Form Submit Button Styling */
div[data-testid="stForm"] button {
    background: linear-gradient(to right, #81D4FA, #A5D6A7);  /* Teal + mint green */
    color: #003c3c;
    font-size: 16px;
    font-weight: 600;
    padding: 0.7rem 1.2rem;
    border: none;
    border-radius: 10px;
    width: 100%;
    cursor: pointer;
    box-shadow: 0 4px 10px rgba(129, 212, 250, 0.3);
    transition: all 0.3s ease-in-out;
    letter-spacing: 0.5px;
}

/* Hover Effect */
div[data-testid="stForm"] button:hover {
    background: linear-gradient(to right, #4DB6AC, #81C784);  /* Slightly darker teal + green */
    transform: scale(1.02);
    box-shadow: 0 6px 16px rgba(77, 182, 172, 0.45);
}
</style>
""", unsafe_allow_html=True)




# Form for inputs
with st.form(key='diabetes_form'):
    
    st.markdown('<h3>📋 Enter Patient Information</h3>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
    with col2:
        glucose = st.number_input('Glucose Level', min_value=0, max_value=300, value=100)
    with col3:
        bp = st.number_input('Blood Pressure', min_value=0, max_value=200, value=70)
    with col4:
        skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
    
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        insulin = st.number_input('Insulin Level', min_value=0, max_value=900, value=80)
    with col6:
        bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
    with col7:
        dpf = st.number_input('DPF (Pedigree)', min_value=0.0, max_value=2.5, value=0.5)
    with col8:
        age = st.number_input('Age', min_value=1, max_value=120, value=25)

    submitted = st.form_submit_button('🔍 Predict Diabetes', use_container_width=True)

# ---- Prediction + Report ----
if submitted:
    input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    risk_status = "HIGH RISK" if probability > 50 else "LOW RISK"
    status_color = "🔴" if risk_status == "HIGH RISK" else "🟢"

    # Display prediction and probability
    st.markdown(f"""
        <div style='
            background-color: #ffffffdd;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-top: 30px;
            margin-bottom: 20px;
        '>
            <h3 style="color: #222;">Prediction Result</h3>
            <p style="font-size: 22px;">{status_color} <strong>{risk_status}</strong></p>
            <p style="font-size: 18px; color: #444;">Estimated Diabetes Risk: <strong>{probability:.2f}%</strong></p>
        </div>

    """, unsafe_allow_html=True)


    # Add chart heading
    st.markdown(" ")
    st.markdown("### 📊 Health Comparison Chart")
    st.markdown(" ")

    # Actual average data (based on PIMA + real references)
    healthy_avg = [1, 100, 70, 20, 85, 25.0, 0.45, 29]
    diabetic_avg = [4, 140, 80, 32, 150, 35.0, 0.55, 45]

    patient_data = [pregnancies, glucose, bp, skin_thickness,
                    insulin, bmi, dpf, age]

    features = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
                'Insulin', 'BMI', 'DPF', 'Age']

    df_compare = pd.DataFrame({
        'Feature': features,
        'Patient': patient_data,
        'Healthy Avg': healthy_avg,
        'Diabetic Avg': diabetic_avg
    })

    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot all three lines
    ax.plot(df_compare['Feature'], df_compare['Patient'], marker='o', label='👤 Patient', linewidth=2.5, color='#1f77b4')
    ax.plot(df_compare['Feature'], df_compare['Healthy Avg'], marker='s', linestyle='--', label='✅ Healthy Avg', color='#4CAF50')
    ax.plot(df_compare['Feature'], df_compare['Diabetic Avg'], marker='^', linestyle='--', label='⚠️ Diabetic Avg', color='#FF9800')

    # Highlight red dots if patient value > diabetic average
    for i, val in enumerate(patient_data):
        if val > diabetic_avg[i]:
            ax.plot(features[i], val, marker='o', color='red', markersize=10, label='🔴 High Risk' if i == 0 else "")

    # Aesthetics
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title("📊 Patient vs Healthy vs Diabetic Averages", fontsize=13)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Show plot in Streamlit
    st.pyplot(fig)


    # Summary Report with table-like layout and white background
    st.markdown(f"""
        <div style="background-color: white; padding: 2rem; border-radius: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-top: 2rem;">
            <h3 style="text-align: center; color: #333;">🧾 Patient Summary Report</h3>
            <hr style="border: 1px solid #ccc;" />
            <table style="width: 100%; font-size: 16px; color: #222; border-collapse: collapse;">
                <tr><td style="padding: 8px;"><strong>👤 Age</strong></td><td style="padding: 8px;">{age} years</td></tr>
                <tr><td style="padding: 8px;"><strong>🤰 Pregnancies</strong></td><td style="padding: 8px;">{pregnancies}</td></tr>
                <tr><td style="padding: 8px;"><strong>💉 Glucose Level</strong></td><td style="padding: 8px;">{glucose} mg/dL</td></tr>
                <tr><td style="padding: 8px;"><strong>❤️ Blood Pressure</strong></td><td style="padding: 8px;">{bp} mm Hg</td></tr>
                <tr><td style="padding: 8px;"><strong>🧪 Skin Thickness</strong></td><td style="padding: 8px;">{skin_thickness} mm</td></tr>
                <tr><td style="padding: 8px;"><strong>🧫 Insulin Level</strong></td><td style="padding: 8px;">{insulin} IU/mL</td></tr>
                <tr><td style="padding: 8px;"><strong>⚖️ BMI</strong></td><td style="padding: 8px;">{bmi}</td></tr>
                <tr><td style="padding: 8px;"><strong>🧬 Diabetes Pedigree Function</strong></td><td style="padding: 8px;">{dpf}</td></tr>
                <tr><td style="padding: 8px;"><strong>📊 Risk Probability</strong></td><td style="padding: 8px;">{probability:.2f}%</td></tr>
                <tr><td style="padding: 8px;"><strong>🔍 Prediction</strong></td><td style="padding: 8px; color: {'red' if prediction == 1 else 'green'};"><strong>{"HIGH RISK" if prediction == 1 else "LOW RISK"}</strong></td></tr>
            </table>
        </div>
    """, unsafe_allow_html=True)
