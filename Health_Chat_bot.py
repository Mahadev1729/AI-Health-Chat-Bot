import streamlit as st
import pandas as pd
import numpy as np
import csv
import re
import random
import pickle
import os
from difflib import get_close_matches
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')


training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
testing.columns = testing.columns.str.replace(r"\.\d+$", "", regex=True)
training = training.loc[:, ~training.columns.duplicated()]
testing = testing.loc[:, ~testing.columns.duplicated()]

cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']


le = preprocessing.LabelEncoder()
y = le.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42
)


model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(x_train, y_train)

# Pickle the model
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)
model_filename = os.path.join(model_dir, 'health_model.pkl')
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved and pickled to {model_filename}")

severityDictionary = {}
description_list = {}
precautionDictionary = {}
symptoms_dict = {symptom: idx for idx, symptom in enumerate(x)}

def getDescription():
    with open('MasterData/symptom_Description.csv') as csv_file:
        for row in csv.reader(csv_file):
            description_list[row[0]] = row[1]

def getSeverityDict():
    with open('MasterData/symptom_severity.csv') as csv_file:
        for row in csv.reader(csv_file):
            try:
                severityDictionary[row[0]] = int(row[1])
            except:
                pass

def getprecautionDict():
    with open('MasterData/symptom_precaution.csv') as csv_file:
        for row in csv.reader(csv_file):
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]


symptom_synonyms = {
    "stomach ache": "stomach_pain",
    "belly pain": "stomach_pain",
    "tummy pain": "stomach_pain",
    "loose motion": "diarrhea",
    "motions": "diarrhea",
    "high temperature": "fever",
    "temperature": "fever",
    "feaver": "fever",
    "coughing": "cough",
    "throat pain": "sore_throat",
    "cold": "chills",
    "breathing issue": "breathlessness",
    "shortness of breath": "breathlessness",
    "body ache": "muscle_pain",
}

def extract_symptoms(user_input, all_symptoms):
    extracted = []
    text = user_input.lower().replace("-", " ")

   
    for phrase, mapped in symptom_synonyms.items():
        if phrase in text:
            extracted.append(mapped)

    
    for symptom in all_symptoms:
        if symptom.replace("_", " ") in text:
            extracted.append(symptom)

   
    words = re.findall(r"\w+", text)
    for word in words:
        close = get_close_matches(
            word, [s.replace("_", " ") for s in all_symptoms], n=1, cutoff=0.8
        )
        if close:
            for sym in all_symptoms:
                if sym.replace("_", " ") == close[0]:
                    extracted.append(sym)

    return list(set(extracted))


def predict_disease(symptoms_list):
    input_vector = np.zeros(len(symptoms_dict))

    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    pred_proba = model.predict_proba([input_vector])[0]
    pred_class = np.argmax(pred_proba)

    disease = le.inverse_transform([pred_class])[0]
    confidence = round(pred_proba[pred_class] * 100, 2)

    return disease, confidence, pred_proba


st.set_page_config(page_title="AI Health ChatBot", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

# Simplified Custom CSS Styling
st.markdown("""
<style>
    /* Main Page */
    .main {
        padding: 0 20px;
    }
    
    /* Headers - Clear and readable */
    h1 {
        color: #1e3a8a;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 10px;
        text-align: center;
    }
    
    h2 {
        color: #1e40af;
        font-size: 28px;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 15px;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 10px;
    }
    
    h3 {
        color: #1e40af;
        font-size: 20px;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f3f4f6;
        padding: 20px;
    }
    
    [data-testid="stSidebar"] h1 {
        text-align: left;
        font-size: 24px;
        color: #1e3a8a;
        margin-bottom: 20px;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        background-color: white;
        padding: 12px 15px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #3b82f6;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    [data-testid="stSidebar"] .stRadio > label:hover {
        background-color: #e0e7ff;
        border-left-color: #1e40af;
    }
    
    /* Cards */
    .info-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #f0f4ff 100%);
        border-left: 4px solid #3b82f6;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    .success-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
        border-left: 4px solid #10b981;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fffbeb 0%, #ffe0b2 100%);
        border-left: 4px solid #f59e0b;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    .error-card {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #ef4444;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border: 2px solid #e5e7eb;
        border-radius: 6px;
        padding: 10px 12px;
        font-size: 15px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Labels */
    .stLabel > label {
        font-weight: 600;
        color: #1f2937;
        font-size: 15px;
        margin-bottom: 8px;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 15px;
        cursor: pointer;
        transition: all 0.2s;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #1e40af;
        transform: none;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f3f4f6;
        color: #1f2937;
        border-radius: 6px;
        padding: 12px 15px;
        font-weight: 600;
        border-left: 4px solid #3b82f6;
    }
    
    /* Steps */
    .step {
        background: white;
        border-left: 4px solid #3b82f6;
        padding: 20px;
        margin: 15px 0;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    /* Text styling */
    p {
        line-height: 1.6;
        color: #374151;
        font-size: 15px;
    }
    
    /* Disclaimer */
    .disclaimer {
        background-color: #fef3c7;
        border-left: 4px solid #d97706;
        padding: 15px;
        border-radius: 6px;
        margin: 15px 0;
        color: #92400e;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("## 🩺 Navigation")
    page_selection = st.radio(
        "Select a page:",
        ["🏥 Diagnosis", "❓ FAQ", "📖 How to Use", "💬 Feedback", "ℹ️ About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 12px;">
        <p><b>AI Health ChatBot</b><br>Version 1.0</p>
    </div>
    """, unsafe_allow_html=True)

# Main content
if page_selection == "🏥 Diagnosis":
    st.markdown("""
    <h1 style="text-align: center; color: #1e3a8a;">🩺 AI Health Diagnosis</h1>
    <p style="text-align: center; color: #666; font-size: 16px; margin-bottom: 20px;">
        Get preliminary health insights based on your symptoms
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <b>Medical Disclaimer:</b> This is for informational purposes only. Always consult a qualified healthcare provider for medical advice and diagnosis.
    </div>
    """, unsafe_allow_html=True)

elif page_selection == "❓ FAQ":
    st.markdown("# ❓ Frequently Asked Questions")
    
    st.markdown("## Common Questions")
    
    with st.expander("🎯 How accurate is this chatbot?"):
        st.markdown("""
        This chatbot uses machine learning to suggest possible diseases based on symptoms.
        - Accuracy depends on the quality and completeness of your symptom descriptions
        - Results are suggestions, not diagnoses
        - **Always consult a healthcare professional for confirmation**
        """)
    
    with st.expander("⚕️ What should I do with the results?"):
        st.markdown("""
        1. Take the result as a suggestion, not a diagnosis
        2. If concerned, consult with a qualified doctor immediately
        3. Provide all your medical history to your doctor
        4. Do not self-medicate without professional advice
        """)
    
    with st.expander("🔍 What symptoms are recognized?"):
        st.markdown("""
        The chatbot recognizes many symptoms including:
        - Fever, headache, cough, sore throat
        - Stomach pain, diarrhea, nausea
        - Joint pain, muscle pain, body ache
        - Chills, breathlessness, and more
        
        Type symptoms naturally - the app understands variations!
        """)
    
    with st.expander("💓 Why are vital signs important?"):
        st.markdown("""
        Vital signs help differentiate between conditions:
        - High fever often indicates infection
        - Abnormal heart rate suggests specific conditions
        - Blood pressure abnormalities identify cardiovascular issues
        """)

elif page_selection == "📖 How to Use":
    st.markdown("# 📖 How to Use the AI Health ChatBot")
    
    st.markdown("""
    ## Quick Start Guide
    
    **Step 1: Personal Information** 👤
    - Enter your name, age, and gender
    - This helps identify conditions more accurately
    
    **Step 2: Describe Symptoms** 🤒
    - Be specific about your symptoms
    - Example: "I have fever, headache, and body pain"
    - The app understands symptom variations
    
    **Step 3: Health Details** 🌡️
    - How long have you had these symptoms?
    - Rate the severity (1-10)
    - Provide vital signs if available
    
    **Step 4: Medical History** 💊
    - List any pre-existing conditions
    - Mention current medications
    - Note any allergies
    - Recent travel or vaccination status
    
    **Step 5: Get Results** ✅
    - View the suggested disease
    - Read about the condition
    - Follow recommended precautions
    """)
    
    st.markdown("---")
    st.markdown("""
    ## Important Tips
    
    ✅ **Do:**
    - Provide complete and accurate information
    - Consult a doctor for confirmation
    - Keep track of your symptoms
    
    ❌ **Don't:**
    - Self-diagnose based solely on this app
    - Self-medicate without advice
    - Ignore serious symptoms
    """)

elif page_selection == "💬 Feedback":
    st.markdown("# 💬 Feedback & Support")
    
    st.markdown("""
    We'd love to hear from you! Your feedback helps us improve.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        feedback_name = st.text_input("Your Name")
    with col2:
        feedback_email = st.text_input("Your Email")
    
    feedback_type = st.selectbox(
        "Feedback Type",
        ["Bug Report", "Feature Request", "General Feedback", "Accuracy Issue", "Other"]
    )
    
    feedback_message = st.text_area(
        "Your Message",
        placeholder="Please describe your feedback...",
        height=120
    )
    
    if st.button("📤 Submit Feedback", use_container_width=True):
        if feedback_name and feedback_email and feedback_message:
            st.success("✅ Thank you for your feedback!")
            st.info(f"A confirmation email will be sent to {feedback_email}")
        else:
            st.error("Please fill in all fields")

elif page_selection == "ℹ️ About":
    st.markdown("# ℹ️ About AI Health ChatBot")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("## 🎯 Project Overview")
        st.markdown("""
        An intelligent health chatbot that helps users get preliminary health insights based on their symptoms and conditions.
        """)
    
    with col2:
        st.markdown("## 🔬 Technology")
        st.markdown("""
        - **ML Algorithm:** Random Forest Classifier
        - **Framework:** Streamlit
        - **Libraries:** Pandas, NumPy, Scikit-learn
        - **Model Storage:** Pickle
        """)
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("## 📊 Model Details")
        st.markdown("""
        - **Algorithm:** Random Forest (300 estimators)
        - **Training:** Comprehensive disease-symptom dataset
        - **Input:** Symptoms, vital signs, medical history
        - **Output:** Disease prediction with confidence
        """)
    
    with col4:
        st.markdown("## ⚖️ Legal Notice")
        st.markdown("""
        - For informational purposes only
        - NOT a substitute for professional medical advice
        - Always consult qualified healthcare providers
        - User assumes responsibility for health decisions
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 📞 Contact & Support
    
    **Email:** support@aihealthchatbot.com  
    **GitHub:** Mahadev1729/AI-Health-Chat-Bot  
    
    We welcome bug reports, feature requests, and contributions!
    
    ---
    
    🔐 **Privacy:** We don't store personal health information. Your data is confidential.
    """)

# Rest of the application continues below
if page_selection == "🏥 Diagnosis":
    getSeverityDict()
    getDescription()
    getprecautionDict()

    st.markdown("## 👤 Personal Information")

    name = st.text_input("Full Name")
    age = st.number_input("Age", min_value=1, max_value=120, step=1, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    st.markdown("## 🤒 Symptoms & Health")

    symptoms_input = st.text_area(
        "Describe your symptoms",
        placeholder="Example: fever, headache, and stomach pain",
        height=80
    )

    num_days = st.number_input(
        "Days with symptoms",
        min_value=1, max_value=30, value=1
    )

    severity_scale = st.slider(
        "Severity (1=mild, 10=very severe)",
        min_value=1, max_value=10, value=5
    )

    st.markdown("## 🌡️ Vital Signs")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        body_temp = st.number_input(
            "Temperature (°F)",
            min_value=95.0, max_value=107.0, value=98.6, step=0.1
        )

    with col2:
        heart_rate = st.number_input(
            "Heart Rate (bpm)",
            min_value=40, max_value=200, value=72, step=1
        )

    with col3:
        blood_pressure = st.text_input(
            "Blood Pressure",
            placeholder="120/80"
        )

    st.markdown("## 💊 Medical Information")

    with col2:
        heart_rate = st.number_input(
            "Heart Rate (bpm):",
            min_value=40, max_value=200, value=72, step=1,
            help="Normal resting heart rate is 60-100 bpm"
        )

    with col3:
        blood_pressure = st.text_input(
            "Blood Pressure (systolic/diastolic):",
            placeholder="Example: 120/80",
            help="Format: systolic/diastolic"
        )

    st.subheader("💊 Medical & Lifestyle Information")

    col4, col5 = st.columns(2)

    with col4:
        sleep_quality = st.selectbox(
            "Sleep Quality",
            ["Good", "Fair", "Poor"]
        )
        pre_exist = st.text_input("Pre-existing Conditions")

    with col5:
        appetite = st.selectbox(
            "Appetite Status",
            ["Normal", "Decreased", "Increased", "Loss of Appetite"]
        )
        allergies = st.text_input("Allergies")

    col6, col7 = st.columns(2)

    with col6:
        medications = st.text_input("Current Medications")
        recent_travel = st.selectbox(
            "Recent Travel",
            ["No", "Yes, Domestic", "Yes, International"]
        )

    with col7:
        vaccination_status = st.selectbox(
            "Vaccination Status",
            ["Fully Vaccinated", "Partially Vaccinated", "Not Vaccinated", "Unknown"]
        )
        family = st.text_input("Family History")

    lifestyle = st.text_input("Lifestyle (smoking, stress, sleep)")

    st.markdown("---")
    if st.button("🔍 Get Diagnosis", use_container_width=True):
        if not name or not symptoms_input.strip():
            st.error("❌ Please enter your name and describe your symptoms")
        else:
            symptoms_list = extract_symptoms(symptoms_input, cols)

            if not symptoms_list:
                st.error("❌ No symptoms detected. Please describe your symptoms in more detail")
            else:
                st.success(f"✅ Detected: {', '.join(symptoms_list)}")

                # Prediction
                disease, confidence, proba = predict_disease(symptoms_list)

                st.markdown("---")
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
                            border-radius: 10px; padding: 25px; text-align: center;
                            border-left: 5px solid #3b82f6; margin: 20px 0;">
                    <h2 style="color: #1e3a8a; margin: 0 0 10px 0;">🩺 Diagnosis Result</h2>
                    <h3 style="color: #1e40af; margin: 10px 0;">Possible Disease: <b>{disease}</b></h3>
                    <h4 style="color: #3b82f6; margin: 10px 0;">Confidence: <b>{confidence}%</b></h4>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### 📖 Disease Information:")
                st.info(description_list.get(disease, "No information available"))

                if disease in precautionDictionary:
                    st.markdown("### 🛡️ Recommended Precautions:")
                    for i, precaution in enumerate(precautionDictionary[disease], 1):
                        st.markdown(f"**{i}.**  {precaution}")

                st.markdown("---")
                st.markdown("### 📋 Your Information Summary:")
                
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.markdown(f"""
                    - **Symptoms:** {', '.join(symptoms_list)}
                    - **Duration:** {num_days} days
                    - **Severity:** {severity_scale}/10
                    - **Temperature:** {body_temp}°F
                    - **Heart Rate:** {heart_rate} bpm
                    """)
                
                with col_info2:
                    st.markdown(f"""
                    - **Blood Pressure:** {blood_pressure if blood_pressure else 'N/A'}
                    - **Sleep Quality:** {sleep_quality}
                    - **Appetite:** {appetite}
                    - **Medications:** {medications if medications else 'None'}
                    - **Allergies:** {allergies if allergies else 'None'}
                    """)

                st.markdown("---")
                st.markdown("""
                <div class="disclaimer">
                ⚠️ <b>Important:</b> This diagnosis is a suggestion only. Please consult a qualified healthcare provider for proper medical advice.
                </div>
                """, unsafe_allow_html=True)

                st.success("✅ Thank you for using AI Health ChatBot. Stay healthy!")
