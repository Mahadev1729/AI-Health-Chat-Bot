import streamlit as st
import pandas as pd
import numpy as np
import csv
import re
import random
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


st.set_page_config(page_title="AI Health ChatBot", page_icon="🩺", layout="centered")

st.title("🩺 AI Health Diagnosis ChatBot")
st.write("Fill in the details below and describe your symptoms clearly.")


getSeverityDict()
getDescription()
getprecautionDict()

st.subheader("👤 Personal Information")

name = st.text_input("Full Name:", placeholder="Enter your full name")
age = st.number_input("Age:", min_value=1, max_value=120, step=1)
gender = st.selectbox("Gender:", ["Male", "Female", "Other"])

st.subheader("🤒 Health Details")

symptoms_input = st.text_area(
    "Describe your symptoms:",
    placeholder="Example: I have fever, headache, and stomach pain"
)

num_days = st.number_input(
    "For how many days have you had these symptoms?",
    min_value=1, max_value=30,
    help="Enter the duration of symptoms"
)

severity_scale = st.slider(
    "How severe is your condition? (1 = mild, 10 = very severe)",
    min_value=1, max_value=10, value=5
)

pre_exist = st.text_input(
    "Pre-existing medical conditions:",
    placeholder="Example: diabetes, blood pressure, none"
)

lifestyle = st.text_input(
    "Lifestyle habits:",
    placeholder="smoking, alcohol, stress, irregular sleep, none"
)

family = st.text_input(
    "Any family history of similar illness?",
    placeholder="Example: yes, my parents had similar illness"
)


if st.button("🔍 Analyze & Predict"):
    if not name or not symptoms_input.strip():
        st.error("Please enter your name and describe your symptoms.")
    else:
        symptoms_list = extract_symptoms(symptoms_input, cols)

        if not symptoms_list:
            st.error("Couldn't detect symptoms. Try adding more details.")
        else:
            st.success(f"Detected symptoms: {', '.join(symptoms_list)}")

            # Prediction
            disease, confidence, proba = predict_disease(symptoms_list)

            st.subheader("🩺 Diagnosis Result")
            st.write(f"### Possible Disease: **{disease}**")
            st.write(f"### Confidence: **{confidence}%**")

            st.write("#### 📖 About the Disease:")
            st.info(description_list.get(disease, "No description available."))

            if disease in precautionDictionary:
                st.write("### 🛡️ Suggested Precautions:")
                for p in precautionDictionary[disease]:
                    st.write("- " + p)

            st.write("### 📌 Additional Info Provided:")
            st.write(f"🕒 Symptom Duration: **{num_days} days**")
            st.write(f"🔥 Severity Level: **{severity_scale}/10**")
            st.write(f"💉 Pre-existing Conditions: **{pre_exist}**")
            st.write(f"🏃 Lifestyle: **{lifestyle}**")
            st.write(f"👪 Family History: **{family}**")

            st.success("Thank you for using the AI Health ChatBot! Stay healthy! 🌿")
