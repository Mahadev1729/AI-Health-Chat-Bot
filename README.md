# 🩺 AI Health Diagnosis ChatBot

A smart **AI-powered health diagnosis chatbot** built using **Python**, **Streamlit**, and **Machine Learning (RandomForestClassifier)**.  
It reads user symptoms written in **normal English**, extracts them using simple **NLP techniques**, and predicts the most likely disease along with precautions and medical information.

---

## 🚀 Features

### ✔ Disease Prediction (Machine Learning)
- Trained using **Random Forest Classifier**
- Predicts possible disease with **confidence score**
- Uses symptom vector encoding for classification

### ✔ NLP-Based Symptom Extraction
Understands natural English text using:
- **Synonym mapping**
- **Keyword matching**
- **Fuzzy matching** (spelling correction)
- **Regex-based text processing**

### ✔ Medical Information Integration
Includes:
- Disease description  
- Symptom severity levels  
- Recommended precautions  
- User details like age, lifestyle & family history  

### ✔ Interactive Streamlit UI
- Clean and modern interface  
- Real-time prediction  
- User-friendly form system  
- Displays complete diagnosis summary  

---


---

## 🧠 Machine Learning Model Details

- Algorithm: **RandomForestClassifier (300 trees)**
- Data preprocessing:
  - Removed duplicate columns
  - Label encoded disease names
- Train-test split: **67% train / 33% test**

---

## 🗣 How NLP Works (Simple Explanation)

This chatbot reads your sentence and extracts symptoms using:

### 1️⃣ Synonyms  
Maps common phrases:
- "stomach ache" → **stomach_pain**
- "loose motion" → **diarrhea**

### 2️⃣ Exact keyword matching  
Checks if symptom words appear in text.

### 3️⃣ Fuzzy matching (handle misspellings)  
Fixes spelling errors:
- "feaver" → **fever**
- "headak" → **headache**

---

## 🛠️ Installation & Setup

### **1. Clone the repository**
```bash
git clone https://github.com/yourusername/AI-Health-ChatBot.git
cd AI-Health-ChatBot

python -m venv venv
venv/Scripts/activate

pip install -r requirements.txt

streamlit run app.py


Mahadev Athani
AI | Machine Learning | Python | Full Stack Developer



