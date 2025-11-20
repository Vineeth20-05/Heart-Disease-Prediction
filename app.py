import streamlit as st
import numpy as np
import joblib

# Load trained model using joblib
clf = joblib.load('heart-disease-model.pkl') 

#  Custom CSS 
st.markdown("""
    <style>
        .stApp {
            background-color: #1e004b;
            color: white;
        }

        /* Make all labels bold and white */
        label, .css-17z5bey, .css-1c7y2kd {
            font-weight: bold !important;
            color: white !important;
            font-size: 16px !important;
        }

        /* Number and text inputs */
        .stNumberInput input, .stTextInput input {
            background-color: #2b2b2b !important;
            color: white !important;
            border-radius: 8px !important;
            border: 1px solid #888 !important;
            padding: 10px !important;
            font-size: 16px !important;
            font-weight: bold !important;
        }

        /* Selectbox input field */
        div[data-baseweb="select"] {
            background-color: #2b2b2b !important;
            color: white !important;
            border-radius: 8px !important;
            border: 1px solid #888 !important;
            padding: 6px !important;
            font-size: 16px !important;
            font-weight: bold !important;
        }

        /* Selected value in dropdown */
        .css-1jqq78o-placeholder, .css-1dimb5e-singleValue {
            color: white !important;
            font-weight: bold !important;
            font-size: 16px !important;
        }

        /* Hover effects */
        .stNumberInput input:hover, .stTextInput input:hover, div[data-baseweb="select"]:hover {
            border: 1px solid #bb86fc !important;
            box-shadow: 0 0 5px #bb86fc !important;
        }

        /* Green result box */
        div.stAlert.success {
            background-color: #0e3d0e !important;
            border: 2px solid #00ff7f;
            border-radius: 10px;
            padding: 20px;
            font-size: 20px !important;
            font-weight: bold;
            color: #aaffaa !important;
            box-shadow: 0px 0px 10px #00ff7f;
        }

        /* Red result box */
        div.stAlert.danger {
            background-color: #300000 !important;
            border: 2px solid red;
            border-radius: 10px;
            padding: 20px;
            font-size: 20px !important;
            font-weight: bold;
            color: #ff9999 !important;
            box-shadow: 0px 0px 10px red;
        }

        /* Styled Predict button */
        .stButton button {
            background-color: #8000ff;
            color: white;
            border-radius: 8px;
            height: 3em;
            font-weight: bold;
            font-size: 16px;
        }

        .stButton button:hover {
            background-color: #5a00b3;
        }
    </style>
""", unsafe_allow_html=True)


#  App Title
st.title("💓 Heart Disease Prediction App")

#  Input Features 
age = st.number_input("Age", min_value=0, max_value=120, step=1)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", options=[0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=200)
chol = st.number_input("Cholesterol", min_value=100, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", options=[0, 1])
restecg = st.selectbox("Rest ECG (0–2)", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", options=[0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0–2)", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", options=[0, 1, 2, 3])
thal = st.selectbox("Thal (1 = Normal; 2 = Fixed Defect; 3 = Reversible Defect)", options=[1, 2, 3])

#  Prediction 
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])
    prediction = clf.predict(input_data)[0]

    if prediction == 0:
        st.markdown('<div class="stAlert success">✅ The person is <strong>not likely</strong> to have heart disease.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="stAlert danger">⚠ The person is <strong>likely</strong> to have heart disease.</div>', unsafe_allow_html=True)
