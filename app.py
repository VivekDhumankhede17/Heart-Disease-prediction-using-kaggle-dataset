
import streamlit as st
import pandas as pd
import pickle

# Load the trained model
# Make sure 'best_model.pkl' is in the same directory as app.py or provide the correct path
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit App Title
st.title('Heart Disease Prediction App')
st.write('Enter the patient details below to predict the likelihood of heart disease.')

# Define the mappings for categorical features (based on LabelEncoder's behavior during training)
sex_options = {'Male': 1, 'Female': 0}
chest_pain_type_options = {'ASY': 0, 'ATA': 1, 'NAP': 2, 'TA': 3}
resting_ecg_options = {'LVH': 0, 'Normal': 1, 'ST': 2}
exercise_angina_options = {'No': 0, 'Yes': 1}
st_slope_options = {'Down': 0, 'Flat': 1, 'Up': 2}

# Input fields for the features
st.sidebar.header('Patient Input Features')

def user_input_features():
    age = st.sidebar.slider('Age', 18, 80, 50)
    sex_display = st.sidebar.selectbox('Sex', list(sex_options.keys()))
    chest_pain_type_display = st.sidebar.selectbox('Chest Pain Type', list(chest_pain_type_options.keys()))
    resting_bp = st.sidebar.slider('Resting Blood Pressure (RestingBP)', 90, 200, 120)
    cholesterol = st.sidebar.slider('Cholesterol', 0, 600, 200)
    fasting_bs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (FastingBS)', [0, 1])
    resting_ecg_display = st.sidebar.selectbox('Resting Electrocardiographic Results (RestingECG)', list(resting_ecg_options.keys()))
    max_hr = st.sidebar.slider('Maximum Heart Rate Achieved (MaxHR)', 60, 220, 150)
    exercise_angina_display = st.sidebar.selectbox('Exercise Induced Angina (ExerciseAngina)', list(exercise_angina_options.keys()))
    oldpeak = st.sidebar.slider('Oldpeak (ST depression induced by exercise relative to rest)', 0.0, 6.0, 1.0, 0.1)
    st_slope_display = st.sidebar.selectbox('ST_Slope (The slope of the peak exercise ST segment)', list(st_slope_options.keys()))

    # Encode selected categorical values
    sex = sex_options[sex_display]
    chest_pain_type = chest_pain_type_options[chest_pain_type_display]
    resting_ecg = resting_ecg_options[resting_ecg_display]
    exercise_angina = exercise_angina_options[exercise_angina_display]
    st_slope = st_slope_options[st_slope_display]

    data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input parameters')
st.write(input_df)

# Predict button
if st.button('Predict Heart Disease'):
    prediction_proba = model.predict_proba(input_df)
    prediction = model.predict(input_df)

    st.subheader('Prediction')
    heart_disease_status = 'Positive' if prediction[0] == 1 else 'Negative'
    st.write(f'Heart Disease Status: **{heart_disease_status}**')
    st.write(f'Probability of Heart Disease (Class 1): **{prediction_proba[0][1]:.2f}**')
    st.write(f'Probability of No Heart Disease (Class 0): **{prediction_proba[0][0]:.2f}**')

    if prediction[0] == 1:
        st.warning('Based on the input parameters, there is a prediction of Heart Disease.')
    else:
        st.success('Based on the input parameters, there is no prediction of Heart Disease.')
