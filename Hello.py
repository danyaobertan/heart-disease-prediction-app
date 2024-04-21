import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def set_page_config():
    st.set_page_config(page_title="Heart Prediction App", layout="wide")


def sidebar_input():
    st.sidebar.header('User Input Features')
    st.sidebar.markdown(
        "[Example CSV input file](https://pastebin.com/8SuCYhcf)")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        input_df = user_input_features()
    return input_df


def user_input_features():
    Age = st.sidebar.slider('Age', 29, 77, 37)
    Sex = st.sidebar.selectbox('Sex', ('M', 'F'))
    ChestPainType = st.sidebar.selectbox(
        'Chest Pain Type', ('ATA', 'NAP', 'ASY', 'TA'))
    RestingBP = st.sidebar.slider(
        'Resting Blood Pressure (mmHg)', 92, 165, 140)
    Cholesterol = st.sidebar.slider('Cholesterol (mg/dl)', 85, 407, 207)
    FastingBS = st.sidebar.selectbox(
        'Fasting Blood Sugar > 120 mg/dl', ('N', 'Y'))
    RestingECG = st.sidebar.selectbox('Resting ECG', ('Normal', 'ST', 'LVH'))
    MaxHR = st.sidebar.slider('Maximum Heart Rate Achieved', 69, 202, 130)
    ExerciseAngina = st.sidebar.selectbox(
        'Exercise-Induced Angina', ('N', 'Y'))
    Oldpeak = st.sidebar.slider(
        'ST Depression Induced by Exercise', -0.1, 6.2, 1.5, 0.1)
    STSlope = st.sidebar.selectbox(
        'Slope of the Peak Exercise ST Segment', ('Up', 'Flat', 'Down'))
    data = {'Age': Age, 'Sex': Sex, 'ChestPainType': ChestPainType, 'RestingBP': RestingBP,
            'Cholesterol': Cholesterol, 'FastingBS': FastingBS, 'RestingECG': RestingECG,
            'MaxHR': MaxHR, 'ExerciseAngina': ExerciseAngina, 'Oldpeak': Oldpeak, 'STSlope': STSlope}
    features = pd.DataFrame(data, index=[0])
    return features


def load_data():
    return pd.read_csv('patients-cleansed-test-edited.csv')


def preprocess_data(input_df, df):
    df_combined = pd.concat([input_df, df], axis=0)
    encode = ['Sex', 'ChestPainType', 'FastingBS',
              'RestingECG', 'ExerciseAngina', 'STSlope']
    for col in encode:
        dummy = pd.get_dummies(df_combined[col], prefix=col)
        df_combined = pd.concat([df_combined, dummy],
                                axis=1).drop(columns=[col])
    df_combined = df_combined.drop(columns=['HeartDisease'], errors='ignore')
    return df_combined[:1]  # Return only the user input data for prediction


def load_model():
    return pickle.load(open('HeartDisease_clf_gpt.pkl', 'rb'))


def predict(model, df):
    prediction_proba = model.predict_proba(df)
    return prediction_proba


def display_header():
    st.write("# Heart Disease Prediction App")
    st.markdown(
        "This app predicts whether a patient has heart disease based on various health parameters!")
    st.markdown(
        "Data obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/index.php).")


def display_user_input(input_df):
    st.subheader('User Input features')
    st.write(input_df)


def display_predictions(prediction_proba):
    sick_proba = prediction_proba[0][1]
    health_proba = prediction_proba[0][0]
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Sick Probability", value=f"{sick_proba:.2%}")
    with col2:
        st.metric(label="Healthy Probability", value=f"{health_proba:.2%}")

    # Display appropriate messages based on the prediction
    if sick_proba > 0.67:
        st.error("High risk of heart disease")
    elif sick_proba > 0.34:
        st.warning("Moderate risk of heart disease")
    else:
        st.success("Low risk of heart disease")


def display_legend():
    with st.expander("See Legend"):
        st.write("""
        Detailed information about each feature used in this prediction model.
        """)


def main():
    set_page_config()
    display_header()
    input_df = sidebar_input()
    patients_data = load_data()
    processed_data = preprocess_data(input_df, patients_data)
    display_user_input(input_df)
    model = load_model()
    prediction_proba = predict(model, processed_data)
    display_predictions(prediction_proba)
    display_legend()


if __name__ == "__main__":
    main()
