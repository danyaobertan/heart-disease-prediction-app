import pickle
import streamlit as st
import pandas as pd


def load_bundle(locale):
    df = pd.read_csv("text_bundle.csv")
    df = df.query(f"locale == '{locale}'")
    lang_dict = {row['key']: row['value'] for _, row in df.iterrows()}
    return lang_dict


def set_page_config():
    st.set_page_config(page_title="Heart Prediction App", layout="wide")


def sidebar_input(lang_dict):
    st.sidebar.header(lang_dict['user_input_features'])
    st.sidebar.markdown(lang_dict['example_csv_input_file'])
    uploaded_file = st.sidebar.file_uploader(
        lang_dict['upload_csv_input_file'], type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        input_df = user_input_features(lang_dict)
    return input_df


def user_input_features(lang_dict):
    Age = st.sidebar.slider(lang_dict['age'], 29, 77, 37)

    Sex = st.sidebar.selectbox(lang_dict['sex'], ('M', 'F'))

    ChestPainType = st.sidebar.selectbox(
        lang_dict['chest_pain_type'], ('ATA', 'NAP', 'ASY', 'TA'))

    RestingBP = st.sidebar.slider(
        lang_dict['resting_blood_pressure'], 92, 165, 140)

    Cholesterol = st.sidebar.slider(lang_dict['cholesterol'], 85, 407, 207)

    FastingBS = st.sidebar.selectbox(
        lang_dict['fasting_blood_sugar'], ('N', 'Y'))

    RestingECG = st.sidebar.selectbox(
        lang_dict['rest_ecg'], ('Normal', 'ST', 'LVH'))

    MaxHR = st.sidebar.slider(lang_dict['max_heart_rate'], 69, 202, 130)

    ExerciseAngina = st.sidebar.selectbox(
        lang_dict['exercise_induced_angina'], ('N', 'Y'))

    Oldpeak = st.sidebar.slider(
        lang_dict['st_depression'], -0.1, 6.2, 1.5, 0.1)

    STSlope = st.sidebar.selectbox(
        lang_dict['st_slope'], ('Up', 'Flat', 'Down'))

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


def display_header(lang_dict):
    st.write(f"# {lang_dict['title']}")
    st.markdown(lang_dict['app_description1'])
    st.markdown(lang_dict['app_description2'])


def display_user_input(input_df, lang_dict):
    st.subheader(lang_dict['user_input_features'])
    st.write(input_df)


def display_predictions(prediction_proba, lang_dict):
    st.subheader(lang_dict['prediction_probability'])
    sick_proba = prediction_proba[0][1]
    health_proba = prediction_proba[0][0]
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=lang_dict['metric_label_sick'],
                  value=f"{sick_proba:.1%}")
    with col2:
        st.metric(label=lang_dict['metric_label_health'],
                  value=f"{health_proba:.1%}")

    # Display appropriate messages based on the prediction
    if sick_proba > 0.67:
        st.error(lang_dict['result_sick_high'])
        st.markdown(lang_dict['result_sick_high_message'],
                    unsafe_allow_html=True)
    elif sick_proba > 0.34:
        st.warning(lang_dict['result_sick_moderate'])
        st.markdown(lang_dict['result_sick_moderate_message'],
                    unsafe_allow_html=True)
    else:
        st.success(lang_dict['result_not_sick'])
        st.markdown(lang_dict['result_not_sick_message'],
                    unsafe_allow_html=True)


def display_legend():
    with st.expander("See legend"):
        st.write("""
        Age: Age of the patient in years.\n
        Sex: Sex of the patient, either Male (M) or Female (F).\n
        ChestPainType: The type of chest pain experienced by the patient, with possible values including Typical Angina (TA), Atypical Angina (ATA), Non-Anginal Pain (NAP), and Asymptomatic (ASY).\n
        RestingBP: The patient's resting blood pressure in mm Hg.\n
        Cholesterol: The patient's serum cholesterol level in mm/dl.\n
        FastingBS: Whether the patient's fasting blood sugar is greater than 120 mg/dl, with a value of Y indicating that it is and N indicating that it is not.\n
        RestingECG: The results of the patient's resting electrocardiogram, with possible values including Normal, ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), and showing probable or definite left ventricular hypertrophy by Estes' criteria (LVH).\n
        MaxHR: The maximum heart rate achieved by the patient, with a numeric value between 60 and 202.\n
        ExerciseAngina: Whether the patient experienced exercise-induced angina, with a value of Y indicating that they did and N indicating that they did not.\n
        Oldpeak: The oldpeak ST value measured in depression.\n
        ST_Slope: The slope of the peak exercise ST segment, with possible values including Upsloping (Up), Flat (Flat), and Downsloping (Down).\n
        HeartDisease: The output class, with a value of 1 indicating that the patient has heart disease and 0 indicating that they do not.\n""")
    with st.expander("Переглянути легенду"):
        st.write("""
        Вік: Вік пацієнта у роках.\n
        Стать: Стать пацієнта, чоловік (M) або жінка (F).\n
        Тип болю в грудях: Тип болю, який відчуває пацієнт, з можливими значеннями, включаючи типову стенокардію (TA), нетипову стенокардію (ATA), негрудний біль (NAP) і асимптоматичну стенокардію (ASY).\n
        Артеріальний тиск у стані спокою: Артеріальний тиск пацієнта у мм ртутного стовпчика.\n
        Рівень холестерину: Рівень холестерину в сироватці пацієнта в мг/дл.\n
        Голодний рівень цукру в крові: Чи перевищує рівень цукру в крові пацієнта показник 120 мг/дл, зі значенням Y вказує, що так, а значення N - що ні.\n
        Електрокардіограма в стані спокою: Результати електрокардіограми пацієнта у стані спокою, з можливими значеннями, включаючи нормальний, аномалії ST-T хвилі (інверсії хвиль T і/або підйому або депресії ST більше 0,05 мВ) та покази ймовірного або визначеного гіпертрофії лівого шлуночка за критеріями Естеса (LVH).\n
        Максимальний пульс: Максимальний досягнутий пульс пацієнта, з числовим значенням в діапазоні від 60 до 202.\n
        Стресова стенокардія: Чи відчував пацієнт стресову стенокардію, зі значенням Y вказує, що так, а значення N - що ні.\n
        Показник депресії ST: Значення показника депресії ST.\n
        Наклон піку ST під час фізичного навантаження: Наклон піку ST під час фізичного навантаження, з можливими значеннями, включаючи підйомний (Up), плоский (Flat) та спадний (Down).\n
        Хвороба серця: Клас вихідних даних, зі значенням 1 вказує, що пацієнт має серцеву хворобу, а 0 - що ні.\n""")


def main():
    set_page_config()

    lang_options = {"English (US)": "en_US", "Українська": "uk_UA"}
    # Default to Ukrainian
    default_lang = lang_options["Українська"]
    # Use keys for user-friendly names, values for internal use
    locale = st.sidebar.selectbox(
        "Language",
        options=list(lang_options.values()),  # List of 'en_US', 'uk_UA'
        index=list(lang_options.values()).index(
            default_lang),  # Set default index for Ukrainian
        format_func=lambda x: "English (US)" if x == "en_US" else "Українська"
    )

    lang_dict = load_bundle(locale)

    display_header(lang_dict)
    # display_header()
    input_df = sidebar_input(lang_dict)
    patients_data = load_data()
    processed_data = preprocess_data(input_df, patients_data)
    display_user_input(input_df, lang_dict)
    model = load_model()
    prediction_proba = predict(model, processed_data)
    display_predictions(prediction_proba, lang_dict)
    display_legend()


if __name__ == "__main__":
    main()
