from unittest.mock import patch
import pandas as pd
import pytest
import streamlit
import Hello as app

# To run the tests
# pytest test_app.py

# 1. Testing set_page_config()


def test_set_page_config():
    with patch('streamlit.set_page_config') as mock_set_page_config:
        app.set_page_config()
        mock_set_page_config.assert_called_once_with(
            page_title="Heart Prediction App", layout="wide")

# 2. Testing sidebar_input()


def test_sidebar_input_file_upload(mocker):
    mocker.patch('streamlit.sidebar.file_uploader',
                 return_value="fakefile.csv")
    mocker.patch('pandas.read_csv',
                 return_value=pd.DataFrame({'data': [1, 2, 3]}))
    assert app.sidebar_input().equals(pd.DataFrame({'data': [1, 2, 3]}))


# def test_sidebar_input_no_file(mocker):
#     mocker.patch('streamlit.sidebar.file_uploader', return_value=None)
#     mocker.patch('app.user_input_features',
#                  return_value=pd.DataFrame({'data': [1, 2, 3]}))
#     assert app.sidebar_input().equals(pd.DataFrame({'data': [1, 2, 3]}))

# 3. Testing user_input_features()


# def test_user_input_features(mocker):
#     mocker.patch('streamlit.sidebar.slider', side_effect=[
#                  37, 140, 207, 130, 1.5, 1.5])  # Extra value for repeated slider calls
#     mocker.patch('streamlit.sidebar.selectbox', side_effect=[
#                  'M', 'ASY', 'Y', 'ST', 'Down', 'Down'])  # Ensure this matches calls
#     expected_data = {
#         'Age': 37, 'Sex': 'M', 'ChestPainType': 'ASY', 'RestingBP': 140,
#         'Cholesterol': 207, 'FastingBS': 'Y', 'RestingECG': 'ST',
#         'MaxHR': 130, 'ExerciseAngina': 'Y', 'Oldpeak': 1.5, 'STSlope': 'Down'
#     }
#     df_expected = pd.DataFrame(expected_data, index=[0])
#     assert app.user_input_features().equals(df_expected)


# 4. Testing load_data()


def test_load_data(mocker):
    mocker.patch('pandas.read_csv',
                 return_value=pd.DataFrame({'data': [1, 2, 3]}))
    assert app.load_data().equals(pd.DataFrame({'data': [1, 2, 3]}))

# 5. Testing preprocess_data()


def test_preprocess_data():
    input_df = pd.DataFrame({
        'Sex': ['M'],
        'ChestPainType': ['ASY'],
        'FastingBS': ['Y'],
        'RestingECG': ['ST'],
        'ExerciseAngina': ['Y'],
        'STSlope': ['Down']
    })
    df = pd.DataFrame({
        'Sex': ['F'],
        'ChestPainType': ['NAP'],
        'FastingBS': ['N'],
        'RestingECG': ['Normal'],
        'ExerciseAngina': ['N'],
        'STSlope': ['Flat'],
        'HeartDisease': [1]
    })
    # Ensure the test expectation matches the function's output post-fix
    expected_columns = ['Sex_M', 'ChestPainType_ASY', 'FastingBS_Y',
                        'RestingECG_ST', 'ExerciseAngina_Y', 'STSlope_Down']
    result_df = app.preprocess_data(input_df, df)
    assert all(column in result_df.columns for column in expected_columns)


# 6. Testing load_model()


def test_load_model(mocker):
    mocker.patch('pickle.load', return_value='model')
    mocker.patch('builtins.open', mocker.mock_open())
    assert app.load_model() == 'model'

# 7. Testing predict()


def test_predict(mocker):
    model_mock = mocker.Mock()
    model_mock.predict_proba.return_value = [[0.1, 0.9]]
    df = pd.DataFrame({'data': [1]})
    prediction = app.predict(model_mock, df)
    model_mock.predict_proba.assert_called_once_with(df)
    assert prediction == [[0.1, 0.9]]

# 8. Display Function Tests (display_header(), display_user_input(), etc.)


def test_display_header(mocker):
    mocker.patch('streamlit.write')
    mocker.patch('streamlit.markdown')
    app.display_header()
    streamlit.write.assert_called_once_with("# Heart Disease Prediction App")
    streamlit.markdown.assert_called()
