
import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('logistic_regression_model.pkl')

# Streamlit app title
st.title("Titanic Survival Prediction")

# Collect user inputs for prediction
age = st.number_input('Age', min_value=1, max_value=100, value=25)
pclass = st.selectbox('Pclass', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
sibsp = st.number_input('SibSp', min_value=0, max_value=10, value=0)
parch = st.number_input('Parch', min_value=0, max_value=10, value=0)
fare = st.number_input('Fare', min_value=0, max_value=500, value=10)
embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'Age': [age],
    'Pclass': [pclass],
    'Sex_male': [1 if sex == 'male' else 0],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked_C': [1 if embarked == 'C' else 0],
    'Embarked_Q': [1 if embarked == 'Q' else 0],
    'Embarked_S': [1 if embarked == 'S' else 0]
})

# Make a prediction
prediction = model.predict(input_data)
st.write("Survival Prediction: ", "Survived" if prediction[0] == 1 else "Not Survived")
