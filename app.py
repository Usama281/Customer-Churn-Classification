import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


model = load_model('artifacts/model.h5')

with open('artifacts/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('artifacts/gender_encoder.pkl', 'rb') as file:
    gender_encoder = pickle.load(file)

with open('artifacts/geography_encoder.pkl', 'rb') as file:
    geography_encoder = pickle.load(file)


st.title('Customer Churn Classification using Artificial Neural Network')

geography = st.selectbox('Geography', geography_encoder.categories_[0])
gender = st.selectbox('Gender', gender_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0,10)
num_of_products = st.slider('Number of products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame({
    'Geography': [geography],
    'CreditScore': [credit_score],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

input_data['Gender'] = gender_encoder.transform([gender])[0]

geography_array = geography_encoder.transform([[geography]]).toarray()
geography_cols = geography_encoder.get_feature_names_out(['Geography'])
geography_df = pd.DataFrame(geography_array, columns=geography_cols)

input_data = pd.concat([input_data.drop(['Geography'], axis=1), geography_df], axis=1)

scaled_data = scaler.transform(input_data)
prediction = model.predict(scaled_data)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba>0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')