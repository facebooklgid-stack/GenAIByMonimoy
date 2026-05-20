import tensorflow as tf
import streamlit as st
import pandas as pd
import pickle


model = tf.keras.models.load_model('salary_regression_model.h5')


with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file) 
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file) 
with open('onehot_encoder_geography.pkl', 'rb') as file:
    onehot_encoder_geography = pickle.load(file) 


## Streamlit app
st.title('Salary Regression Prediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geography.categories_[0])
age = st.slider('Age', 18, 65)
gender = st.selectbox('Gender', ['Male', 'Female'])
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
exited = st.selectbox('Exited', [0, 1])



# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})


# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))

# Combine input data with one-hot encoded 'Geography'
input_data = pd.concat([input_data, geo_encoded_df], axis=1)
# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict salary
predicted_salary = model.predict(input_data_scaled)
st.subheader('Predicted Salary')
st.write(f"${predicted_salary[0][0]:.2f}")