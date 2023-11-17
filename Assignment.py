import streamlit as st
import pandas as pd,pickle
from keras.models import load_model
from keras import backend as K
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load your pre-trained Keras model
model_path = 'new_model.h5'
model = load_model(model_path)
with open("scaler.pkl",'rb') as scale:
        scaling=pickle.load(scale)
print(model.summary())
# Mapping for categorical features
contract_mapping = {'Month-to-Month': 0, 'One year': 1, 'Two year': 2}
payment_method_mapping = {'Electronic': 2, 'Check': 2, 'Mailed Check': 3, 'Bank transaction': 0, 'Credit card': 1}
gender_mapping = {'Female': 0, 'Male': 1}
internet_service_mapping = {'DSL': 0, 'Fibre option': 1, 'No': 2}
online_backup_mapping = {'Yes': 2, 'No': 2, 'No internet': 1}
online_security_mapping = {'Yes': 0, 'No': 2, 'No internet': 1}
tech_support_mapping = {'No': 0, 'Yes': 2, 'No internet': 1}

# Streamlit app
def main():
    st.title('Churn Prediction App')

    # Allow users to input values for features
    total_charges = st.number_input('Enter TotalCharges:', min_value=0.0)
    monthly_charges = st.number_input('Enter MonthlyCharges:', min_value=0.0)
    tenure = st.number_input('Enter tenure:', min_value=0.0)

    contract = st.selectbox('Select Contract:', list(contract_mapping.keys()))
    payment_method = st.selectbox('Select Payment Method:', list(payment_method_mapping.keys()))
    gender = st.selectbox('Select Gender:', list(gender_mapping.keys()))
    internet_service = st.selectbox('Select InternetService:', list(internet_service_mapping.keys()))
    online_backup = st.selectbox('Select OnlineBackup:', list(online_backup_mapping.keys()))
    online_security = st.selectbox('Select OnlineSecurity:', list(online_security_mapping.keys()))
    tech_support = st.selectbox('Select TechSupport:', list(tech_support_mapping.keys()))
    print("Hello Lets'  check here",tech_support)
    print("Hello")
    # Prepare input values
    input_values = {
        'TotalCharges': total_charges,
        'MonthlyCharges': monthly_charges,
        'tenure': tenure,
        'Contract': contract_mapping[contract],
        'PaymentMethod': payment_method_mapping[payment_method],
        'OnlineSecurity': online_security_mapping[online_security],
        'TechSupport': tech_support_mapping[tech_support],
        'gender': gender_mapping[gender],
        'InternetService': internet_service_mapping[internet_service],
        'OnlineBackup': online_backup_mapping[online_backup]
    }

    # Convert input values to a DataFrame
    # input_df = pd.DataFrame([input_values])

    # Make prediction using the loaded Keras model
    # prediction = model.predict(np.array(input_df))

    # Display prediction
    # if prediction[0, 0] > 0.5:
    #     st.error('Prediction: Customer will churn.')
    # else:
    #     st.success('Prediction: Customer will not churn.')
    

   
    if st.button('Predict'):
        number_columns=scaling.transform(np.array([0,tenure,monthly_charges,total_charges]).reshape(-1, 1))
        print('hi')
        print(number_columns)
        print('hi2')
        tenure=number_columns[0][0]
        monthly_charges=number_columns[1][0]
        total_charges=number_columns[2][0]
        prediction = model.predict([tenure,monthly_charges,total_charges,contract,payment_method,gender,internet_service,online_backup,online_security,tech_support])
        print(prediction)
        # if prediction > 0.5:
        #     st.success('Prediction: Customer will not churn.')


if __name__ == '__main__':
    main()