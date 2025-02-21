import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title='Car Price Prediction App', layout='centered')

# Load the trained model
model = joblib.load('car_price_model.pkl')

st.title('Car Price Prediction App')
st.markdown('### Predict the selling price of a car based on its features')

# Sidebar for user input
st.sidebar.header('Input Car Details')

# Function to get user input
def get_user_input():
    fuel_type = st.sidebar.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
    seller_type = st.sidebar.selectbox('Seller Type', ['Dealer', 'Individual'])
    transmission = st.sidebar.selectbox('Transmission', ['Manual', 'Automatic'])
    car_age = st.sidebar.slider('Car Age', 0, 20, 5)
    present_price = st.sidebar.number_input('Present Price (in lakhs)', 0.0, 50.0, 5.0)
    kms_driven = st.sidebar.number_input('Kilometers Driven', 0, 200000, 30000)
    owner = st.sidebar.selectbox('Number of Previous Owners', [0, 1, 2, 3])
    
    # Create a dataframe from user input
    user_data = {
        'Fuel_Type': [fuel_type],
        'Seller_Type': [seller_type],
        'Transmission': [transmission],
        'Car_Age': [car_age],
        'Present_Price': [present_price],
        'Kms_Driven': [kms_driven],
        'Owner': [owner]
    }
    return pd.DataFrame(user_data)

# Display user input
user_input = get_user_input()
st.write('### Car Details Provided by User:')
st.dataframe(user_input)

# Prediction
if st.button('Predict Car Price'):
    prediction = model.predict(user_input)
    st.success(f'Estimated Selling Price: ₹{prediction[0]:.2f} lakhs')

# Footer
st.markdown('---')
st.markdown('Developed with ❤️ using Streamlit')
