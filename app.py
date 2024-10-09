import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset and train the model
file_path = 'FuelEconomy.csv'
data = pd.read_csv(file_path)
data = data.dropna(subset=['Horse Power', 'Fuel Economy (MPG)'])
X = data[['Horse Power']]
y = data['Fuel Economy (MPG)']

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Save the trained model to a file
joblib.dump(model, 'model.pkl')

# Load the model from the file
model = joblib.load('model.pkl')

# Streamlit code for the web interface
st.title('Fuel Economy Prediction')
st.write('Enter the horse power of the vehicle to predict the fuel economy (MPG):')

# Input field for the user to enter the horse power
horse_power = st.number_input('Horse Power', min_value=0.0, step=0.1)

# Button to trigger the prediction
if st.button('Predict'):
    if horse_power > 0:
        prediction = model.predict([[horse_power]])
        st.success(f'Fuel Economy (MPG): {prediction[0]:.2f}')
    else:
        st.error('Please enter a positive value for Horse Power.')
