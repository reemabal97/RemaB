from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

app = Flask(__name__)

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        horse_power = float(request.form['horse_power'])
        if horse_power <= 0:
            return render_template('index.html', prediction_text="Please enter a positive value for Horse Power.")
        prediction = model.predict([[horse_power]])
        return render_template('index.html', prediction_text=f'Fuel Economy (MPG): {prediction[0]:.2f}')
    except ValueError:
        return render_template('index.html', prediction_text="Please enter a valid number for Horse Power.")

if __name__ == "__main__":
    app.run(debug=True)

