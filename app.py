import numpy as np
from flask import Flask, render_template, request
import pickle

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb')) 

# Initializing the Flask Application
app = Flask(__name__)

# Create a route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route(When a form is submitted, this route will be accessed.)
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        cgpa = float(request.form['cgpa'])
        iq = float(request.form['iq'])
        
        # Scale the input using the loaded scaler
        input_data = scaler.transform(np.array([[cgpa, iq]]))

        print(f"Raw Input: CGPA: {cgpa}, IQ: {iq}")  
        print(f"Scaled Input: {input_data}")  

        # Make prediction
        prediction = model.predict(input_data)

        # Interpret the result
        result = "Placement will happen" if prediction[0] == 1 else "No placement"
        print(f"Prediction: {result}") 
        
        return render_template('index.html', prediction=result)
# This checks if the script is being run directly 
if __name__ == '__main__':
    app.run(debug=True)
