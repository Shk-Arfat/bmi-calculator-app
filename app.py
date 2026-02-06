from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model and encoder
model = pickle.load(open('bmi_model.pkl', 'rb'))
label_decoder = pickle.load(open('label_encoder.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    bmi_value = None

    if request.method == 'POST':
        try:
            gender = 1 if request.form['gender'] == 'Male' else 0
            age = int(request.form['age'])
            height = float(request.form['height'])  # cm
            weight = float(request.form['weight'])  # kg
        except ValueError:
            return render_template('index.html', result="Invalid input")

        # Predict
        features = np.array([[gender, age, height, weight]])
        prediction_encoded = model.predict(features)[0]
        prediction = label_decoder.inverse_transform([prediction_encoded])[0]

        # Calculate BMI manually
        height_m = height / 100
        bmi_value = weight / (height_m ** 2)

        if bmi_value < 18.5:
            category = "Underweight"
        elif 18.5 <= bmi_value < 25:
            category = "Normal"
        elif 25 <= bmi_value < 30:
            category = "Overweight"
        else:
            category = "Obese"

        result = {
            'bmi': round(bmi_value, 2),
            'rule_category': category,
            'model_prediction': prediction
        }

    return render_template('index.html', result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
