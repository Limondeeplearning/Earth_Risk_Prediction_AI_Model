from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    age = int(request.form['age'])
    health = label_encoders['Health Status'].transform([request.form['health']])[0]
    floor = int(request.form['floor'])
    location = label_encoders['Emergency Exit'].transform([request.form['location']])[0]
    protocol = label_encoders['Earthquake Safety Protocol'].transform([request.form['protocol']])[0]
    intensity = int(request.form['Earthquake Intensity'])

    # Make prediction
    input_data = pd.DataFrame([{
        'Age': age,
        'Health Status': health,
        'Floor Assignment': floor,
        'Emergency Exit': location,
        'Earthquake Safety Protocol': protocol,
        'Earthquake Intensity': intensity,
    }])

    prediction = model.predict(input_data)[0]
    status = label_encoders['Status'].inverse_transform([prediction])[0]
    risk = "High Risk" if status == 'Dead' else "Low Risk"

    return render_template('index.html', name=name, risk=risk)

if __name__  == '__main__':
    app.run(debug=True)

