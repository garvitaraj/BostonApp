from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    prediction = model.predict([data])
    return render_template('index.html', prediction_text=f"Predicted Price: ${{round(prediction[0], 2)}}")

if __name__ == '__main__':
    app.run(debug=True)
