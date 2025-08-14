import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load trained model
with open('regmodel.pkl', 'rb') as f:
    regmodel = pickle.load(f)

# Load fitted scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print("Model and scaler loaded successfully!")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.json['data']
    print("Received data:", data)

    # Convert data to numpy array
    input_array = np.array(list(data.values())).reshape(1, -1)
    print("Input array:", input_array)

    # Transform using scaler
    new_data = scaler.transform(input_array)

    # Predict
    output = regmodel.predict(new_data)
    print("Prediction:", output[0])

    return jsonify({'prediction': float(output[0])})

if __name__ == '__main__':
    app.run(debug=True)
