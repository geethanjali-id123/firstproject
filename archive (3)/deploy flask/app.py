from flask import Flask, request, render_template
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.sav', 'rb'))

# Load the standard scaler
scaler = StandardScaler()
z = np.load('scaled-input.npy')
scaler.fit_transform(z)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from form
        input_features = [
            float(request.form['same_srv_rate']),
            float(request.form['dst_host_same_srv_rate']),
            float(request.form['dst_host_srv_rerror_rate']),
            float(request.form['dst_host_rerror_rate']),
            float(request.form['rerror_rate']),
            float(request.form['srv_rerror_rate']),
            float(request.form['dst_host_srv_count']),
            float(request.form['count']),
            float(request.form['diff_srv_rate']),
            float(request.form['dst_host_diff_srv_rate'])
        ]

        # Ensure the input features are in the correct order and format
        input_features = np.array(input_features).reshape(1, -1)

        # Scale the input features
        scaled_features = scaler.transform(input_features)

        # Make predictions using the loaded decision tree model
        prediction = model.predict(scaled_features)

        # Return the prediction
        prediction_text = 'Attack' if prediction[0] == 1 else 'Normal'
        return render_template('index.html', prediction_text=f'Model Prediction: {prediction_text}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)