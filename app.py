from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        availability = request.form['availability']
        total_sqft = float(request.form['total_sqft'])
        bath = int(request.form['bath'])
        balcony = int(request.form['balcony'])
        bhk = int(request.form['BHK'])
        area_type = request.form['area_type']

        # Prepare feature array for model
        features = [
            1 if availability == "Ready to move in" else 0,
            total_sqft,
            bath,
            balcony,
            bhk,
            1 if area_type == "Carpet Area" else 0,
            1 if area_type == "Plot Area" else 0,
            1 if area_type == "Super built-up Area" else 0
        ]
        final_features = np.array([features])

        # Make prediction
        prediction = model.predict(final_features)
        predicted_price = prediction[0]

        return render_template('index.html', prediction_text=f'Predicted Price in Lakhs: ₹{predicted_price:,.2f}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
