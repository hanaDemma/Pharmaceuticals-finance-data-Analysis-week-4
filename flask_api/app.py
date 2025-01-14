from flask import Flask, request, jsonify
from flask_api.model import load_model
from .preprocess import preprocess_input


# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and scaler
model, scaler = load_model('/D:\week-4\Pharmaceuticals-finance-data-Analysis-week-4\models\model-2025-01-13-10-06-10.pkl', '/D:\week-4\Pharmaceuticals-finance-data-Analysis-week-4\models\model-2025-01-13-13-11-53.pkl')

# Define an endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    request_data = request.get_json()
    
    # Preprocess the input data
    features_scaled = preprocess_input(request_data, scaler)
    
    # Make predictions
    predictions = model.predict(features_scaled)
    
    # Format predictions for response
    response = {'predictions': predictions.tolist()}  # Convert numpy array to list
    return jsonify(response)
    
# Run the app
if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False in production