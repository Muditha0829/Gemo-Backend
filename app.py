from flask import Flask, request, jsonify
from PricePrediction.src.GemstonePricePredictor import GemstonePricePredictor

app = Flask(__name__)

# Create an instance of GemstonePricePredictor
predictor = GemstonePricePredictor()

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    predicted_price = predictor.calculate_price(input_data)
    return jsonify({'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # Make the server accessible on your local network
