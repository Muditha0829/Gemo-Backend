from flask import Flask, request, jsonify, render_template
from PricePrediction.src.GemstonePricePredictor import GemstonePricePredictor

app = Flask(__name__)

# Create an instance of GemstonePricePredictor
predictor = GemstonePricePredictor()

@app.route('/')
def index():
    # Render the HTML template
    return render_template('index.html')

@app.route('/pricepredict', methods=['POST'])
def predictPrice():
    input_data = request.get_json()
    predicted_price = predictor.calculate_price(input_data)
    return jsonify({'predicted_price': predicted_price})

@app.route('/cutprediction', methods=['POST'])
def predictCut():
    input_data = request.get_json()
    return jsonify({'value':input_data})

@app.route('/colorclarityprediction', methods=['POST'])
def predictColorClarity():
    input_data = request.get_json()
    return jsonify({'value':input_data})

@app.route('/recommendation', methods=['POST'])
def recommend():
    input_data = request.get_json()
    return jsonify({'value':input_data})


if __name__ == '__main__':
    print("The Flask Server is Running. Please try API calls from the Postman for now")
    app.run(host='0.0.0.0', port=5001)  # Make the server accessible on your local network