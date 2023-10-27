from flask import Flask, request, jsonify
from PricePrediction.src.GemstonePricePredictor import GemstonePricePredictor
from Recommendation.src.GemstoneRecommendation import GemstoneRecommendation 

app = Flask(__name__)

# Create an instance of GemstoneRecommendation
gemstone_recommender = GemstoneRecommendation()

# Create an instance of GemstonePricePredictor
predictor = GemstonePricePredictor()

@app.route('/gemstonerecommendation', methods=['POST'])
def recommend_gemstone():
    try:
        user_input = request.get_json()
        # Use the GemstoneRecommendation instance to make gemstone recommendations
        recommendation = gemstone_recommender.recommend_gemstone(user_input)
        return jsonify(recommendation)
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/priceprediction', methods=['POST'])
def predict():
    input_data = request.get_json()
    predicted_price = predictor.calculate_price(input_data)
    return jsonify({'predicted_price': predicted_price})




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # Make the server accessible on your local network
