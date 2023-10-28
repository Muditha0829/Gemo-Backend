from flask import Flask, request, jsonify, render_template
import os

from PricePrediction.src.GemstonePricePredictor import GemstonePricePredictor
from Recommendation.src.GemstoneRecommendation import GemstoneRecommendation 
from ColorClarityIdentification.src.ColorClarityIdentification import GemIdentificationModel

app = Flask(__name__)

# Create an instance of GemstoneRecommendation
gemstone_recommender = GemstoneRecommendation()

# Create an instance of GemstonePricePredictor
predictor = GemstonePricePredictor()

# Create an instance of colorclarityidentificator
colorclarityidentificator = GemIdentificationModel()

@app.route('/')
def index():
    # Render the HTML template
    return render_template('index.html')


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

@app.route('/colorclarityidentification', methods=['POST'])
def identify_gem():
    try:
        # Check if the 'image' field is in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        # Get the uploaded image file
        image_file = request.files['image']

        # Save the image temporarily
        temp_image_path = "temp_image.png"
        image_file.save(temp_image_path)

        # Call the gem identification method with the temporary image path
        identification_result = colorclarityidentificator.identify_gem(temp_image_path)

        # Remove the temporary image after processing
        os.remove(temp_image_path)

        return jsonify(identification_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500





if __name__ == '__main__':
    print("The Flask Server is Running. Please try API calls from the Postman for now")
    app.run(host='0.0.0.0', port=5001)  # Make the server accessible on your local network