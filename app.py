from flask import Flask, request, jsonify
from PricePrediction.src.GemstonePricePredictor import GemstonePricePredictor
from gemstone_recommendation import GemstoneRecommendation
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
# Define paths and class_dict_rev
model_path = 'Recommendation/model/query-analysis.h5'
tokenizer_path = 'Recommendation/model/query-analysis.pkl'
max_length = 35
class_dict_rev = {
            0: 'Amethyst',
            1: 'Agate',
            2: 'Alexandrite',
            3: 'Almandine',
            4: 'Amber',
            5: 'Bloodstone',
            6: 'Carnelian',
            7: "Cat's-Eye",
            8: 'Black Chalcedony',
            9: 'Chalcedony',
            10: 'Citrine',
            11: 'Coral',
            12: 'Emerald',
            13: 'Hematite',
            14: 'Hessonite',
            15: 'Iolite',
            16: 'Jade',
            17: 'Kunzite',
            18: 'Moonstone',
            19: 'Morganite',
            20: 'Rose Quartz',
            21: 'Ruby',
            22: 'Sapphire',
            23: 'Topaz',
            24: 'Diamond',
            25: 'Aquamarine',
            26: 'Opal',
            27: 'Onyx',
            28: 'Larimar',
            29: 'Garnet',
            30: 'Peridot',
            31: 'Tourmaline',
            32: 'Turquoise',
            33: 'Lapis Lazuli',
            34: 'Labradorite',
            35: 'Chrysocolla',
            36: 'Rhodonite',
            37: 'Obsidian',
            38: 'Sodalite',
            39: 'Sunstone',
            40: "Tiger's Eye",
            41: 'Amazonite',
            42: 'Smoky Quartz',
            43: 'Apatite',
            44: 'Lepidolite',
            45: 'Black Onyx',
            46: 'Lemon Quartz',
            47: 'Howlite',
            48: 'Prehnite',
            49: 'Moss Agate',
            50: 'Pink Opal',
            51: 'Pietersite',
            52: 'Red Jasper',
            53: 'Blue Lace Agate',
            54: 'Aragonite',
            55: 'Rhodochrosite',
            56: 'Bumblebee Jasper',
            57: 'Dumortierite',
            58: 'Azurite',
            59: 'Serpentine',
            60: 'Blue Apatite',
            61: 'Charoite',
            62: 'Nuummite',
            63: 'Pearl',
            64: 'Malachite',
            65: 'Zircon',
            66: 'Tigers Eye',
            67: 'Selenite',
            68: 'Snowflake Obsidian',
            69: 'Ametrine',
            70: 'Fluorite',
            71: 'Green Aventurine',
            72: 'Blue Chalcedony',
            73: 'Blue Sapphire',
            74: 'Yellow Citrine',
            75: 'Blue Topaz',
            76: 'Green Tourmaline',
            77: 'White Agate',
            78: 'Tanzanite',
            79: 'Red Coral',
            80: 'Yellow Sapphire',
            81: "Cat's Eye",
            82: 'White Sapphire',
            83: 'Yellow Diamond',
            84: 'Blue Zircon',
            85: 'Red Spinel',
            86: 'Yellow Topaz',
            87: 'Red Tourmaline',
            88: 'White Pearl',
            89: 'Red Garnet',
            90: 'Blue Moonstone',
            91: 'Chrysoberyl',
            92: 'Green Jade',
            93: 'Red Agate',
            94: 'Padparadscha',
            95: 'Pink Diamond',
            96: 'Paraiba Tourmaline',
            97: 'Spinel',
            98: 'Rhodolite Garnet',
            99: 'Pink Sapphire',
            100: 'Fancy Color Diamonds',
            101: 'Pink Tourmaline',
            102: 'Padparadscha Sapphire',
            103: 'Clear Quartz',
            104: 'Green Jasper',
            105: 'Green Adventurine'
     }

# Create an instance of the GemstoneRecommendation class
gemstone_recommender = GemstoneRecommendation(model_path, tokenizer_path, max_length, class_dict_rev)

@app.route('/recommendation', methods=['POST'])
def recommendation():
    user_input = request.form['user_input']

    # Get a gemstone recommendation based on user input
    recommendation = gemstone_recommender.get_recommendation(user_input)
    return jsonify({'gemstone_name': recommendation})
    



# Create an instance of GemstonePricePredictor
predictor = GemstonePricePredictor()

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    predicted_price = predictor.calculate_price(input_data)
    return jsonify({'predicted_price': predicted_price})




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # Make the server accessible on your local network
