import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

class GemstoneRecommendation:
    def __init__(self):
        # Set the current working directory to the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)

        # Load the Keras model
        model_path = os.path.join(script_dir, '../model/query-analysis.h5')
        self.model = load_model(model_path)

        # Load the tokenizer using Pickle
        tokenizer_path = os.path.join(script_dir, '../model/query-analysis.pkl')
        with open(tokenizer_path, 'rb') as tokenizer_file:
            self.tokenizer = pickle.load(tokenizer_file)

        nltk.download("stopwords")
        self.stopwords_list = set(stopwords.words("english"))

        self.class_dict_rev = {
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

    # Preprocessing functions
    def lemmatization(self, lemmatizer, sentence):
        lem = [lemmatizer.lemmatize(k) for k in sentence]
        return [k for k in lem if k]

    def remove_stop_words(self, stopwords_list, sentence):
        return [k for k in sentence if k not in stopwords_list]

    def preprocess_one(self, description):
        description = description.lower()
        remove_punc = re.sub(r'[^\w\s]', '', description)  # Remove punctuations
        remove_num = re.sub(r'[0-9]', '', remove_punc)  # Remove numbers
        words = remove_num.split()
        lemmatizer = nltk.WordNetLemmatizer()
        lemmatized = self.lemmatization(lemmatizer, words)  # Word Lemmatization
        remove_stop = self.remove_stop_words(self.stopwords_list, lemmatized)  # Remove stop words
        updated_description = ' '.join(remove_stop)
        return updated_description

    def preprocess_data(self, descriptions):
        if isinstance(descriptions, str):
            descriptions = [descriptions]
        updated_descriptions = [self.preprocess_one(description) for description in descriptions]
        return updated_descriptions

    def inference_gemstone(self, description):
        description = self.preprocess_data(description)
        description = self.tokenizer.texts_to_sequences(description)
        description = tf.keras.preprocessing.sequence.pad_sequences(description, maxlen=35, padding='pre', truncating='pre')
        pred = self.model.predict(description)
        pred = np.argmax(pred, axis=1)
        return self.class_dict_rev[pred[0]]

    def recommend_gemstone(self, user_input):
        try:
            # Extract the text from the user input JSON
            description = user_input.get('user_input')
            recommendation = self.inference_gemstone(description)
            return {'recommendation': recommendation}
        except Exception as e:
            return {'error': str(e)}
