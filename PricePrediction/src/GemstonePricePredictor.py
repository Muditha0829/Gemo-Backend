import joblib
import pandas as pd
from sklearn.preprocessing import  OrdinalEncoder
import os


class GemstonePricePredictor:
    def __init__(self, model_path='../model/random_forest_model.pkl'):
        # Get the current directory of this script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Combine the current directory with the model filename
        model_path = os.path.join(current_dir, model_path)

        self.loaded_model = joblib.load(model_path)

    def calculate_price(self, input_data):
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Perform data preprocessing
        cut_encoder = OrdinalEncoder(categories=[['Oval', 'Emerald', 'Heart', 'Peer']])
        color_encoder = OrdinalEncoder(
            categories=[['Blue', 'Pink', 'Yellow', 'Gold', 'Purple', 'White', 'Red', 'Green', 'Brown']])
        clarity_encoder = OrdinalEncoder(categories=[['Transparent', 'Translucent']])
        gemstone_encoder = OrdinalEncoder(
            categories=[['Spinel', "Cat's Eye", "Sapphire", "Ruby", "Zircon", "Topaz", "Alexandrite"]])

        input_df['Cut'] = cut_encoder.fit_transform(input_df[['Cut']])
        input_df['Color'] = color_encoder.fit_transform(input_df[['Color']])
        input_df['Clarity'] = clarity_encoder.fit_transform(input_df[['Clarity']])
        input_df['GemstoneName'] = gemstone_encoder.fit_transform(input_df[['GemstoneName']])

        # Perform prediction
        predicted_price = self.loaded_model.predict(input_df)[0]
        return predicted_price


