import cv2
import numpy as np
import tensorflow as tf
import os

class GemIdentificationModel:

    def __init__(self, model_path='../model'):
        # Get the current directory of this script
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Combine the current directory with the model directory
        model_dir = os.path.join(current_dir, model_path)

        # Load the models for clarity, variety, and color
        self.model_clarity = tf.keras.models.load_model(os.path.join(model_dir, 'gem-identification-clarity.h5'))
        self.model_variety = tf.keras.models.load_model(os.path.join(model_dir, 'gem-identification-variety.h5'))
        self.model_color = tf.keras.models.load_model(os.path.join(model_dir, 'gem-identification-color.h5'))

        # Define class dictionaries for clarity, variety, and color
        self.clarity_classes = {
            0: 'I',
            1: 'SI',
            2: 'VS',
            3: 'VVS'
        }
        self.variety_classes = {
            0: 'Alexandrite',
            1: 'Cats Eye',
            2: 'Sapphire',
            3: 'Topaz'
        }
        self.color_classes = {
            0: 'BLACK',
            1: 'BLUE',
            2: 'GREEN',
            3: 'PINK',
            4: 'PURPLE',
            5: 'RED',
            6: 'WHITE',
            7: 'YELLOW'
        }

    def preprocessing_function_xception(self, img):
        # Replace with actual preprocessing for Xception model
        preprocessed_img = tf.keras.applications.xception.preprocess_input(img)
        return preprocessed_img

    def preprocessing_function_resnet(self, img):
        # Replace with actual preprocessing for ResNet model
        preprocessed_img = tf.keras.applications.resnet50.preprocess_input(img)
        return preprocessed_img

    def preprocessing_function_inception_resnet_v2(self, img):
        # Replace with actual preprocessing for Inception-ResNet-v2 model
        preprocessed_img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)
        return preprocessed_img

    def preprocess_image(self, image_path, model_name):
        # Preprocess the image for a specific model
        if model_name == 'clarity':
            preprocessing_function = self.preprocessing_function_xception
        elif model_name == 'variety':
            preprocessing_function = self.preprocessing_function_resnet
        elif model_name == 'color':
            preprocessing_function = self.preprocessing_function_inception_resnet_v2

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = preprocessing_function(img)
        img = np.expand_dims(img, axis=0)

        return img

    def inference_single(self, image_path, model_name):
        if model_name == 'clarity':
            loaded_model = self.model_clarity
            class_dict = self.clarity_classes
        elif model_name == 'variety':
            loaded_model = self.model_variety
            class_dict = self.variety_classes
        elif model_name == 'color':
            loaded_model = self.model_color
            class_dict = self.color_classes

        img = self.preprocess_image(image_path, model_name)

        prediction = loaded_model.predict(img, verbose=0)
        prediction = prediction.squeeze()
        prediction = np.argmax(prediction)

        return class_dict[prediction]

    def identify_gem(self, image_path):
        clarity = self.inference_single(image_path, 'clarity')
        variety = self.inference_single(image_path, 'variety')
        color = self.inference_single(image_path, 'color')

        return {
            "clarity": clarity,
            "variety": variety,
            "color": color
        }


