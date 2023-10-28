import os
import cv2
import numpy as np
import tensorflow as tf

class GemIdentificationModel:
    def __init__(self, model_path='../model'):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, model_path)

        self.model_clarity = self.load_model(model_dir, 'gem-identification-clarity.h5')
        self.model_variety = self.load_model(model_dir, 'gem-identification-variety.h5')
        self.model_color = self.load_model(model_dir, 'gem-identification-color.h5')

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

    def load_model(self, model_dir, model_file):
        model_path = os.path.join(model_dir, model_file)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_file} not found in {model_dir}")
        return tf.keras.models.load_model(model_path)

    def preprocess_image(self, image_path, model_name):
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
    
    """ def preprocessing_function_xception(img):
        img = tf.keras.applications.xception.preprocess_input(img)
        return img

    def preprocessing_function_resnet(img):
        img = tf.keras.applications.resnet50.preprocess_input(img)
        return img

    def preprocessing_function_inception_resnet_v2(img):
        img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)
        return img
 """

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

        print(clarity,variety,color)

        return {
            "clarity": clarity,
            "variety": variety,
            "color": color
        }
