from keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

class GemstoneCutPredictor:
    def __init__(self,model_path='../model'):
        # Get the directory of this script (src directory)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_directory = os.path.join(current_dir, model_path)
        self.class_labels = ['oval', 'square', 'trilliant']
        self.model = None

    def load_model(self):
        # Load your trained Xception model from the specified model directory
        model_path = os.path.join(self.model_directory, 'xception_model.h5')
        self.model = load_model(model_path)

    def preprocess_image(self, image_path):
        # Load and preprocess the image
        img = Image.open(image_path)
        img = img.resize((299, 299))  # Resize to match model input size
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = tf.keras.applications.xception.preprocess_input(img)
        return img

    def predict_cut(self, image_path):
        if self.model is None:
            self.load_model()  # Load the model if not already loaded

        # Preprocess the image
        preprocessed_img = self.preprocess_image(image_path)

        # Make predictions using the model
        predictions = self.model.predict(preprocessed_img)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = self.class_labels[predicted_class]
        return predicted_label
