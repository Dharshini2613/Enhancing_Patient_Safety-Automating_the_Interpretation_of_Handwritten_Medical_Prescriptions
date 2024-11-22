import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

import joblib

# Load the saved model and label encoder
model = load_model('model.h5')
label_encoder = joblib.load('label_encoder.pkl')

# Prepare a new input image
def prepare_image(img_path):
    img = load_img(img_path, target_size=(64, 64))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Example usage
img_path = 'Data/Training/training_words/0.png'
input_image = prepare_image(img_path)

# Make a prediction
predictions = model.predict(input_image)
predicted_class_index = np.argmax(predictions, axis=1)
predicted_class_label = label_encoder.inverse_transform(predicted_class_index)

print(f'Predicted class label: {predicted_class_label[0]}')
