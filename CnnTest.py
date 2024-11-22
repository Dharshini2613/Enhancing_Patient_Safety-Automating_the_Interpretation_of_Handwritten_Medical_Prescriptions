import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Define the paths to the test images and labels
test_image_folder = 'Data/Testing/testing_words'
test_label_file = 'Data/Testing/testing_labels.csv'

# Load test labels
test_labels_df = pd.read_csv(test_label_file)

# Column names in the CSV file
filename_col = 'IMAGE'
label_col = 'MEDICINE_NAME'

# Debug: print the first few rows to understand the structure
print("First few rows of the test dataframe:")
print(test_labels_df.head())

# Initialize lists for test images and labels
test_images = []
test_labels = []

# Load test images and their corresponding labels
for index, row in test_labels_df.iterrows():
    img_path = os.path.join(test_image_folder, row[filename_col])
    img = load_img(img_path, target_size=(64, 64))  # Resize to a fixed size
    img = img_to_array(img)
    test_images.append(img)
    test_labels.append(row[label_col])

# Convert lists to numpy arrays
test_images = np.array(test_images, dtype="float") / 255.0  # Normalize pixel values
test_labels = np.array(test_labels)

# Encode test labels
label_encoder = LabelEncoder()
test_labels_encoded = label_encoder.fit_transform(test_labels)
test_labels_categorical = tf.keras.utils.to_categorical(test_labels_encoded, num_classes=78)  # 78 classes

# Load the trained model
model = load_model('model.h5')

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_images, test_labels_categorical)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Optionally: Predict and show a confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

# Predict the classes
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels_categorical, axis=1)
print(true_classes)

# Generate classification report
print(classification_report(true_classes, predicted_classes, target_names=label_encoder.classes_))

# Generate confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:")
print(conf_matrix)