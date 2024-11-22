import os
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define the paths to the images and labels
image_folder = 'Data/Training/training_words'
label_file = 'Data/Training/training_labels.csv'

# Load labels
labels_df = pd.read_csv(label_file)

# Column names in the CSV file
filename_col = 'IMAGE'
label_col = 'MEDICINE_NAME'

# Debug: print the first few rows to understand the structure
print("First few rows of the dataframe:")
print(labels_df.head())

# Initialize lists for images and labels
images = []
labels = []

# Load images and their corresponding labels
for index, row in labels_df.iterrows():
    img_path = os.path.join(image_folder, row[filename_col])
    img = load_img(img_path, target_size=(64, 64))  # Resize to a fixed size
    img = img_to_array(img)
    images.append(img)
    labels.append(row[label_col])

# Convert lists to numpy arrays
images = np.array(images, dtype="float") / 255.0  # Normalize pixel values
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

import joblib

joblib.dump(label_encoder, 'label_encoder.pkl')

labels_categorical = to_categorical(labels_encoded, num_classes=78)  # 78 classes
print(labels_categorical)
# Split the data
X_train, X_test, y_train, y_test = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

# Build the model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(78, activation='softmax'))  # 78 classes

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save the model in the newer format
model.save('model.h5')