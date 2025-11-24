
I designed and implemented a machine-learning based system capable of distinguishing between vitiligo/leucoderma-affected skin and healthy skin with high accuracy.with the help of the data set I mentioned below 
<img width="1470" height="956" alt="Screenshot 2025-11-24 at 5 17 04 PM" src="https://github.com/user-attachments/assets/c3dc5600-f3a5-4289-80ad-acf9a42268ea" />
<img width="1470" height="956" alt="Screenshot 2025-11-24 at 5 17 12 PM" src="https://github.com/user-attachments/assets/acef64fe-77d9-4eaf-84d2-1cb094177907" />
<img width="1470" height="956" alt="Screenshot 2025-11-24 at 5 17 23 PM" src="https://github.com/user-attachments/assets/90a4125c-5708-42bd-9d53-d1cc3d862874" />


## Overview
A deep learning solution to classify skin images as 'vitiligo' or 'normal', aiding early diagnosis.

## Features
- Automatic image classification
- Model persistence
- Prediction from user-provided images

## Technologies
- Python, TensorFlow, Keras, Google Colab

## Installation & Usage
1. Clone the repository
2. Upload dataset to Google Drive
3. Update dataset paths in the code
4. Run notebook or Python script
5. For predictions, use provided code with new image paths

## Instructions for Testing
- Use test images from each class
- Check prediction accuracy

## Screenshots of code output and accuracy 
<img width="1470" height="956" alt="Screenshot 2025-11-24 at 5 28 20 PM" src="https://github.com/user-attachments/assets/68a5841f-c8d5-428d-9b0e-ee8a6bb5413a" />
<img width="1470" height="956" alt="Screenshot 2025-11-24 at 5 28 25 PM" src="https://github.com/user-attachments/assets/1cbc2a4a-1431-40b1-8daa-bd812d11c426" />
<img width="1470" height="956" alt="Screenshot 2025-11-24 at 5 28 28 PM" src="https://github.com/user-attachments/assets/11dd4481-ba58-4ce0-9080-773521d9b9f2" />

##CODE
from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32


train_dir = '/content/drive/MyDrive/cse'
val_dir = '/path/to/validation_data' 


train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary', 
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary', 
    subset='validation'
)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=IMAGE_SIZE + (3,)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)


model.save('vitiligo_detector_model.h5')




