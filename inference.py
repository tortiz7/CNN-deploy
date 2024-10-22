import redis
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import os

# Connect to Redis
r = redis.Redis(host='<redis_server_private_ip>', port=6379, db=0)

# Load the model
model = load_model('pneumonia_model.keras')

# Directory containing test images
test_dir = '/home/ubuntu/chest_xray/test/PNEUMONIA'

# Process each image in the directory
for filename in os.listdir(test_dir):
    if filename.endswith(('.jpeg', '.jpg', '.png')):
        try:
            # Full path to image
            image_path = os.path.join(test_dir, filename)

            # Load and preprocess image
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            prediction = model.predict(img_array, verbose=0)
            confidence = float(prediction[0][0])
            result = "Pneumonia" if confidence > 0.5 else "Normal"

            # Store result in Redis
            r.hset(filename, mapping={
                'prediction': result,
                'confidence': confidence
            })

            # Print result
            print(f"File: {filename}")
            print(f"Prediction: {result} (confidence: {confidence:.2%})")
            print("-" * 50)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
