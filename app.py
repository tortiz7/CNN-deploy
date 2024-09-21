from flask import Flask, request
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    img_path = request.files['file']
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    return "Pneumonia Detected" if prediction[0][0] > 0.5 else "Normal"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
