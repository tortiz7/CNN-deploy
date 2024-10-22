import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Download ResNet50(pre-trained CNN on ImageNet, great for image classification)
from tensorflow.keras.applications import ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Download dataset from S3 
# aws s3 cp s3://x-raysbucket/chest_xray/ /home/ubuntu/chest_xray --recursive

# Add classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)  # This reduces the 7x7x2048 output to 2048
x = Dense(512, activation='relu')(x)  # Add a dense layer to help with feature processing
x = Dense(1, activation='sigmoid')(x)  # Final layer for binary classification

model = Model(inputs=base_model.input, outputs=x)
# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
# Load the data
train_generator = train_datagen.flow_from_directory(
    '/home/ubuntu/chest_xray/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    '/home/ubuntu/chest_xray/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    '/home/ubuntu/chest_xray/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)


# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=5,
    callbacks=callbacks
)

# Fine-tune the pre-trained CNN model with data
# model.fit(
#    train_generator,
#    steps_per_epoch=train_generator.samples // train_generator.batch_size,
#    validation_data=val_generator,
#    validation_steps=val_generator.samples // val_generator.batch_size,
#    epochs=5
#)
# Save the model after training
# model.save('pneumonia_model.h5')

model.save('pneumonia_model.keras')  # Using .keras format instead of .h5
# Load the trained model for inference
from tensorflow.keras.models import load_model
# Test prediction
test_image_path = '/home/ubuntu/chest_xray/test/PNEUMONIA/person10_virus_35.jpeg'  # Update with actual image path
img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.
img_array = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_array)
confidence = float(prediction[0][0])
result = "Pneumonia" if confidence > 0.5 else "Normal"
print(f"Prediction: {result} (confidence: {confidence:.2%})")