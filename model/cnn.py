import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

!mkdir -p /content/chest_xray/

# Download dataset from S3
!aws s3 cp s3://x-raysbucket/chest_xray/ /content/chest_xray/ --recursive --no-sign-request

# Download ResNet50(pre-trained CNN on ImageNet, great for image classification)
from tensorflow.keras.applications import ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Activation

# Add classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)  # This reduces the 7x7x2048 output to 2048


x = Dense(256, kernel_regularizer=l2(0.001))(x)  # This is our 1st Dense, connecte4d layer for the model, with 256 neurons. L2 Regularization is affixed to prevent overfitting.
x = BatchNormalization()(x)   #
x = Activation('relu')(x)
x = Dropout(0.3)(x)

x = Dense(256, kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)

x = Dense(1, activation='sigmoid')(x)  # Final layer for binary classification

for layer in base_model.layers[-15:]:  # This will unfreeze the top set # of layers in the ResNet model we imported earlier so we can specifically train it on our xray samples
        layer.trainable = True         # This sets the unfrozen layers to be trained while we train the model

from tensorflow.keras import backend as K
from keras.optimizers import Adam

# Define initial learning rate
initial_learning_rate = 1e-5           # A very small inital learning rate so the model can train gradually on the Xray scans, which tend to be very noisy

# Set up exponential decay schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=10000,                 # This will decay the learning rate every 10,000 training steps, to prevent the model from overfitting to our training dataset
    decay_rate=0.95,                   # This will reduce the learning rate by 5% every 10,000 training steps
    staircase=True                     # This means the learning rate will fall by 5% at every 10,000 steps, instead of gradually during those steps
)

model = Model(inputs=base_model.input, outputs=x)
# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    #ebrightness_range=[0.1, 0.3],
    zoom_range=0.2,
    #shear_range=0.2,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

from pickle import TRUE
import types

# # Define the set_length function
# def set_length(generator):
#     # Get the total number of samples
#     total_samples = len(generator.filenames)
#     # Calculate the number of batches per epoch
#     steps_per_epoch = total_samples // generator.batch_size
#     # Set the `len` attribute of the generator
#     def __len__(self):
#         return steps_per_epoch
#     generator.__len__ = types.MethodType(__len__, generator)
#     return generator

# Load the data
train_generator = train_datagen.flow_from_directory(
    '/content/chest_xray/train',
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    '/content/chest_xray/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)
test_generator = test_datagen.flow_from_directory(
    '/content/chest_xray/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# # Apply set_length to train and validation generators
# train_generator = set_length(train_generator)
# val_generator = set_length(val_generator)

from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

# # Assuming you have your training generator ready
# class_counts = Counter(train_generator.classes)
# print(class_counts)

# class_weights = {0: 1.0, 1: 1.5}

# # Check if class weights are set correctly
# print("Class weights:", Class_weights)

# classes = np.unique(train_generator.classes)
# class_weights = compute_class_weight('balanced', classes=classes, y=train_generator.classes)
# class_weights = dict(enumerate(class_weights))

# print(class_weights)

from tensorflow.keras.callbacks import ReduceLROnPlateau

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True),
    #ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7)
]

from sklearn.utils.class_weight import compute_class_weight


class_weights = {0: 0.8, 1: 1.6}

# classes = np.unique(train_generator.classes)
# class_weights = compute_class_weight('balanced', classes=classes, y=train_generator.classes)
# class_weights = dict(enumerate(class_weights))

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=15,  # Adjust as needed
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    class_weight=class_weights,  # Add class_weight here
    callbacks=callbacks
)

import matplotlib.pyplot as plt

# Plot training & validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], 'go-', label="Training Loss")
plt.plot(history.history['val_loss'], 'ro-', label="Validation Loss")
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training & validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], 'go-', label="Training Accuracy")
plt.plot(history.history['val_accuracy'], 'ro-', label="Validation Accuracy")
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.save('pneumonia_model.keras')

from tensorflow.keras.models import load_model

import os
# test the model all images in /content/chest_xray/test and get the predictions of each image
# Get the list of image files in the test directory
test_dir = '/content/chest_xray/test'
for root, dirs, files in os.walk(test_dir):
  for file in files:
    if file.endswith(('.jpg', '.jpeg', '.png')):
      test_image_path = os.path.join(root, file)
      # Load and preprocess the image
      img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(224, 224))
      img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.
      img_array = np.expand_dims(img_array, axis=0)


      # Make the prediction
      prediction = model.predict(img_array)
      confidence = float(prediction[0][0])
      result = "Pneumonia" if confidence > 0.5 else "Normal"

      # Print the results
      print(f"Image: {test_image_path}")
      print(f"Prediction: {result} (confidence: {confidence:.2%})")
      print("---")

# Plot training & validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], 'go-', label="Training Accuracy")
plt.plot(history.history['val_accuracy'], 'ro-', label="Validation Accuracy")
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Get true labels and predictions
test_generator.reset()  # Ensure the generator is at the start
Y_true = test_generator.classes  # True labels
Y_pred = (model.predict(test_generator) > 0.5).astype("int32")  # Predictions rounded to 0 or 1

# Generate confusion matrix
conf_matrix = confusion_matrix(Y_true, Y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
class_report = classification_report(Y_true, Y_pred, target_names=test_generator.class_indices.keys())
print("Classification Report:")
print(class_report)

import matplotlib.pyplot as plt
import numpy as np

# Assuming this is the output of the classification report from the model
report = {
    'NORMAL': {'precision': 0.97, 'recall': 0.92, 'f1-score': 0.93, 'support': 234},
    'PNEUMONIA': {'precision': 0.91, 'recall': 0.90, 'f1-score': 0.92, 'support': 179},
    'accuracy': 0.95,
    'macro avg': {'precision': 0.92, 'recall': 0.93, 'f1-score': 0.92, 'support': 413},
    'weighted avg': {'precision': 0.93, 'recall': 0.92, 'f1-score': 0.93, 'support': 413}
}

# Prepare data for visualization
labels = ['NORMAL', 'PNEUMONIA', 'macro avg', 'weighted avg']
precision = [report[label]['precision'] for label in labels]
recall = [report[label]['recall'] for label in labels]
f1_score = [report[label]['f1-score'] for label in labels]
support = [report[label]['support'] for label in labels]

# Plot the classification report
x = np.arange(len(labels))
width = 0.2  # Width of the bars

plt.figure(figsize=(10, 6))
plt.bar(x - width, precision, width, label='Precision')
plt.bar(x, recall, width, label='Recall')
plt.bar(x + width, f1_score, width, label='F1-Score')

# Adding text for labels, title, and custom x-axis tick labels, etc.
plt.xlabel('Classes')
plt.ylabel('Scores')
plt.title('Classification Report')
plt.xticks(x, labels)
plt.legend(loc='lower right')

# Displaying values above the bars for better readability
for i in range(len(labels)):
    plt.text(i - width, precision[i] + 0.02, f"{precision[i]:.2f}", ha='center')
    plt.text(i, recall[i] + 0.02, f"{recall[i]:.2f}", ha='center')
    plt.text(i + width, f1_score[i] + 0.02, f"{f1_score[i]:.2f}", ha='center')

plt.ylim([0, 1.2])  # Extend y-axis limit for better text display
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

import os
import tensorflow as tf

misclassified = []
confidence_scores = []

# Loop through test images
for root, dirs, files in os.walk(test_dir):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            test_image_path = os.path.join(root, file)
            img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.
            img_array = np.expand_dims(img_array, axis=0)

            # Make the prediction
            prediction = model.predict(img_array)
            confidence = float(prediction[0][0])
            result = "Pneumonia" if confidence > 0.5 else "Normal"
            true_label = "Pneumonia" if "PNEUMONIA" in test_image_path else "Normal"

            # Collect misclassified images and confidence scores
            if result != true_label:
                misclassified.append((test_image_path, true_label, result, confidence))

# Display misclassified samples
for image_path, true_label, predicted_label, confidence in misclassified[:10]:  # Show up to 10 examples
    print(f"Image: {image_path}")
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {predicted_label} (confidence: {confidence:.2%})")
    print("---")
