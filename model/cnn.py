# Import TensorFlow and numpy, core libraries for building and training the model
import tensorflow as tf
import numpy as np

# Import Model and essential layers for building and configuring the model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Check if GPUs are available for faster training
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Download ResNet50, a pre-trained convolutional neural network on ImageNet (widely used for image classification)
from tensorflow.keras.applications import ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Import additional layers to enhance the model's ability to generalize and prevent overfitting
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Activation

# Begin adding custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Reduces the 7x7x2048 output from ResNet50 to a 2048-dimensional vector

# Add Dense layer with L2 regularization to prevent overfitting
x = Dense(256, kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)  # Normalize activations to speed up training and improve stability
x = Activation('relu')(x)  # Apply ReLU activation for non-linearity
x = Dropout(0.3)(x)  # Dropout to prevent overfitting

# Add another Dense layer with the same structure for further learning capacity
x = Dense(256, kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)

# Final classification layer with sigmoid activation for binary classification (pneumonia or no pneumonia)
x = Dense(1, activation='sigmoid')(x)

# Unfreeze the last 15 layers of ResNet50 for fine-tuning on X-ray dataset
for layer in base_model.layers[-15:]:
    layer.trainable = True

# Import backend and optimizer to set up the learning rate schedule
from tensorflow.keras import backend as K
from keras.optimizers import Adam

# Define initial learning rate for gradual training on noisy X-ray data
initial_learning_rate = 1e-5

# Set up exponential decay schedule to gradually reduce learning rate and prevent overfitting
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=10000,                 # Decays learning rate every 10,000 training steps
    decay_rate=0.95,                   # Reduces learning rate by 5% at each step
    staircase=True                     # Applies decay in discrete steps rather than gradually
)

# Create model with specified input and output layers
model = Model(inputs=base_model.input, outputs=x)

# Compile model with binary crossentropy for binary classification and accuracy as a metric
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Import ImageDataGenerator for real-time data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configure data augmentation for training data to improve generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=32,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
    zoom_range=0.2,
    fill_mode='nearest',
    dtype=np.float32
)

# Validation data generator with only rescaling
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    dtype=np.float32,
)

# Test data generator with only rescaling
test_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    dtype=np.float32,
)

from pickle import TRUE
import types

# Load and preprocess training, validation, and testing datasets
train_generator = train_datagen.flow_from_directory(
    '/home/ubuntu/chest_xray/train',
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    '/home/ubuntu/chest_xray/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)
test_generator = test_datagen.flow_from_directory(
    '/home/ubuntu/chest_xray/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights to address any imbalance in the training data
train_labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weights_dict)

# Custom generator wrapper to match expected output format to head off TensorFlow errors
def generator_wrapper(generator):
    while True:
        x, y = next(generator)
        yield (x, y, np.ones(y.shape))

from collections import Counter

# Wrap train and validation generators to ensure expected output format to resolve TensorFlow errors
train_generator_wrapped = generator_wrapper(train_generator)
val_generator_wrapped = generator_wrapper(val_generator)

# Create generator that applies sample weights based on class weights
def generator_wrapper_with_sample_weights(generator, class_weights):
    while True:
        x, y = next(generator)
        sample_weights = np.array([class_weights[int(label)] for label in y])
        yield (x, y, sample_weights)

# Wrap the alread-wrapped train and validation generators to include sample weights
train_generator_wrapped = generator_wrapper_with_sample_weights(train_generator, class_weights_dict)
val_generator_wrapped = generator_wrapper_with_sample_weights(val_generator, class_weights_dict)

# Define callbacks to monitor training progress, checkpoint model, and reduce learning rate if needed
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True),
]

# Calculate validation steps for the generator
val_generator.batch_size = min(val_generator.batch_size, val_generator.samples)
val_steps = max(1, val_generator.samples // val_generator.batch_size)

# Train the model with training and validation data, using sample weights and specified callbacks
history = model.fit(
    train_generator_wrapped,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator_wrapped,
    validation_steps=val_steps,
    epochs=20,
    verbose=1,
    callbacks=callbacks
)

# Save trained model
model.save('pneumonia_model.keras')

from tensorflow.keras.models import load_model
import os

# Test model on individual images from test directory and print predictions
test_dir = '/content/chest_xray/test'
for root, dirs, files in os.walk(test_dir):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            test_image_path = os.path.join(root, file)
            img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction on each test image
            prediction = model.predict(img_array)
            confidence = float(prediction[0][0])
            result = "Pneumonia" if confidence > 0.5 else "Normal"

            # Print the prediction result and confidence level
            print(f"Image: {test_image_path}")
            print(f"Prediction: {result} (confidence: {confidence:.2%})")
            print("---")

from sklearn.metrics import classification_report, confusion_matrix

# Generate true labels and predictions for test data
test_generator.reset()
Y_true = test_generator.classes
Y_pred = (model.predict(test_generator) > 0.5).astype("int32")

# Display confusion matrix for evaluation
conf_matrix = confusion_matrix(Y_true, Y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Print classification report for precision, recall, and F1-score
class_report = classification_report(Y_true, Y_pred, target_names=test_generator.class_indices.keys())
print("Classification Report:")
print(class_report)
