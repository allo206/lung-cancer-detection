import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ===========================
# Configuration
# ===========================

DATASET_PATH = "lung_subset_small"

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10

# load dataset

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.20
)
#dataset folder teke image pore CNN er traing er prostut kora
train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

#Display class names
print(train_data.class_indices)

# Build CNN Model

model = keras.Sequential([

    layers.Input(shape=(128,128,3)),

    layers.Conv2D(32,3,activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64,3,activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128,3,activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(256,activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(128,activation="relu"),

    layers.Dense(3,activation="softmax")

])

# Display model architecture
model.summary()

# ==========================
# Compile CNN Model
# ==========================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# Define Training Callbacks
# ==========================

class MyCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):

        if logs and logs.get("val_accuracy", 0) > 0.90:
            print("\nValidation Accuracy reached above 90%")
            print("Stopping Training...\n")
            self.model.stop_training = True


early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    patience=2,
    factor=0.5
)

callback = MyCallback()

# ==========================
# Train CNN Model
# ==========================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[
        early_stop,
        reduce_lr,
        callback
    ]
)



# ==========================
# Save Trained Model
# ==========================

model.save("model.keras")

print("\nModel Saved Successfully")

print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

# ==========================
# Plot Accuracy Graph
# ==========================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.savefig("accuracy.png")
plt.show()

# ==========================
# Plot Loss Graph
# ==========================

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.savefig("loss.png") 
plt.show()

# ==========================
# Model Evaluation
# ==========================

from sklearn.metrics import classification_report

predictions = model.predict(val_data)

predicted_classes = np.argmax(predictions, axis=1)

true_classes = val_data.classes

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=list(train_data.class_indices.keys())
    )
)

