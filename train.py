# train.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# =========================
# CONFIG
# =========================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
DATASET_PATH = "../dataset_final"
MODEL_SAVE_PATH = "../model/best_model.h5"

# =========================
# DATA AUGMENTATION
# =========================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,

    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

NUM_CLASSES = len(train_data.class_indices)

print("Classes:", train_data.class_indices)

# =========================
# CNN MODEL
# =========================

def build_model():
    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2,2))

    # Block 2
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2,2))

    # Block 3
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2,2))

    # Block 4
    model.add(layers.Conv2D(256, (3,3), activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.GlobalAveragePooling2D())

    # Fully Connected
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

model = build_model()
model.summary()

# =========================
# CALLBACKS
# =========================

checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

lr_reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    verbose=1
)

# =========================
# TRAINING
# =========================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop, lr_reduce]
)

# =========================
# SAVE FINAL MODEL
# =========================

model.save("../model/final_model.h5")

print(" Training Completed Successfully!")
print(" Best model saved at:", MODEL_SAVE_PATH)