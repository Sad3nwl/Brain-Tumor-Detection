import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras import regularizers
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# =========================
# Dataset Paths
# =========================

train_path ="archive/Brain_Tumor_Datasets/train"

test_path ="archive/Brain_Tumor_Datasets/test"

# =========================
# Parameters
# =========================

IMG_SIZE = (128,128)
BATCH_SIZE = 32
EPOCHS = 20

# =========================
# Count Images
# =========================

yes_count = len(
    os.listdir(
        os.path.join(train_path, "yes")
    )
)

no_count = len(
    os.listdir(
        os.path.join(train_path, "no")
    )
)

print("Training YES images:", yes_count)
print("Training NO images:", no_count)

# =========================
# Brightness Analysis
# =========================

image_paths = []

for folder in [

    os.path.join(train_path, "yes"),
    os.path.join(train_path, "no")

]:

    for img_name in os.listdir(folder):

        image_paths.append(
            os.path.join(folder, img_name)
        )

brightness_values = []

for path in image_paths:

    img = cv2.imread(path)

    if img is None:
        continue

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    brightness_values.append(
        np.mean(gray)
    )

plt.figure(figsize=(8,6))

plt.hist(brightness_values)

plt.xlabel("Average Brightness")
plt.ylabel("Number of Images")
plt.title("Brightness Distribution")

plt.show()

# =========================
# Data Generators
# =========================

train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    validation_split=0.2,

    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8,1.2]

)

test_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input

)

# =========================
# Training Data
# =========================

train_data = train_datagen.flow_from_directory(

    train_path,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode='binary',

    subset='training'

)

# =========================
# Validation Data
# =========================

val_data = train_datagen.flow_from_directory(

    train_path,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode='binary',

    subset='validation',

    shuffle=False

)

# =========================
# Test Data
# =========================

test_data = test_datagen.flow_from_directory(

    test_path,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode='binary',

    shuffle=False

)

# =========================
# MobileNetV2
# =========================

base_model = tf.keras.applications.MobileNetV2(

    input_shape=(128,128,3),

    include_top=False,

    weights='imagenet'

)

base_model.trainable = True

for layer in base_model.layers[:-30]:

    layer.trainable = False

# =========================
# Build Model
# =========================

model = models.Sequential([

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(

        128,

        activation='relu',

        kernel_regularizer=
        regularizers.l2(0.001)

    ),

    layers.Dropout(0.3),

    layers.Dense(

        1,

        activation='sigmoid'

    )

])

# =========================
# Compile
# =========================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=0.0001

    ),

    loss='binary_crossentropy',

    metrics=['accuracy']

)

# =========================
# Callbacks
# =========================

early_stop = EarlyStopping(

    monitor='val_loss',

    patience=3,

    restore_best_weights=True

)

reduce_lr = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.5,

    patience=2,

    min_lr=0.000001,

    verbose=1

)

checkpoint = ModelCheckpoint(

    "best_brain_tumor_model.keras",

    monitor='val_accuracy',

    save_best_only=True,

    mode='max',

    verbose=1

)

# =========================
# Train Model
# =========================

history = model.fit(

    train_data,

    validation_data=val_data,

    epochs=EPOCHS,

    callbacks=[

        early_stop,

        checkpoint,

        reduce_lr

    ]

)

# =========================
# Accuracy Plot
# =========================

plt.figure(figsize=(8,6))

plt.plot(

    history.history['accuracy'],

    label='Training Accuracy'

)

plt.plot(

    history.history['val_accuracy'],

    label='Validation Accuracy'

)

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.title('Training and Validation Accuracy')

plt.legend()

plt.show()

# =========================
# Loss Plot
# =========================

plt.figure(figsize=(8,6))

plt.plot(

    history.history['loss'],

    label='Training Loss'

)

plt.plot(

    history.history['val_loss'],

    label='Validation Loss'

)

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.title('Training and Validation Loss')

plt.legend()

plt.show()

# =========================
# Test Evaluation
# =========================

loss, accuracy = model.evaluate(

    test_data

)

print(

    "\nTest Accuracy:",

    accuracy

)

# =========================
# Predictions
# =========================

test_data.reset()

predictions = model.predict(

    test_data

)

predicted_classes = (

    predictions > 0.5

).astype(int)

true_classes = test_data.classes

# =========================
# Classification Report
# =========================

print(

    "\nClassification Report:\n"

)

print(

    classification_report(

        true_classes,

        predicted_classes

    )

)

# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(

    true_classes,

    predicted_classes

)

print(

    "\nConfusion Matrix:\n"

)

print(cm)

# =========================
# Plot Confusion Matrix
# =========================

plt.figure(figsize=(6,6))

plt.imshow(cm, cmap='Blues')

plt.colorbar()

plt.xticks(

    [0,1],

    ['No Tumor','Tumor']

)

plt.yticks(

    [0,1],

    ['No Tumor','Tumor']

)

plt.xlabel('Predicted Label')

plt.ylabel('True Label')

plt.title('Confusion Matrix')

for i in range(2):

    for j in range(2):

        plt.text(j,i,cm[i,j],ha='center',va='center')

plt.savefig(

    "confusion_matrix.png",

    dpi=300,

    bbox_inches='tight'

)

plt.show()

# =========================
# Sample Predictions
# =========================

images, labels = next(test_data)

predictions = model.predict(images)

plt.figure(figsize=(12,8))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.imshow(images[i])

    plt.axis('off')

    predicted_label = (

        "Tumor"

        if predictions[i] > 0.5

        else "No Tumor"

    )

    true_label = (

        "Tumor"

        if labels[i] == 1

        else "No Tumor"

    )

    plt.title(

        f"Pred: {predicted_label}\nTrue: {true_label}"

    )

plt.tight_layout()

plt.show()

# =========================
# Save Model
# =========================

model.save(

    "brain_tumor_model.keras"

)

print(

    "\nModel saved successfully!"

)

