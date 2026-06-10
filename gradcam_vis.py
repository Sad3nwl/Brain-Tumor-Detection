import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# Load Saved Model
# =========================

model = tf.keras.models.load_model(
    "brain_tumor_model.keras"
)

# Force model build
dummy_input = tf.random.normal(
    (1, 128, 128, 3)
)

_ = model(dummy_input)

# =========================
# Image Path
# =========================

img_path = "archive/Brain_Tumor_Datasets/test/yes/Te-me_0023.jpg"

IMG_SIZE = (128,128)

# =========================
# Load Image
# =========================

img = image.load_img(
    img_path,
    target_size=IMG_SIZE
)

img_array = image.img_to_array(img)

img_array = np.expand_dims(
    img_array,
    axis=0
)

img_array = preprocess_input(
    img_array
)

# =========================
# Get MobileNetV2 Base Model
# =========================

base_model = model.layers[0]

# Last Conv Layer
last_conv_layer = base_model.get_layer(
    "Conv_1"
)

# =========================
# Create Intermediate Model
# =========================

last_conv_model = tf.keras.Model(
    inputs=base_model.input,
    outputs=last_conv_layer.output
)

# =========================
# Compute Gradients
# =========================

with tf.GradientTape() as tape:

    conv_output = last_conv_model(
        img_array
    )

    tape.watch(
        conv_output
    )

    x = conv_output

    for layer in model.layers[1:]:

        x = layer(x)

    prediction = x

grads = tape.gradient(
    prediction,
    conv_output
)

# Safety Check
if grads is None:

    raise ValueError(
        "Gradients are None. Grad-CAM cannot be computed."
    )

pooled_grads = tf.reduce_mean(
    grads,
    axis=(0,1,2)
)

conv_output = conv_output[0]

heatmap = tf.reduce_sum(
    conv_output * pooled_grads,
    axis=-1
)

heatmap = np.maximum(
    heatmap,
    0
)

heatmap = heatmap / (
    np.max(heatmap) + 1e-8
)

# =========================
# Show Heatmap
# =========================
# Resize heatmap

heatmap = cv2.resize(
    heatmap,
    (128,128)
)

# Original image

original_img = cv2.imread(img_path)

original_img = cv2.resize(
    original_img,
    (128,128)
)

original_img = cv2.cvtColor(
    original_img,
    cv2.COLOR_BGR2RGB
)

# Convert heatmap to color

heatmap_uint8 = np.uint8(
    255 * heatmap
)

heatmap_color = cv2.applyColorMap(
    heatmap_uint8,
    cv2.COLORMAP_JET
)

heatmap_color = cv2.cvtColor(
    heatmap_color,
    cv2.COLOR_BGR2RGB
)

# Overlay

superimposed_img = cv2.addWeighted(
    original_img,
    0.6,
    heatmap_color,
    0.4,
    0
)

# Show result

plt.figure(figsize=(6,6))

plt.imshow(
    superimposed_img
)

plt.axis("off")

plt.title(
    "Grad-CAM Overlay"
)

plt.show()