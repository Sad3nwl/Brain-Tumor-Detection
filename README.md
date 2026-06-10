# Brain Tumor Detection & Explainable AI using MobileNetV2

This repository contains an end-to-end Deep Learning pipeline designed to detect brain tumors from MRI images. The project leverages **Transfer Learning** with **MobileNetV2** for highly accurate binary classification and integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide visual explanations of the model's predictions.

---

## 📌 Project Overview
* **Objective:** Binary classification of MRI scans into `Tumor` and `No Tumor`.
* **Architecture:** Pre-trained **MobileNetV2** base (fine-tuned the last 30 layers) followed by Global Average Pooling, a Dense layer with L2 regularization, Dropout, and a Sigmoid output layer.
* **Explainability (XAI):** Implemented a custom TensorFlow Grad-CAM pipeline to generate visual heatmaps overlaying the MRI images, highlighting the exact areas influencing the network's classification.
* **Pre-processing:** Includes image brightness distribution analysis and real-time image data augmentation (rotation, zoom, horizontal flip, and brightness adjustment).

---

## 📂 Dataset
The dataset used in this project consists of **8,764 images** and is available on **Kaggle**. 
You can download it directly from here: [Brain Tumor Detection Dataset on Kaggle](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection).

To run the pipeline, extract the dataset into your project root directory under the following structure:
```text
archive/
└── Brain_Tumor_Datasets/
    ├── train/
    └── test/
---

## 👥 Project Team
Development and implementation done by:
* **Sadeen Abdelalrahman** - [@sad3nwl](https://github.com/sad3nwl)
* **Raneem Abuhawash** - [@raneem2205](https://github.com/raneem2205)
* **Aseel Mannoun** - [GitHub Profile](https://github.com/ASEEL)
