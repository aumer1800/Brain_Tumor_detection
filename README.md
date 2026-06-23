Brain Tumor Detection using CNN with Grad-CAM Heatmap

Overview

This project is a Deep Learning-based Brain Tumor Detection System using a Convolutional Neural Network (CNN) trained on MRI images.

To improve interpretability, the system uses Grad-CAM heatmaps to visually highlight tumor-affected regions in the brain.

The model classifies MRI scans into:

Tumor Detected

No Tumor

Key Features

CNN-based medical image classification

High-accuracy prediction on MRI scans

Grad-CAM heatmap visualization (Explainable AI)

Simple web interface (Frontend)

Fast real-time prediction

Modular and clean project structure

Tech Stack

Python

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

Flask

HTML, CSS, JavaScript

Model Architecture

The CNN model consists of:

Convolution Layers → Feature extraction

MaxPooling Layers → Dimensionality reduction

Dropout Layers → Overfitting prevention

Dense Layers → Final classification

Grad-CAM Heatmap

Grad-CAM is used to generate visual explanations of model predictions:

🔴 Red / Yellow → High attention (possible tumor region)
🔵 Blue → Low attention

This improves trust and interpretability of AI predictions.
