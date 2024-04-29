Handwritten Digit Recognition with ANN and CNN:
This project explores building two models for recognizing handwritten digits: an Artificial Neural Network (ANN) and a Convolutional Neural Network (CNN). Both models are trained on the MNIST dataset, a popular benchmark for image classification tasks.

Project Overview
Goal: Develop and compare the performance of ANN and CNN models for handwritten digit recognition.
Dataset: MNIST dataset containing 70,000 images of handwritten digits (0-9).
Models:
Artificial Neural Network (ANN) with a fully-connected architecture.
Convolutional Neural Network (CNN) with convolutional layers for feature extraction.
Getting Started
Prerequisites: Python (3.x recommended), libraries like NumPy, pandas, matplotlib (depending on chosen framework).
MNIST Dataset: Download the MNIST dataset from http://yann.lecun.com/exdb/mnist/.
Choose a Framework: Select a deep learning framework like TensorFlow, PyTorch, or Keras for model implementation.
Run the Scripts: Execute the provided Python scripts for training and evaluating both ANN and CNN models.

Usage:
Modify file paths in data/ to point to your downloaded MNIST data.
Install required libraries using pip install -r requirements.txt.
Train the ANN model: python ann/train.py (or similar command for your framework).
Train the CNN model: python cnn/train.py (or similar command for your framework).
Evaluate the models: python evaluation.py (This script might need adjustments based on your implementation).
Additional Notes

Resources-->
MNIST Dataset: http://yann.lecun.com/exdb/mnist/
TensorFlow Tutorial (Keras): https://www.tensorflow.org/tutorials
PyTorch Tutorials: https://pytorch.org/tutorials/
Keras Documentation: https://keras.io/
