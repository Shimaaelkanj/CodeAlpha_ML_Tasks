
# Handwritten Digit Recognition using CNN

## Project Overview
This project focuses on recognizing handwritten digits (0–9) using **Deep Learning** and **Convolutional Neural Networks (CNN)**. The system analyzes images of handwritten digits and predicts the number with high accuracy.

This project is part of the **CodeAlpha Machine Learning Tasks**.

---

## Objective
The main objective is to build a model capable of identifying handwritten digits from images using:

- **CNN (Convolutional Neural Network)** for image classification
- **MNIST dataset** stored as `.npy` files for training and testing

---

## Dataset
The project uses the MNIST dataset, which contains grayscale images of handwritten digits.

- **Training Images:** `train_images.npy` (60,000 images)
- **Training Labels:** `train_labels.npy`
- **Testing Images:** `test_images.npy` (10,000 images)
- **Testing Labels:** `test_labels.npy`


---

## Technologies Used

- Python
- NumPy
- Matplotlib
- TensorFlow / Keras

---

## Preprocessing

The system applies the following preprocessing steps:

1. **Normalization:** Pixel values are scaled between 0 and 1.
2. **Reshaping:** Images are reshaped to `(28,28,1)` for CNN input.
3. **One-hot encoding:** Labels are converted into categorical vectors for classification.

---

## Model Architecture

The CNN model consists of:

1. **Conv2D Layer (32 filters, 3x3, ReLU)**
2. **MaxPooling2D Layer (2x2)**
3. **Conv2D Layer (64 filters, 3x3, ReLU)**
4. **MaxPooling2D Layer (2x2)**
5. **Flatten Layer**
6. **Dense Layer (128 units, ReLU)**
7. **Output Layer (10 units, Softmax)**

The model learns features such as edges, curves, and shapes to identify digits accurately.

---

## Workflow

1. Load training and testing data from `.npy` files
2. Normalize and reshape images
3. Convert labels to categorical format
4. Build and compile CNN model
5. Train the model
6. Evaluate model on test data
7. Predict a sample digit and visualize it

---

## Installation

Install the required libraries:

```bash
pip install numpy matplotlib tensorflow
