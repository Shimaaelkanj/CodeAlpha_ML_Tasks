
# Emotion Recognition from Speech (CNN + MFCC)

## Project Overview

This project focuses on recognizing human emotions from speech audio using **Deep Learning** and **Speech Signal Processing** techniques. The system analyzes audio signals, extracts important acoustic features, and classifies the emotional state of the speaker.

The goal is to automatically detect emotions such as **happy, sad, angry, and neutral** from speech recordings.

This project is part of the **CodeAlpha Machine Learning Internship Tasks**.

---

## Objective

The main objective of this project is to build a model capable of identifying emotions from speech audio using:

* **MFCC (Mel-Frequency Cepstral Coefficients)** for feature extraction
* **Convolutional Neural Network (CNN)** for emotion classification

---

## Dataset

A small dataset named **small_dataset** is used for this project.
It contains approximately **20 audio files (.wav)** extracted from emotional speech datasets.

Example datasets commonly used for emotion recognition include:

* RAVDESS
* TESS
* EMO-DB

Each audio file represents a speech recording associated with a specific emotion.

### Dataset Structure

Emotion_Project
│
├── emotion_recognition_cnn.py
├── README.md
└── small_dataset
  ├── 03-01-03-01-01-01-01.wav
  ├── 03-01-04-01-01-01-02.wav
  ├── 03-01-05-01-01-01-03.wav
  └── ...

---

## Technologies Used

* Python
* Librosa (Audio Processing)
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Matplotlib

---

## Feature Extraction

The system extracts **MFCC (Mel-Frequency Cepstral Coefficients)** from each audio signal.

MFCCs are widely used in speech processing because they capture the **spectral characteristics of human speech**, making them effective for emotion detection.

Steps:

1. Load audio file
2. Convert speech signal to MFCC features
3. Average MFCC values
4. Use extracted features as model input

---

## Model Architecture

A **Convolutional Neural Network (CNN)** is used to classify emotions from MFCC features.

Main layers used in the model:

* Convolutional Layer (Conv2D)
* Max Pooling Layer
* Dropout Layer
* Fully Connected Layer (Dense)
* Output Softmax Layer

The model learns patterns from MFCC features to distinguish between emotional states.

---

## Workflow

The project follows these steps:

1. Load audio dataset
2. Extract MFCC features using Librosa
3. Encode emotion labels
4. Split data into training and testing sets
5. Train a CNN model
6. Evaluate model accuracy
7. Visualize training results

---

## Installation

Install the required libraries before running the project.

```bash
pip install numpy librosa matplotlib scikit-learn tensorflow
```

---

## How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Emotion_Recognition_Project.git
```

2. Navigate to the project folder:

```bash
cd Emotion_Recognition_Project
```

3. Run the Python script:

```bash
python emotion_recognition_cnn.py
```

---

## Output

The model will:

* Train on the speech dataset
* Evaluate prediction accuracy
* Display training vs validation accuracy graph

Example output:

Test Accuracy: 0.75

(Note: Accuracy may vary depending on dataset size.)

---

## Applications

Emotion recognition from speech can be used in many real-world applications such as:

* Human-computer interaction
* Customer service analysis
* Mental health monitoring
* Virtual assistants
* Smart call centers

---

## Future Improvements

Possible improvements for this project include:

* Using larger datasets
* Applying **LSTM or RNN models**
* Adding more audio features (Chroma, Spectral Contrast)
* Real-time emotion detection
* Audio data augmentation

---

## Conclusion

This project demonstrates how **Deep Learning and Speech Signal Processing** can be combined to recognize emotions from speech audio. Using **MFCC feature extraction** and a **CNN classifier**, the system learns patterns in human speech and predicts emotional states with reasonable accuracy.

---

## Author

Shaimaa Kanj
Machine Learning Internship Project – CodeAlpha
