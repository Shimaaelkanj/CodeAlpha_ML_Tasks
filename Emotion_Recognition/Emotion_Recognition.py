

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical



DATASET_PATH = "small_dataset"


emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


# Function to Extract MFCC

def extract_mfcc(file_path):

    audio, sr = librosa.load(file_path, duration=3, offset=0.5)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc



X = []
y = []

for file in os.listdir(DATASET_PATH):

    if file.endswith(".wav"):

        file_path = os.path.join(DATASET_PATH, file)

        mfcc = extract_mfcc(file_path)

        # extract emotion id from filename
        emotion_code = file.split("-")[2]

        emotion = emotion_dict.get(emotion_code)

        if emotion is not None:

            X.append(mfcc)
            y.append(emotion)

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)



label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

y_categorical = to_categorical(y_encoded)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42
)



X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)


# Build CNN Model


model = Sequential()

model.add(
    Conv2D(
        32,
        (3,1),
        activation='relu',
        input_shape=(X_train.shape[1],1,1)
    )
)

model.add(MaxPooling2D(pool_size=(2,1)))

model.add(Dropout(0.3))

model.add(
    Conv2D(
        64,
        (3,1),
        activation='relu'
    )
)

model.add(MaxPooling2D(pool_size=(2,1)))

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(y_categorical.shape[1], activation='softmax'))



model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()



history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=8,
    validation_data=(X_test, y_test)
)



loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)



plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')

plt.legend()

plt.title("Training vs Validation Accuracy")

plt.show()