


import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical



x_train = np.load("mnist/train_images.npy")
y_train = np.load("mnist/train_labels.npy")

x_test = np.load("mnist/test_images.npy")
y_test = np.load("mnist/test_labels.npy")

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)



x_train = x_train / 255.0
x_test = x_test / 255.0



x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)



y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)



model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax'))



model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()



history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(x_test, y_test)
)



loss, accuracy = model.evaluate(x_test, y_test)

print("Test Accuracy:", accuracy)



predictions = model.predict(x_test)

digit = np.argmax(predictions[0])

plt.imshow(x_test[0].reshape(28,28), cmap='gray')
plt.title(f"Predicted Digit: {digit}")
plt.show()