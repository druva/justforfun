import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Load and prepare data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {accuracy:.4f}")

# Predict a random image
index = np.random.randint(0, len(x_test))
image = x_test[index]

plt.imshow(image, cmap='gray')
plt.title("Actual Digit")
plt.show()

prediction = model.predict(image.reshape(1, 28, 28))
print("Predicted digit:", prediction.argmax())
