import os
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load dataset
(x_train, y_train), (_, _) = mnist.load_data()

# Create a folder to save images
os.makedirs("mnist_samples", exist_ok=True)

# Save first 100 images
for i in range(100):
    plt.imsave(f"mnist_samples/img_{i}_label_{y_train[i]}.png", x_train[i], cmap="gray")
# for i in range(100, 1100):
#     plt.imsave(f"mnist_samples/img_{i}_label_{y_train[i]}.png", x_train[i], cmap="gray")
# for i in range(1100, 1500):
#     plt.imsave(f"mnist_samples/img_{i}_label_{y_train[i]}.png", x_train[i], cmap="gray")
# for i in range(20000, 40000):
#     plt.imsave(f"mnist_samples/img_{i}_label_{y_train[i]}.png", x_train[i], cmap="gray")
