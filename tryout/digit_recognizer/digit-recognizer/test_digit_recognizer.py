from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load the saved model
model = load_model("saved_model/model.h5")

# Load and preprocess a new image (from mnist_samples/ or elsewhere)
img_path = "mnist_samples/img_285_label_9.png"  # Path to a test image
img = load_img(img_path, target_size=(28, 28), color_mode="grayscale")
img_array = img_to_array(img) / 255.0
img_input = img_array.reshape(1, 28, 28, 1)  # Reshape to (1, 28, 28, 1)

# Predict using the model
prediction = model.predict(img_input)
predicted_digit = prediction.argmax()

# Show the image and the predicted digit
plt.imshow(img_array, cmap="gray")
plt.title(f"Predicted: {predicted_digit}")
plt.axis('off')
plt.show()

print(f"Predicted digit: {predicted_digit}")
