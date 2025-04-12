import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set up the folder with your images
image_folder = "mnist_samples/"
image_size = (28, 28)  # The size to which you will resize the images

# List all files in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

# Load images and labels
images = []
labels = []

for img_file in image_files:
    # Load the image
    img_path = os.path.join(image_folder, img_file)
    img = load_img(img_path, target_size=image_size, color_mode="grayscale")  # Resize to 28x28 and ensure grayscale
    img_array = img_to_array(img) / 255.0  # Convert to array and normalize (0-1)

    # Extract the label from the filename (assuming format: img_<index>_label_<label>.png)
    label = int(img_file.split('_')[-2])  # Extract label from filename
    images.append(img_array)
    labels.append(label)

# Convert to numpy arrays
x_data = np.array(images)
y_data = np.array(labels)

# Reshape x_data to match the input shape (samples, height, width, channels)
x_data = x_data.reshape(-1, 28, 28, 1)  # Add channel dimension (grayscale)

# One-hot encode labels
y_data = to_categorical(y_data, 10)  # 10 digits (0-9)

# Split into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),  # Input shape is (28, 28, 1) for grayscale images
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {accuracy:.4f}")

# Save the model
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/model.h5")
print("✅ Model saved to 'saved_model/model.h5'")
