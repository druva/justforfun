import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the weather data
data = pd.read_csv('GlobalWeatherRepository.csv')

# Convert 'Date' column to datetime
data['last_updated'] = pd.to_datetime(data['last_updated'])

# Optional: Sort the data by date
data = data.sort_values('last_updated')

# Display the first few rows of the data
print(data.head())

# Let's extract the 'Day of the Year' feature to capture seasonal patterns
data['DayOfYear'] = data['last_updated'].dt.dayofyear

# Use 'DayOfYear', 'Humidity', and 'Rainfall' as features for the model
X = data[['DayOfYear', 'humidity', 'cloud']]
y = data['temperature_celsius']  # Target variable: Temperature


# Split the data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the size of the training and test sets
print(f'Training set size: {len(X_train)}')
print(f'Test set size: {len(X_test)}')

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the true vs predicted temperatures
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('True Temperature')
plt.ylabel('Predicted Temperature')
plt.title('True vs Predicted Temperature')
plt.show()


input_data = np.array([[150, 75, 25]])  # DayOfYear, Humidity, Rainfall
predicted_temp = model.predict(input_data)
print(f"Predicted Temperature: {predicted_temp[0]} °C")