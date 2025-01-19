# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split

# Load the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

# Load the data into a DataFrame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)

# Add the 'label' column to the DataFrame
data_frame['label'] = breast_cancer_dataset.target

# Splitting the data into features (X) and target (Y)
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Import TensorFlow and Keras
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras

# Define the Neural Network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(X_train.shape[1],)),  # Ensure input_shape matches feature count
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='softmax')  # Use 'softmax' for multi-class classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the Neural Network
history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)

# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test_std, Y_test)
print(f"Test Accuracy: {accuracy}")

# Predict on test data
Y_pred = model.predict(X_test_std)
Y_pred_labels = [np.argmax(i) for i in Y_pred]

# Example input data for prediction
input_data = (11.76, 21.6, 74.72, 427.9, 0.08637, 0.04966, 0.01657, 0.01115, 0.1495, 0.05888, 
              0.4062, 1.21, 2.635, 28.47, 0.005857, 0.009758, 0.01168, 0.007445, 0.02406, 0.001769, 
              12.98, 25.72, 82.98, 516.5, 0.1085, 0.08615, 0.05523, 0.03715, 0.2433, 0.06563)

# Change input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array for a single prediction
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data
input_data_std = scaler.transform(input_data_reshaped)

# Make a prediction
prediction = model.predict(input_data_std)
prediction_label = np.argmax(prediction)

# Output the prediction
if prediction_label == 0:
    print('The tumor is Malignant')
else:
    print('The tumor is Benign')
