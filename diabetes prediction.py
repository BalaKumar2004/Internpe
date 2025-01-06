import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Loading the diabetes dataset into a pandas DataFrame
diabetes_dataset = pd.read_csv(r"C:\Users\balak\Downloads\internpe internship\WEEK 1\diabetes - diabetes.csv")

# Printing the first 5 rows of the dataset
print(diabetes_dataset.head())

# Number of rows and columns in this dataset
print("Shape of the dataset:", diabetes_dataset.shape)

# Statistical measures of the data
print(diabetes_dataset.describe())

# Distribution of the target column
print(diabetes_dataset['Outcome'].value_counts())

# Mean values grouped by the outcome
print(diabetes_dataset.groupby('Outcome').mean())

# Separating the data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardizing the data
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

X = standardized_data
print("Standardized Data:\n", X)

# Splitting the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print("Shapes:", X.shape, X_train.shape, X_test.shape)

# Training the SVM Classifier with a linear kernel
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of the training data:', training_data_accuracy)

# Accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of the test data:', test_data_accuracy)

# Predicting for new input data
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshaping the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardizing the input data
std_data = scaler.transform(input_data_reshaped)
print("Standardized Input Data:", std_data)

# Making a prediction
prediction = classifier.predict(std_data)
if prediction[0] == 0:
    print('The person is not diabetic.')
else:
    print('The person is diabetic.')
