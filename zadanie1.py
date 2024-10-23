import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

data = pd.read_csv(
    "/Users/bartlomiejsadza/Documents/Projekty/ProjektyStudia/Machine Learning/Stroke_data.csv",
    sep=";",
    decimal=",",
)
print(data.head())

# Identify categorical columns and apply one-hot encoding
categorical_columns = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]
data = pd.get_dummies(data, columns=categorical_columns)

# Separate numeric and non-numeric columns
numeric_columns = data.select_dtypes(include=[np.number]).columns
non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns

# Fill missing values in numeric columns with the mean
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Fill missing values in non-numeric columns with the mode
for column in non_numeric_columns:
    data[column] = data[column].fillna(data[column].mode()[0])

# Assign features to X and target variable to y
X = data.drop(columns=["stroke"])
y = data["stroke"]

# Ensure X and y have the same number of samples
assert len(X) == len(y), "Mismatch in number of samples between X and y"

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Ensure the split is correct
assert len(X_train) == len(
    y_train
), "Mismatch in number of samples between X_train and y_train"
assert len(X_test) == len(
    y_test
), "Mismatch in number of samples between X_test and y_test"

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Make predictions
y_train_prediction = knn.predict(X_train_scaled)
y_test_prediction = knn.predict(X_test_scaled)

# Print confusion matrices
print(confusion_matrix(y_train, y_train_prediction))
print(confusion_matrix(y_test, y_test_prediction))


weighted_knn = KNeighborsClassifier(n_neighbors=3, weights="distance")
weighted_knn.fit(X_train_scaled, y_train)

y_train_prediction_wknn = weighted_knn.predict(X_train_scaled)
y_test_prediction_wknn = weighted_knn.predict(X_test_scaled)

print("---------------------------------------")
print(confusion_matrix(y_train, y_train_prediction_wknn))
print(confusion_matrix(y_test, y_test_prediction_wknn))
