import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

data = pd.read_csv("/content/sample_data/PimaIndiansDiabetes.csv", sep=";", decimal=",")
print(data.head())

X = data.iloc[:, 0:8]
y = data.diabetes
# Podział na zbiór uczący i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Standaryzacja
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Metoda K njabliższych sąsiadów
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

y_train_pred_knn = knn.predict(X_train_scaled)  # Predykcja na zbiorze uczącym
y_test_pred_knn = knn.predict(X_test_scaled)  # Predykcja na zbiorze testowym

print(confusion_matrix(y_train_pred_knn, y_train))  # Macierz błędów - zbiór uczący
print(confusion_matrix(y_test_pred_knn, y_test))  # Macierz błędów - zbiór testowy


# Metoda ważonych odległości najbliższych sąsiadów
weighted_knn = KNeighborsClassifier(n_neighbors=3, weights="distance")
weighted_knn.fit(X_train_scaled, y_train)

y_train_pred_weighted_knn = weighted_knn.predict(
    X_train_scaled
)  # Predykcja na zbiorze uczącym
y_test_pred_weighted_knn = weighted_knn.predict(
    X_test_scaled
)  # Predykcja na zbiorze testowym

print(
    confusion_matrix(y_train_pred_weighted_knn, y_train)
)  # Macierz błędów - zbiór uczący
print(
    confusion_matrix(y_test_pred_weighted_knn, y_test)
)  # Macierz błędów - zbiór testowy
