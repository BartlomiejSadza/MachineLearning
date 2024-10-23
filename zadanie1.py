import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

data = pd.read_csv(
    "/Users/bartlomiejsadza/Documents/Projekty/ProjektyStudia/Machine Learning/Stroke_data.csv",
    sep=";",
    decimal=","
)
print(data.head())

X = data.iloc[:, 0:8]
y = data.diabetes

# no zbiór uczący no i zbiór testowy 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# no standaryzacja 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)