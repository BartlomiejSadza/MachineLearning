## 1. W jakim celu dokonuje się podziału zbioru danych na zbiór uczący i testowy? W jaki sposób ten podział powinien zostać wykonany?

### Podział zbioru danych na zbiór uczący i testowy

Podział zbioru danych na zbiór uczący i testowy jest kluczowym krokiem w procesie tworzenia modeli uczenia maszynowego. Oto główne cele i sposób wykonania tego podziału:

### Cele podziału zbioru danych:
1. **Ocena modelu**: Podział danych pozwala na ocenę modelu na danych, które nie były używane podczas jego trenowania. Dzięki temu można uzyskać realistyczną ocenę jego wydajności.
2. **Unikanie przeuczenia (overfitting)**: Trenowanie modelu na całym zbiorze danych może prowadzić do przeuczenia, gdzie model dobrze radzi sobie na danych treningowych, ale słabo na nowych, niewidzianych danych. Podział na zbiór uczący i testowy pomaga w identyfikacji tego problemu.
3. **Walidacja modelu**: Podział danych umożliwia walidację modelu, co jest kluczowe dla wyboru najlepszego modelu i jego parametrów.

### Sposób wykonania podziału:
1. **Losowy podział**: Najczęściej stosowaną metodą jest losowy podział danych na zbiór uczący i testowy. Typowy podział to 80% danych na zbiór uczący i 20% na zbiór testowy, ale proporcje mogą się różnić w zależności od wielkości zbioru danych i specyfiki problemu.
2. **Stratyfikacja**: W przypadku danych z klasami (np. klasyfikacja), warto zastosować stratyfikację, aby zapewnić, że proporcje klas w zbiorze uczącym i testowym są takie same jak w całym zbiorze danych.
3. **Funkcje biblioteki**: W Pythonie można użyć funkcji `train_test_split` z biblioteki `scikit-learn`, która automatycznie wykonuje losowy podział danych i może również zastosować stratyfikację.

### Przykład kodu w Pythonie:
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Wczytanie danych
data = pd.read_csv('path_to_your_data.csv')

# Podział na cechy (X) i etykiety (y)
X = data.drop(columns=['target_column'])
y = data['target_column']

# Podział na zbiór uczący i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Sprawdzenie rozmiarów zbiorów
print(f'Rozmiar zbioru uczącego: {X_train.shape[0]}')
print(f'Rozmiar zbioru testowego: {X_test.shape[0]}')
```

### Jaka jest różnica między dokładnością, czułością a specyficznością? Która z tych miar i w jakim przypadku jest ważniejsza. Podać przykłady.

1. **Dokładność (Accuracy)**: Jest to miara, która określa, jaki procent wszystkich przewidywań modelu jest poprawny. Oblicza się ją jako stosunek liczby poprawnych przewidywań do całkowitej liczby przypadków. Dokładność jest użyteczna, gdy klasy są zrównoważone, czyli liczba przypadków każdej klasy jest podobna.
   - Przykład: W klasyfikacji emaili jako spam lub nie-spam, jeśli mamy 1000 emaili, z czego 950 to nie-spam i 50 to spam, model, który zawsze przewiduje "nie-spam", będzie miał dokładność 95%, ale nie będzie użyteczny w wykrywaniu spamu.

2. **Czułość (Sensitivity, Recall)**: Jest to miara, która określa, jaki procent rzeczywistych pozytywnych przypadków został poprawnie zidentyfikowany przez model. Oblicza się ją jako stosunek liczby prawdziwych pozytywnych przewidywań do liczby wszystkich rzeczywistych pozytywnych przypadków. Czułość jest ważna w sytuacjach, gdzie istotne jest wykrycie wszystkich pozytywnych przypadków.
   - Przykład: W diagnostyce medycznej, gdzie chcemy wykryć wszystkie przypadki choroby, wysoka czułość jest kluczowa, aby zminimalizować ryzyko przeoczenia chorego pacjenta.

3. **Specyficzność (Specificity)**: Jest to miara, która określa, jaki procent rzeczywistych negatywnych przypadków został poprawnie zidentyfikowany przez model. Oblicza się ją jako stosunek liczby prawdziwych negatywnych przewidywań do liczby wszystkich rzeczywistych negatywnych przypadków. Specyficzność jest ważna w sytuacjach, gdzie istotne jest unikanie fałszywych alarmów.
   - Przykład: W systemach bezpieczeństwa, gdzie fałszywe alarmy mogą być kosztowne i uciążliwe, wysoka specyficzność jest kluczowa, aby zminimalizować liczbę fałszywych alarmów.

### Przykład kodu w Pythonie:
```python
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Zakładając, że mamy prawdziwe etykiety (y_true) i przewidywane etykiety (y_pred)
y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
y_pred = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]

# Obliczenie dokładności
accuracy = accuracy_score(y_true, y_pred)
print(f'Dokładność: {accuracy}')

# Obliczenie czułości
sensitivity = recall_score(y_true, y_pred)
print(f'Czułość: {sensitivity}')

# Obliczenie specyficzności
specificity = recall_score(y_true, y_pred, pos_label=0)
print(f'Specyficzność: {specificity}')