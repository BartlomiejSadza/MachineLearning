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