# Zadanie 3. Metoda KKNN

## Co odróżnia metodę KKNN od metody KNN?
Metoda KKNN (Kernel K-Nearest Neighbors) różni się od klasycznej metody KNN (K-Nearest Neighbors) tym, że wprowadza wagowanie sąsiadów za pomocą funkcji jądra (kernel). W klasycznym KNN każdy z k najbliższych sąsiadów ma równy wpływ na decyzję, podczas gdy w KKNN sąsiedzi bliżsi mają większy wpływ niż dalsi.

## Czy w metodzie tej występują ograniczenia dotyczące wyboru wartości k?
Tak, podobnie jak w klasycznym KNN, wybór wartości k w KKNN jest istotny. Zbyt mała wartość k może prowadzić do nadmiernego dopasowania (overfitting), podczas gdy zbyt duża wartość k może prowadzić do niedopasowania (underfitting). Wartość k powinna być dobrana w taki sposób, aby zbalansować te dwa zjawiska.

## Czy wielkość zbioru danych ma znaczenie w przypadku tej metody?
Tak, wielkość zbioru danych ma znaczenie w przypadku metody KKNN. Większe zbiory danych mogą prowadzić do bardziej dokładnych wyników, ponieważ więcej danych dostarcza więcej informacji do analizy. Jednakże, większe zbiory danych mogą również zwiększyć czas obliczeń. 

## Sprawdzić, czy liczba obserwacji wpływa na uzyskiwane wyniki.
Aby sprawdzić wpływ liczby obserwacji na uzyskiwane wyniki, można przeprowadzić eksperymenty z różnymi rozmiarami zbiorów danych i porównać wyniki. Można to zrobić poprzez podział danych na różne podzbiory i analizę wyników dla każdego z nich.

### Podział treningowy:
|True |False|
|-----|-----|
| 4377|  0  |
|  0  | 221 |

### Podział testowy:
|     |     |
|-----|-----|
| 468 |  15 |
| 26  |  2  |

