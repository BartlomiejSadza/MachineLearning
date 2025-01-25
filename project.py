import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC


pd.set_option('display.max_columns', None)
data = pd.read_csv("Hotel Reservations.csv")
data.head()
df = data.copy()
df.drop(columns=["Booking_ID", "arrival_date"], axis=1, inplace=True)

print(df.isnull().sum())
print(df.shape)

df["type_of_meal_plan"] = df["type_of_meal_plan"].astype('category')
df["required_car_parking_space"] = df["required_car_parking_space"].astype('category')
df["room_type_reserved"] = df["room_type_reserved"].astype('category')
df["market_segment_type"] = df["market_segment_type"].astype('category')
df["repeated_guest"] = df["repeated_guest"].astype('category')
df["booking_status"] = df["booking_status"].astype('category')

print(df.dtypes)
print(df.nunique())
print(df.describe().T)

catcols = df.select_dtypes(include='category').columns

plt.figure(figsize=(18, len(catcols) // 3 * 6))
rows = 2

for i, column in enumerate(catcols, 1):
    plt.subplot(rows, 3, i)
    ax = sns.countplot(data=df, x=column, palette="viridis")
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height / len(df) * 100:.1f}%',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='baseline', fontsize=10, color='black')
        ax.tick_params(axis='x', rotation=90)

    plt.title(f'Procentowy Countplot dla kolumny: {column}')
    plt.ylabel('Liczność')
    plt.xlabel(column)

plt.tight_layout()
plt.show()

df.drop(["required_car_parking_space", "repeated_guest"],axis=1, inplace=True)
plt.figure(figsize=(40, 40))
axes = df.select_dtypes(include=[int, float]).hist(layout=(4,3), figsize=(40, 40))
for ax in axes.flat:
    ax.set_title(ax.get_title(), fontsize=30)
plt.suptitle("Histogramy dla pozostałych zmiennych", fontsize=50, y=0.92)
plt.show()

df.drop(columns=["no_of_previous_cancellations", "no_of_previous_bookings_not_canceled"], axis=1, inplace=True)
df['no_of_children'].value_counts()/len(df)*100

df.drop(columns=["no_of_children"], axis=1, inplace=True)
sns.countplot(data=df, x='booking_status')

data = []

for col in df.select_dtypes(include=[int, float]).columns:
    mean = df[col].mean()
    std = df[col].std()
    min_val = df[col].min()
    max_val = df[col].max()

    data.append({
        "Column": col,
        "Mean + 3*Std": mean + 3 * std,
        "Mean - 3*Std": mean - 3 * std,
        "Min": min_val,
        "Max": max_val
    })

summary_df = pd.DataFrame(data)
print(summary_df)

df.plot(kind="box", subplots=True, figsize=(15,15), layout=(5,5))

plt.show()
for col in df.select_dtypes(include=[int, float]).columns:
    mean = df[col].mean()
    std = df[col].std()
    df = df[(df[col]<=mean+3*std) & (df[col]>=mean-3*std)]

print(f"pozostało rekordów: {df.shape[0]}")

object_columns = df.drop(["lead_time", "avg_price_per_room", "booking_status"], axis=1)
fig, ax = plt.subplots(3,3, figsize=(20, 20))
ax = ax.flatten()

# Tworzenie wykresów
for i, column in enumerate(object_columns):
     df_grouped = df.groupby([column, 'booking_status'], observed=False).size().unstack().fillna(0)
     df_grouped = df_grouped.div(df_grouped.sum(axis=1), axis=0)  # Normalizacja do procentów
     df_grouped.plot(kind='bar', stacked=True, ax=ax[i], width=0.8)  # Rysowanie na odpowiedniej osi
     ax[i].set_title(f'Barplot dla {column}')
     ax[i].set_xlabel(column)
     ax[i].set_ylabel('Procent')
     ax[i].legend(title='Booking Status')


plt.tight_layout()
plt.show()

numeric_columns = df.loc[:, ["lead_time", "avg_price_per_room"]].columns
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax = ax.flatten()

# Tworzenie wykresów
for i, column in enumerate(numeric_columns):
    sns.kdeplot(data=df, x=column, hue='booking_status', fill=True, ax=ax[i])
    ax[i].set_title(f'Wykres gęstości dla {column}')

plt.tight_layout()
plt.show()

from scipy.stats import chi2_contingency

df_pvals = pd.DataFrame(columns=["Variable", "P-value"])
for column in object_columns:
    data_crosstab = np.array(pd.crosstab(df['booking_status'],
                                         df[column]))

    res = chi2_contingency(data_crosstab)
    df_pvals.loc[len(df_pvals)] = [column, res.pvalue]

for column in ["lead_time", "avg_price_per_room"]:
    # Dane wejściowe
    x1 = df[df["booking_status"] == "Canceled"][column]
    x2 = df[df["booking_status"] == "Not_Canceled"][column]
    bins = np.linspace(0, max(x1.max(), x2.max()), 15)  # Przedziały
    hist1, _ = np.histogram(x1, bins=bins)
    hist2, _ = np.histogram(x2, bins=bins)
    data_crosstab = np.array([hist1, hist2])
    res = chi2_contingency(data_crosstab)
    df_pvals.loc[len(df_pvals)] = [column, res.pvalue]

print(df_pvals)


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        "sensitivity": sensitivity,
        "precision": precision,
        "specificity": specificity,
        "accuracy": accuracy
    }


def aggregate_metrics(metrics_list):
    metrics_df = pd.DataFrame(metrics_list)
    averaged_metrics = metrics_df.mean().to_dict()
    return pd.DataFrame([averaged_metrics])


def print_conf_matrix(y, y_pred):
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

df=pd.get_dummies(df, columns=["type_of_meal_plan", "room_type_reserved", "market_segment_type"], drop_first=True, dtype=int)
df.booking_status = df.booking_status.map({"Canceled" : 1, "Not_Canceled" : 0})
df.head()

X = df.drop('booking_status', axis=1)
X = X.values
y = df['booking_status']

scaler = StandardScaler()
X_standard = scaler.fit_transform(X)

naive_bayes_und = MultinomialNB()
naive_bayes_ov = MultinomialNB()

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

accuracies_train_und = []
accuracies_test_und = []
y_preds_und = []
y_true_und = []

accuracies_train_ov = []
accuracies_test_ov = []
y_preds_ov = []
y_true_ov = []

for train_idx, test_idx in cv.split(X, y.values):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    undersampler = RandomUnderSampler(random_state=1)
    X_train_resampled_und, y_train_resampled_und = undersampler.fit_resample(X_train, y_train)

    smote = SMOTE(random_state=2)
    X_train_resampled_ov, y_train_resampled_ov = smote.fit_resample(X_train, y_train)

    # Trenowanie na undersamplingu
    naive_bayes_und.fit(X_train_resampled_und, y_train_resampled_und)

    # Trenowanie na oversamplingu
    naive_bayes_ov.fit(X_train_resampled_ov, y_train_resampled_ov)

    # Ocena na oryginalnych danych testowych
    y_pred_test_und = naive_bayes_und.predict(X_test)
    y_pred_train_und = naive_bayes_und.predict(X_train)

    y_pred_test_ov = naive_bayes_ov.predict(X_test)
    y_pred_train_ov = naive_bayes_ov.predict(X_train)

    accuracies_train_und.append(calculate_metrics(y_train, y_pred_train_und))
    accuracies_test_und.append(calculate_metrics(y_test, y_pred_test_und))

    accuracies_train_ov.append(calculate_metrics(y_train, y_pred_train_ov))
    accuracies_test_ov.append(calculate_metrics(y_test, y_pred_test_ov))

    y_preds_und.extend(y_pred_test_und)
    y_true_und.extend(y_test)
    y_preds_ov.extend(y_pred_test_ov)
    y_true_ov.extend(y_test)

result_und = pd.concat([aggregate_metrics(accuracies_train_und), aggregate_metrics(accuracies_test_und)])
result_und.index = ["train", "test"]

result_ov = pd.concat([aggregate_metrics(accuracies_train_ov), aggregate_metrics(accuracies_test_ov)])
result_ov.index = ["train", "test"]

print("Wyniki dla undersamplingu")
print(result_und)
print_conf_matrix(y_true_und, y_preds_und)

print(
    "-----------------------------------------------------------------------------------------------------------------------------------------")

print("Wyniki dla oversamplingu")
print(result_ov)
print_conf_matrix(y_true_ov, y_preds_ov)


def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def optimize_parameters(model, X, y, param_grid):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

    best_accuracy = 0
    best_params = {}
    undersampling = [True, False]
    best_resampling = True

    # Iteracja przez wszystkie kombinacje parametrów
    for undersampling in undersampling:
        for min_samples_split in param_grid['min_samples_split']:
            for max_depth in param_grid['max_depth']:
                for criterion in param_grid['criterion']:
                    current_params = {
                        'min_samples_split': min_samples_split,
                        'max_depth': max_depth,
                        'criterion': criterion
                    }

                    fold_accuracies = []

                    for train_idx, test_idx in cv.split(X, y):
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                        if undersampling == True:
                            undersampler = RandomUnderSampler(random_state=1)
                            X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
                        else:
                            smote = SMOTE(random_state=2)
                            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

                        # Trenowanie modelu
                        model.set_params(**current_params)
                        model.fit(X_train_resampled, y_train_resampled)

                        # Ewaluacja na zbiorze testowym
                        y_pred_test = model.predict(X_test)
                        accuracy = np.mean(y_test == y_pred_test)
                        fold_accuracies.append(accuracy)

                    # Średnia dokładność dla bieżącej kombinacji parametrów
                    mean_accuracy = np.mean(fold_accuracies)

                    # Sprawdzenie czy model jest najlepszy
                    if mean_accuracy > best_accuracy:
                        best_accuracy = mean_accuracy
                        best_params = current_params
                        best_resampling = undersampling

    print("Najlepsze parametry:", best_params)
    print("Undersampling =", best_resampling)
    return best_params

min_split = np.array([2, 3, 4, 5, 6, 7])
max_nvl = np.array([3, 4, 5, 6, 7, 9, 11, 13])
alg = ['entropy', 'gini']
values_grid = {
    'min_samples_split': min_split,
    'max_depth': max_nvl,
    'criterion': alg
}

model = DecisionTreeClassifier()


random_indices = np.random.choice(X.shape[0], size=5000, replace=False)
X_sampled = X[random_indices]
y_sampled = y.iloc[random_indices]
def_params = optimize_parameters(model, X_sampled, y_sampled, values_grid)

model = DecisionTreeClassifier(min_samples_split=7, max_depth=11, criterion='entropy')
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

accuracies_train = []
accuracies_test = []
y_preds = []
y_true = []

for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    smote = SMOTE(random_state=2)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model.fit(X_train_resampled, y_train_resampled)

    # Ocena na oryginalnych danych testowych
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    accuracies_train.append(calculate_metrics(y_train, y_pred_train))
    accuracies_test.append(calculate_metrics(y_test, y_pred_test))

    y_preds.extend(y_pred_test)
    y_true.extend(y_test)

result = pd.concat([aggregate_metrics(accuracies_train), aggregate_metrics(accuracies_test)])
result.index = ["train", "test"]

print_conf_matrix(y_true, y_preds)
print(result)


def optimize_parameters_rf(model, X, y, param_grid):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

    best_accuracy = 0
    best_params = {}
    undersampling = [True, False]
    best_resampling = "Undersampling"

    # Iteracja przez wszystkie kombinacje parametrów
    for undersampling in undersampling:
        for n_estimators in param_grid['n_estimators']:
            for max_depth in param_grid['max_depth']:
                for criterion in param_grid['criterion']:
                    for min_samples_split in param_grid['min_samples_split']:
                        current_params = {
                            'n_estimators': n_estimators,
                            'min_samples_split': min_samples_split,
                            'max_depth': max_depth,
                            'criterion': criterion
                        }

                        fold_accuracies = []

                        for train_idx, test_idx in cv.split(X, y):
                            X_train, X_test = X[train_idx], X[test_idx]
                            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                            # Undersampling

                            if undersampling == True:
                                undersampler = RandomUnderSampler(random_state=1)
                                X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
                            else:
                                smote = SMOTE(random_state=2)
                                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

                            # Trenowanie modelu
                            model.set_params(**current_params)
                            model.fit(X_train_resampled, y_train_resampled)

                            # Ewaluacja na zbiorze testowym
                            y_pred_test = model.predict(X_test)
                            accuracy = np.mean(y_test == y_pred_test)
                            fold_accuracies.append(accuracy)

                        # Średnia dokładność dla bieżącej kombinacji parametrów
                        mean_accuracy = np.mean(fold_accuracies)

                        # Sprawdzenie, czy aktualny model jest najlepszy
                        if mean_accuracy > best_accuracy:
                            best_accuracy = mean_accuracy
                            best_params = current_params
                            best_resampling = undersampling

    print("Najlepsze parametry:", best_params)
    print("Undersampling =", best_resampling)
    return best_params

from sklearn.ensemble import RandomForestClassifier

n_estimators = np.array([25,50,150])
alg = ['entropy', 'gini']
min_split = np.array([2, 3, 4, 5, 7])
max_nvl = np.array([3, 4, 5, 6, 9, 11])
values_grid = {'n_estimators': n_estimators, 'min_samples_split': min_split, 'max_depth': max_nvl, 'criterion': alg}


random_indices = np.random.choice(X.shape[0], size=5000, replace=False)
X_sampled = X[random_indices]
y_sampled = y.iloc[random_indices]
model = RandomForestClassifier()
def_params = optimize_parameters_rf(model, X_sampled, y_sampled, values_grid)

model = RandomForestClassifier(min_samples_split=5, max_depth=11, criterion='entropy', n_estimators=150)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

accuracies_train = []
accuracies_test = []
y_preds = []
y_true = []

for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    smote = SMOTE(random_state=2)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model.fit(X_train_resampled, y_train_resampled)

    # Ocena na oryginalnych danych testowych
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    accuracies_train.append(calculate_metrics(y_train, y_pred_train))
    accuracies_test.append(calculate_metrics(y_test, y_pred_test))

    y_preds.extend(y_pred_test)
    y_true.extend(y_test)

result = pd.concat([aggregate_metrics(accuracies_train), aggregate_metrics(accuracies_test)])
result.index = ["train", "test"]

print_conf_matrix(y_true, y_preds)
print(result)

from sklearn.ensemble import ExtraTreesClassifier

n_estimators = np.array([25,50,100])
alg = ['entropy', 'gini']
min_split = np.array([2, 3, 4, 5, 7])
max_nvl = np.array([3, 4, 5, 6, 9, 11, None])
values_grid = {'n_estimators': n_estimators, 'min_samples_split': min_split, 'max_depth': max_nvl, 'criterion': alg}


random_indices = np.random.choice(X.shape[0], size=5000, replace=False)
X_sampled = X[random_indices]
y_sampled = y.iloc[random_indices]

model = ExtraTreesClassifier()
params_et = optimize_parameters_rf(model, X_sampled, y_sampled, values_grid)

model = ExtraTreesClassifier(min_samples_split=5, max_depth=None, criterion='entropy', n_estimators=50)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

accuracies_train = []
accuracies_test = []
y_preds = []
y_true = []

for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    smote = SMOTE(random_state=2)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model.fit(X_train_resampled, y_train_resampled)

    # Ocena na oryginalnych danych testowych
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    accuracies_train.append(calculate_metrics(y_train, y_pred_train))
    accuracies_test.append(calculate_metrics(y_test, y_pred_test))

    y_preds.extend(y_pred_test)
    y_true.extend(y_test)

result = pd.concat([aggregate_metrics(accuracies_train), aggregate_metrics(accuracies_test)])
result.index = ["train", "test"]

print_conf_matrix(y_true, y_preds)
print(result)


def optimize_parameters_svm_lin(model, X, y, param_grid):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

    best_accuracy = 0
    best_params = {}
    undersampling = [True, False]
    best_resampling = "Undersampling"

    # Iteracja przez wszystkie kombinacje parametrów
    for undersampling in undersampling:
        for c in param_grid['C']:
            current_params = {
                'C': c,
                'kernel': param_grid['kernel'],
            }
            fold_accuracies = []
            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                # Undersampling
                if undersampling == True:
                    undersampler = RandomUnderSampler(random_state=1)
                    X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
                else:
                    smote = SMOTE(random_state=2)
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                # Trenowanie modelu
                model.set_params(**current_params)
                model.fit(X_train_resampled, y_train_resampled)
                # Ewaluacja na zbiorze testowym
                y_pred_test = model.predict(X_test)
                accuracy = np.mean(y_test == y_pred_test)
                fold_accuracies.append(accuracy)
            # Średnia dokładność dla bieżącej kombinacji parametrów
            mean_accuracy = np.mean(fold_accuracies)
            # Sprawdzenie, czy aktualny model jest najlepszy
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_params = current_params
                best_resampling = undersampling

    print("Najlepsze parametry:", best_params)
    print("Undersampling =", best_resampling)
    print("Najlepszy wynik:", best_accuracy)
    return best_params


def optimize_parameters_svm_poly(model, X, y, param_grid):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

    best_accuracy = 0
    best_params = {}
    undersampling = [True, False]
    best_resampling = "Undersampling"

    # Iteracja przez wszystkie kombinacje parametrów
    for undersampling in undersampling:
        for c in param_grid['C']:
            for degree in param_grid['degree']:
                for gamma in param_grid['gamma']:
                    current_params = {
                        'C': c,
                        'kernel': param_grid['kernel'],
                        'degree': degree,
                        'gamma': gamma
                    }
                    fold_accuracies = []
                    for train_idx, test_idx in cv.split(X, y):
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                        # Undersampling
                        if undersampling == True:
                            undersampler = RandomUnderSampler(random_state=1)
                            X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
                        else:
                            smote = SMOTE(random_state=2)
                            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                        # Trenowanie modelu
                        model.set_params(**current_params)
                        model.fit(X_train_resampled, y_train_resampled)
                        # Ewaluacja na zbiorze testowym
                        y_pred_test = model.predict(X_test)
                        accuracy = np.mean(y_test == y_pred_test)
                        fold_accuracies.append(accuracy)
                    # Średnia dokładność dla bieżącej kombinacji parametrów
                    mean_accuracy = np.mean(fold_accuracies)
                    # Sprawdzenie, czy aktualny model jest najlepszy
                    if mean_accuracy > best_accuracy:
                        best_accuracy = mean_accuracy
                        best_params = current_params
                        best_resampling = undersampling

    print("Najlepsze parametry:", best_params)
    print("Undersampling =", best_resampling)
    print("Najlepszy wynik:", best_accuracy)
    return best_params


def optimize_parameters_svm_rbf(model, X, y, param_grid):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

    best_accuracy = 0
    best_params = {}
    undersampling = [True, False]
    best_resampling = "Undersampling"

    # Iteracja przez wszystkie kombinacje parametrów
    for undersampling in undersampling:
        for c in param_grid['C']:
            for gamma in param_grid['gamma']:
                current_params = {
                    'C': c,
                    'kernel': param_grid['kernel'],
                    'gamma': gamma
                }
                fold_accuracies = []
                for train_idx, test_idx in cv.split(X, y):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    # Undersampling
                    if undersampling == True:
                        undersampler = RandomUnderSampler(random_state=1)
                        X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
                    else:
                        smote = SMOTE(random_state=2)
                        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                    # Trenowanie modelu
                    model.set_params(**current_params)
                    model.fit(X_train_resampled, y_train_resampled)
                    # Ewaluacja na zbiorze testowym
                    y_pred_test = model.predict(X_test)
                    accuracy = np.mean(y_test == y_pred_test)
                    fold_accuracies.append(accuracy)
                # Średnia dokładność dla bieżącej kombinacji parametrów
                mean_accuracy = np.mean(fold_accuracies)
                # Sprawdzenie, czy aktualny model jest najlepszy
                if mean_accuracy > best_accuracy:
                    best_accuracy = mean_accuracy
                    best_params = current_params
                    best_resampling = undersampling

    print("Najlepsze parametry:", best_params)
    print("Undersampling =", best_resampling)
    print("Najlepszy wynik:", best_accuracy)
    return best_params

param_grid_svm_lin = {
    'C': [0.001, 0.01, 0.05, 0.1, 0.25, 1],
    'kernel': "linear"
}

model = SVC()

random_indices = np.random.choice(X.shape[0], size=4000, replace=False)
X_sampled = X_standard[random_indices]
y_sampled = y.iloc[random_indices]
optimize_parameters_svm_lin(model, X_sampled, y_sampled, param_grid_svm_lin)

param_grid_svm_poly = {
    'C': [0.001, 0.01, 0.25, 0.5, 1],
    'kernel': "poly",
    'gamma': [0.01, 0.02, 0.075, 0.1, 0.2],
    'degree': [2,3,4,5]
}
model = SVC()
random_indices = np.random.choice(X.shape[0], size=1000, replace=False)
X_sampled = X_standard[random_indices]
y_sampled = y.iloc[random_indices]
optimize_parameters_svm_poly(model, X_sampled, y_sampled, param_grid_svm_poly)

param_grid_svm_rbf = {
    'C': [0.01, 0.05, 0.1, 0.25, 1],
    'kernel': "rbf",
    'gamma': [0.01, 0.02, 0.05, 0.075, 0.1,0.2,0.5]
}

model = SVC()
random_indices = np.random.choice(X.shape[0], size=3000, replace=False)
X_sampled = X_standard[random_indices]
y_sampled = y.iloc[random_indices]
optimize_parameters_svm_rbf(model, X_sampled, y_sampled, param_grid_svm_rbf)

model = SVC(kernel="rbf", C=1, gamma=0.2)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

accuracies_train = []
accuracies_test = []
y_preds = []
y_true = []

for train_idx, test_idx in cv.split(X_standard, y):
    X_train, X_test = X_standard[train_idx], X_standard[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    smote = SMOTE(random_state=2)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model.fit(X_train_resampled, y_train_resampled)

    # Ocena na oryginalnych danych testowych
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    accuracies_train.append(calculate_metrics(y_train, y_pred_train))
    accuracies_test.append(calculate_metrics(y_test, y_pred_test))

    y_preds.extend(y_pred_test)
    y_true.extend(y_test)

result = pd.concat([aggregate_metrics(accuracies_train), aggregate_metrics(accuracies_test)])
result.index = ["train", "test"]

print_conf_matrix(y_true, y_preds)
print(result)

data = {
    "Model": [
        "Naive Bayes (train)", "Naive Bayes (test)",
        "Decision Tree (train)", "Decision Tree (test)",
        "Random Forest (train)", "Random Forest (test)",
        "Extremely Randomized Trees (train)", "Extremely Randomized Trees (test)",
        "SVM (train)", "SVM (test)"
    ],
    "Sensitivity": [0.619392, 0.619325, 0.813767, 0.798574, 0.804424, 0.794336, 0.959615, 0.805603, 0.851959, 0.820202],
    "Precision": [0.566123, 0.566236, 0.803092, 0.786119, 0.818979, 0.808429, 0.980650, 0.854667, 0.752641, 0.728521],
    "Specificity": [0.773077, 0.773049, 0.904561, 0.896003, 0.915001, 0.909918, 0.990948, 0.934431, 0.866142, 0.853783],
    "Accuracy": [0.723372, 0.723330, 0.875196, 0.864492, 0.879238, 0.872537, 0.980814, 0.892766, 0.861555, 0.842923]
}

models_result = pd.DataFrame(data)
models_result_train = models_result[models_result["Model"].str.contains("train")]
models_result_test = models_result[models_result["Model"].str.contains("test")]
models_result_train_melted = models_result_train.melt(id_vars="Model", var_name="Metric", value_name="Value")
models_result_test_melted = models_result_test.melt(id_vars="Model", var_name="Metric", value_name="Value")

# Wykres dla danych treningowych
plt.figure(figsize=(12, 6))
sns.barplot(data=models_result_train_melted, x="Model", y="Value", hue="Metric")
plt.xticks(rotation=45, ha="right")
plt.title("Porównanie metryk dla danych treningowych")
plt.ylabel("Wartość")
plt.xlabel("Model")
plt.legend(title="Metryka")
plt.tight_layout()
plt.grid()
plt.show()

# Wykres dla danych testowych
plt.figure(figsize=(12, 6))
sns.barplot(data=models_result_test_melted, x="Model", y="Value", hue="Metric")
plt.xticks(rotation=45, ha="right")
plt.title("Porównanie metryk dla danych testowych")
plt.ylabel("Wartość")
plt.xlabel("Model")
plt.legend(title="Metryka")
plt.tight_layout()
plt.grid()
plt.show()

best_model = ExtraTreesClassifier(min_samples_split=5, max_depth=None, criterion='entropy', n_estimators=50)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
X_train_lab = pd.DataFrame(X_train, columns=df.drop(["booking_status"], axis=1).columns)
y_train_lab = pd.DataFrame(y_train, columns=["booking_status"])
best_model.fit(X_train_lab, y_train_lab)
importance = best_model.feature_importances_
plt.bar(range(len(importance)), importance)
plt.xticks(range(len(importance)), df.drop('booking_status', axis=1).columns, rotation=90, ha="right")
plt.xlabel("Feature")
plt.ylabel("Importances")
plt.show()

import shap
import dalex as dx

X_test_lab = pd.DataFrame(X_test, columns=df.drop(["booking_status"], axis=1).columns)
y_test_lab = pd.DataFrame(y_test, columns=["booking_status"])
explainer = dx.Explainer(best_model, X_test_lab, y_test_lab, label="Extremely Randomized Trees", verbose=0)

X_test_lab["arrival_month"].value_counts()
observation = X_test_lab.iloc[4]
print(observation)
pcp = explainer.predict_profile(observation)
pcp.plot(variables=["lead_time"])
pcp.plot(variables=["avg_price_per_room"])
pcp.plot(variables=["no_of_special_requests"])

pdp = explainer.model_profile(variables = ['lead_time'], groups="no_of_special_requests")
pdp.plot()
pdp.plot(geom='profiles')

pdp = explainer.model_profile(variables = ['avg_price_per_room'], groups="no_of_adults")
pdp.plot()
pdp.plot(geom='profiles')


observation1 = X_test_lab.iloc[[0]]
observation2 = X_test_lab.iloc[[1]]
observation3 = X_test_lab.iloc[[1440]]

print(observation1)
bd = explainer.predict_parts(observation1, type = 'break_down_interactions')
bd.plot()

print(observation2)
bd = explainer.predict_parts(observation2, type = 'break_down_interactions')
bd.plot()

print(observation3)
bd = explainer.predict_parts(observation3, type = 'break_down_interactions')
bd.plot()
























