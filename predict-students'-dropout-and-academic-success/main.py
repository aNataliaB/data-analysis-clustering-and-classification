import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. Wczytywanie danych
print("Wczytywanie danych z dysku lokalnego...")
df = pd.read_csv("data.csv", sep=";")
print(f"Dane zostały pomyślnie wczytane. Liczba wierszy: {len(df)}")

# 2. Selekcja zmiennych
# Definicja zmiennych wybranych do analizy w raporcie
selected_features = [
    "Age at enrollment",
    "Admission grade",
    "Debtor",
    "Tuition fees up to date",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
]

# Wycięcie podzbioru danych
df_analiza = df[selected_features].copy()

# Wyodrębnienie zmiennych ilościowych (ciągłych i dyskretnych) do statystyk i wykresów
quantitative_features = [
    "Age at enrollment",
    "Admission grade",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
]

# 3. Statystyki opisowe
print("\nGENEROWANIE STATYSTYK OPISOWYCH")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

stats = pd.DataFrame()

stats["Średnia"] = df_analiza.mean()
stats["Mediana"] = df_analiza.median()
stats["Minimum"] = df_analiza.min()
stats["Maksimum"] = df_analiza.max()
stats["Odchylenie std."] = df_analiza.std()
stats["Skośność"] = df_analiza.skew()

stats.index = [
    "Age at enroll.",
    "Admission grade",
    "Debtor",
    "Tuition fees",
    "Units 1st(app)",
    "Units 1st(grad)",
    "Units 2nd(app)",
    "Units 2nd(grad)"
]

stats_rounded = stats.round(2)
print(stats_rounded)

# Weryfikacja braków danych
print("\nKONTROLA BRAKÓW DANYCH (Suma brakujących wartości)")
print(df_analiza.isnull().sum())

# 4. Wizualizacja (Przed transformacjami)
sns.set_theme(style="whitegrid")

# Boxploty
plt.figure(figsize=(14, 8))
for i, feature in enumerate(quantitative_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df_analiza[feature], color="skyblue")
    plt.title(f"Wykres pudełkowy:\n{feature}", fontsize=11)
    plt.ylabel("Wartość")
plt.suptitle(
    "Wykresy pudełkowe dla zmiennych ilościowych (przed transformacją)",
    y=0.98,
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout(h_pad=3.0, w_pad=1.5, rect=[0, 0, 1, 0.93])
plt.show()

# histogramy
plt.figure(figsize=(14, 8))
for i, feature in enumerate(quantitative_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df_analiza[feature], kde=True, color="teal", bins=15, edgecolor="white", linewidth=0.7)
    plt.title(f"Histogram:\n{feature}", fontsize=11)
    plt.xlabel("Wartość")
    plt.ylabel("Częstość")
plt.suptitle(
    "Histogramy z krzywą gęstości (przed transformacją)",
    y=0.98,
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout(h_pad=3.0, w_pad=1.5, rect=[0, 0, 1, 0.93])
plt.show()

# Wykresy kołowe
debtor_counts = df["Debtor"].value_counts()
tuition_counts = df["Tuition fees up to date"].value_counts()

labels_debtor = ["Brak zadłużenia (0)", "Posiada zadłużenie (1)"]
labels_tuition = ["Nieuregulowane (0)", "Uregulowane na bieżąco (1)"]
colors_debtor = ["#99ff99", "#ff9999"]
colors_tuition = ["#ff9999", "#99ff99"]

plt.figure(figsize=(12, 5))

# Wykres 1: Debtor
plt.subplot(1, 2, 1)
plt.pie(
    debtor_counts,
    labels=labels_debtor,
    autopct="%.1f%%",
    startangle=90,
    colors=colors_debtor,
    textprops={'fontsize': 11}
)
plt.title("Struktura studentów pod kątem\nposiadanego zadłużenia (Debtor)", fontsize=12, pad=15)

# Wykres 2: Tuition fees
plt.subplot(1, 2, 2)
plt.pie(
    tuition_counts,
    labels=labels_tuition,
    autopct="%.1f%%",
    startangle=90,
    colors=colors_tuition,
    textprops={'fontsize': 11}
)
plt.title("Struktura studentów pod kątem\nterminowości opłat (Tuition fees up to date)", fontsize=12, pad=15)

plt.suptitle(
    "Struktura procentowa zmiennych binarnych w badanej próbie",
    y=0.98,
    fontsize=14,
    fontweight="bold"
)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.show()

# 5. Transformacje danych
print("\nRozpoczynanie procedury transformacji danych...")

df_transformed = df_analiza.copy()

# A. Transformacja logarytmiczna zmiennej 'Age at enrollment' (log(x + 1))
df_transformed["Age at enrollment"] = np.log1p(df_transformed["Age at enrollment"])
print(
    f"Skośność zmiennej 'Age at enrollment' po logarytmowaniu: {df_transformed['Age at enrollment'].skew():.2f}"
)

# B. Standaryzacja (Skalowanie) wszystkich ciągłych cech ilościowych
scaler = StandardScaler()
df_transformed[quantitative_features] = scaler.fit_transform(
    df_transformed[quantitative_features]
)
print("Standaryzacja cech ilościowych została zakończona.")

# 6. Wizaualizacja po transformacjach (Weryfikacja efektów)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df_analiza["Age at enrollment"], kde=True, color="darkred", bins=20)
plt.title("Rozkład oryginalny 'Age at enrollment'\n(Silna asymetria dodatnia)")

plt.subplot(1, 2, 2)
sns.histplot(
    df_transformed["Age at enrollment"], kde=True, color="darkgreen", bins=20
)
plt.title("Rozkład po transformacji logarytmicznej\n(Skorygowana asymetria)")

plt.suptitle(
    "Porównanie rozkładu wieku przed i po transformacji",
    y=0.98,
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

#  7. Przygotowanie danych do modelu
y = df["Target"]

# Kodowanie klas tekstowych na liczby
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X = df_transformed

# Podział danych: 80% treningowe, 20% testowe
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nDane zostały podzielone na zbiory treningowe i testowe.")

# Regresja logistyczna
log_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

# Trenowanie modelu
log_model.fit(X_train, y_train)

# Predykcja
y_pred_log = log_model.predict(X_test)

# Metryki
accuracy_log = accuracy_score(y_test, y_pred_log)
f1_log = f1_score(y_test, y_pred_log, average="weighted")

print("\nREGRESJA LOGISTYCZNA")
print(f"Accuracy: {accuracy_log:.4f}")
print(f"F1-score: {f1_log:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

from sklearn.ensemble import RandomForestClassifier

# 8. Las losowy
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

# Trenowanie modelu
rf_model.fit(X_train, y_train)

# Predykcja
y_pred_rf = rf_model.predict(X_test)

# Metryki
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average="weighted")

print("\nRANDOM FOREST")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"F1-score: {f1_rf:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

from sklearn.svm import SVC

# 9. SVM
svm_model = SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced",
    random_state=42
)

# Trenowanie modelu
svm_model.fit(X_train, y_train)

# Predykcja
y_pred_svm = svm_model.predict(X_test)

# Metryki
accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average="weighted")

print("\n--- SVM ---")
print(f"Accuracy: {accuracy_svm:.4f}")
print(f"F1-score: {f1_svm:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

# 10. Model hybrydowy
proba_log = log_model.predict_proba(X_test)
proba_rf = rf_model.predict_proba(X_test)
proba_svm = svm_model.predict_proba(X_test)

# Większa waga dla Random Forest (najlepszy model)
hybrid_proba = (
    0.2 * proba_log +
    0.5 * proba_rf +
    0.3 * proba_svm
)

# Wybór klasy o najwyższym prawdopodobieństwie
y_pred_hybrid = np.argmax(hybrid_proba, axis=1)

# Metryki
accuracy_hybrid = accuracy_score(y_test, y_pred_hybrid)
f1_hybrid = f1_score(y_test, y_pred_hybrid, average="weighted")

print("\nMODEL HYBRYDOWY")
print(f"Accuracy: {accuracy_hybrid:.4f}")
print(f"F1-score: {f1_hybrid:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_hybrid))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_hybrid))
from sklearn.model_selection import cross_val_score

# 11. Walidacja krzyżowa
print("\nWALIDACJA KRZYŻOWA")
cv_scores_rf = cross_val_score(
    rf_model,
    X,
    y_encoded,
    cv=5,
    scoring="accuracy"
)

print(f"Wyniki foldów: {cv_scores_rf}")
print(f"Średnia Accuracy CV: {cv_scores_rf.mean():.4f}")
print(f"Odchylenie standardowe CV: {cv_scores_rf.std():.4f}")

# 12. Porównanie modeli
# Tabela wyników
results_df = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "SVM",
        "Hybrid Ensemble"
    ],
    "Accuracy": [
        accuracy_log,
        accuracy_rf,
        accuracy_svm,
        accuracy_hybrid
    ],
    "F1-score": [
        f1_log,
        f1_rf,
        f1_svm,
        f1_hybrid
    ]
})

print("\nPORÓWNANIE MODELI")
print(results_df)

# Wykres Accuracy
plt.figure(figsize=(8, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="Accuracy",
    hue="Model",
    palette="viridis",
    legend=False
)

plt.title("Porównanie Accuracy modeli klasyfikacyjnych",
          y=0.98,
          fontsize=14,
          fontweight="bold",
          pad=15
)
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.ylim(0.6, 0.8)

for i, value in enumerate(results_df["Accuracy"]):
    plt.text(i, value + 0.005, f"{value:.3f}", ha='center')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

# Wykres F1-score
plt.figure(figsize=(8, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="F1-score",
    hue="Model",
    palette="magma",
    legend=False
)

plt.title("Porównanie F1-score modeli klasyfikacyjnych",
          y=0.98,
          fontsize=14,
          fontweight="bold",
          pad=15
)
plt.ylabel("F1-score")
plt.xlabel("Model")
plt.ylim(0.6, 0.8)

for i, value in enumerate(results_df["F1-score"]):
    plt.text(i, value + 0.005, f"{value:.3f}", ha='center')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

# 13. FEATURE IMPORTANCE (RANDOM FOREST)
# Pobranie ważności cech
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

# Sortowanie malejąco
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n--- WAŻNOŚĆ CECH (RANDOM FOREST) ---")
print(feature_importance)

# Wykres ważności cech
plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature",
    hue="Feature",
    palette="crest",
    legend=False
)

plt.title("Ważność cech w modelu Random Forest",
          y=0.98,
          fontsize=14,
          fontweight="bold",
          pad=15
          )
plt.xlabel("Importance")
plt.ylabel("Cecha")


plt.tight_layout(w_pad=1.5, rect=[0, 0, 1, 0.93])
plt.show()

# 14. Przykład użycia modelu
print("\nPRZYKŁADOWA PREDYKCJA NOWEGO STUDENTA")
# Dane przykładowego studenta
new_student = pd.DataFrame([{
    "Age at enrollment": 22,
    "Admission grade": 135,
    "Debtor": 0,
    "Tuition fees up to date": 1,
    "Curricular units 1st sem (approved)": 6,
    "Curricular units 1st sem (grade)": 14,
    "Curricular units 2nd sem (approved)": 5,
    "Curricular units 2nd sem (grade)": 13
}])

# Transformacja logarytmiczna wieku
new_student["Age at enrollment"] = np.log1p(
    new_student["Age at enrollment"]
)

# Skalowanie danych
new_student[quantitative_features] = scaler.transform(
    new_student[quantitative_features]
)

# Predykcja modelem hybrydowym
new_proba_log = log_model.predict_proba(new_student)
new_proba_rf = rf_model.predict_proba(new_student)
new_proba_svm = svm_model.predict_proba(new_student)

new_hybrid_proba = (
    0.2 * new_proba_log +
    0.5 * new_proba_rf +
    0.3 * new_proba_svm
)

new_prediction = np.argmax(new_hybrid_proba, axis=1)

# Odkodowanie klasy
predicted_class = label_encoder.inverse_transform(new_prediction)

print(f"Przewidywany status studenta: {predicted_class[0]}")
print("\nSkrypt wykonał się pomyślnie. Wszystkie wykresy zostały wygenerowane.")