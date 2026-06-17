# PROJEKT: Air Quality 

# 1. Import bibliotek
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2. Wczytanie danych
df = pd.read_csv("AirQualityUCI.csv", sep=';', decimal=',')

# usunięcie pustych kolumn
df = df.drop(columns=['Unnamed: 15', 'Unnamed: 16'])

# zamiana -200 na NaN
df = df.replace(-200, np.nan)

# 3. Wybór zmiennych
features = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)', 'T']
df_selected = df[features]

# usunięcie braków
df_selected = df_selected.dropna()

print("Rozmiar danych po czyszczeniu:", df_selected.shape)

# 4. Analiza danych
print("\nStatystyki:")
print(df_selected.describe())

print("\nSkośność:")
print(df_selected.skew())

# histogramy
df_selected.hist(figsize=(10, 8))
plt.suptitle("Histogramy zmiennych")
plt.show()

# boxplot
df_selected.boxplot(figsize=(10,6))
plt.title("Boxplot zmiennych")
plt.show()

# 5. Standaryzacja
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(df_selected)

# 6. Wybór listy klastrów (ELBOW)
from sklearn.cluster import KMeans

inertia = []

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(range(2, 10), inertia, marker='o')
plt.title("Elbow method")
plt.xlabel("Liczba klastrów")
plt.ylabel("Inertia")
plt.show()

# 7. clustering (k=3)
kmeans = KMeans(n_clusters=3, random_state=0)
labels_kmeans = kmeans.fit_predict(X)

# 8. meoda warda (hierarchiczna)
from sklearn.cluster import AgglomerativeClustering

ward = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_ward = ward.fit_predict(X)

# 9. Ocena
from sklearn.metrics import silhouette_score

print("\nSilhouette score:")
print("KMeans:", silhouette_score(X, labels_kmeans))
print("Ward:", silhouette_score(X, labels_ward))

# 10. Porównanie metod
similarity = np.mean(labels_kmeans == labels_ward)
print("\nPodobieństwo klastrów:", similarity)

# 11. Wizualizacja (PCA)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# KMeans
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans)
plt.title("KMeans clustering")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

# Ward
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_ward)
plt.title("Ward clustering")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

# 12. Miara redukcji wariancji
# Obliczenie całkowitej sumy kwadratów (TSS)
overall_mean = np.mean(X, axis=0)
TSS = np.sum((X - overall_mean) ** 2)

#KMeans
WSS_kmeans = kmeans.inertia_  # suma kwadratów wewnątrz klastrów
BSS_kmeans = TSS - WSS_kmeans
R2_kmeans = BSS_kmeans / TSS

# Metoda Warda
# Funkcja pomocnicza do obliczenia WSS dla dowolnych etykiet klastrów
def compute_wss(X, labels):
    wss = 0
    for cluster in np.unique(labels):
        cluster_points = X[labels == cluster]
        centroid = np.mean(cluster_points, axis=0)
        wss += np.sum((cluster_points - centroid) ** 2)
    return wss

WSS_ward = compute_wss(X, labels_ward)
BSS_ward = TSS - WSS_ward
R2_ward = BSS_ward / TSS

print("\nMiara redukcji wariancji (R^2):")
print(f"KMeans: {R2_kmeans:.4f} ({R2_kmeans*100:.2f}%)")
print(f"Ward: {R2_ward:.4f} ({R2_ward*100:.2f}%)")