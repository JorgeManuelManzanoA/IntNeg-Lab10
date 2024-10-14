import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generar datos de ejemplo
np.random.seed(42)
n_samples = 200
precio_actual = np.random.uniform(10, 1000, n_samples)
precio_final = precio_actual * np.random.uniform(0.8, 1.2, n_samples)

# Crear un DataFrame
data = pd.DataFrame({'Precio actual': precio_actual, 'Precio final': precio_final})

# Seleccionar características para K-means
X_kmeans = data[['Precio actual', 'Precio final']]

# Encontrar el número óptimo de clústeres utilizando el método del codo
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_kmeans)
    inertia.append(kmeans.inertia_)

# Graficar el método del codo con mejoras de apariencia
plt.figure(figsize=(10, 7))
plt.plot(range(1, 11), inertia, marker='o', color='darkorange', linestyle='-', linewidth=2, markersize=10)
plt.title('Método del Codo para K-means', fontsize=18, fontweight='bold', color='navy')
plt.xlabel('Número de Clústeres', fontsize=14)
plt.ylabel('Inercia', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Elegir un número de clústeres y ajustar el modelo
n_clusters = 4  # Elegir 4 clústeres
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_kmeans)

# Graficar los clústeres con mejoras visuales
plt.figure(figsize=(10, 7))
sns.scatterplot(data=data, x='Precio actual', y='Precio final', hue='Cluster', palette='coolwarm', s=150, edgecolor='black', linewidth=1.5)
plt.title('Clustering de K-means con Datos Simulados', fontsize=18, fontweight='bold', color='navy')
plt.xlabel('Precio Actual', fontsize=14)
plt.ylabel('Precio Final', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Clúster', fontsize=12, title_fontsize=14, loc='upper right', bbox_to_anchor=(1.15, 1))
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
