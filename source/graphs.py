import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def visualizacion_matriz_distancias(distance_matrix):
    # Visualización de la matriz de distancias
    sns.heatmap(distance_matrix, cmap='viridis')
    plt.title("Matriz de Distancias (RMSD)")
    plt.show()

def visualizacion_cluster(type_cluster, labels, num_frames):
    # Visualización de los resultados de clustering
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 1, 1)
    sns.scatterplot(x=np.arange(num_frames), y=labels, palette="viridis", marker="o")
    plt.title("Etiquetas "+type_cluster)
    plt.xlabel("Conformación")
    plt.ylabel("Cluster")
    