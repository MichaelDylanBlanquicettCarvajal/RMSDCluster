import mdtraj as md # type: ignore
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans, SpectralClustering
import numpy as np


def cargar_trajectoria(trajectory):
# Cargar la trayectoria
    traj = md.load(trajectory)

    # Calcular la matriz de distancias usando RMSD entre todas las conformaciones
    num_frames = traj.n_frames
    distance_matrix = np.zeros((num_frames, num_frames))

    for i in range(num_frames):
        rmsd_values = md.rmsd(traj, traj, frame=i)
        distance_matrix[i, :] = rmsd_values
        distance_matrix[:, i] = rmsd_values  # Simetría en la matriz de distancias
        
    return distance_matrix, num_frames

def clustering_jerarquico(distance_matrix, threshold):
    # Clustering jerárquico
    linkage_matrix = linkage(squareform(distance_matrix), method='average')
    labels_hierarchical = fcluster(linkage_matrix, threshold, criterion='distance')
    return labels_hierarchical

def clustering_dbscan(distance_matrix, eps_value):
    # DBSCAN con eps ajustado
    dbscan_model = DBSCAN(eps=eps_value, min_samples=2, metric='precomputed')
    labels_dbscan = dbscan_model.fit_predict(distance_matrix)
    return labels_dbscan

def clustering_kmeans(distance_matrix, num_clusters):
    # Clustering con K-means
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels_kmeans = kmeans.fit_predict(distance_matrix)
    return labels_kmeans

def clustering_spectral(distance_matrix, num_clusters):
    # Clustering espectral (Spectral Clustering)
    spectral = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=0)
    labels_spectral = spectral.fit_predict(distance_matrix)
    return labels_spectral
