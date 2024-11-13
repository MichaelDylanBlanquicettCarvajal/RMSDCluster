import numpy as np

def medoids(distance_matrix, labels):
    # Encontrar la conformación representativa (medoid) para cada cluster
    unique_clusters = np.unique(labels)
    medoids = {}
    for cluster in unique_clusters:
        indices = np.where(labels == cluster)[0]
        sub_matrix = distance_matrix[np.ix_(indices, indices)]
        medoid_index = indices[np.argmin(sub_matrix.sum(axis=0))]  # Índice del medoid
        medoids[cluster] = medoid_index
    return medoids


