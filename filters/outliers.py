import numpy as np

def filtrar_outliers_percentiles(distance_matrix, lower_percentile=1, upper_percentile=99):
    """
    Filtra outliers en una matriz de distancias usando percentiles.

    :param distance_matrix: Matriz de distancias (np.array)
    :param lower_percentile: Percentil inferior para filtrar (por defecto 1)
    :param upper_percentile: Percentil superior para filtrar (por defecto 99)
    :return: Matriz de distancias filtrada (np.array)
    """
    # Calcular las distancias promedio de cada conformación
    avg_distances = np.mean(distance_matrix, axis=1)
    
    # Calcular los percentiles
    lower_threshold = np.percentile(avg_distances, lower_percentile)
    upper_threshold = np.percentile(avg_distances, upper_percentile)
    
    # Filtrar las conformaciones que están fuera de los percentiles
    non_outliers = (avg_distances >= lower_threshold) & (avg_distances <= upper_threshold)
    
    # Aplicar el filtro a la matriz de distancias
    filtered_distance_matrix = distance_matrix[non_outliers][:, non_outliers]

    return filtered_distance_matrix



