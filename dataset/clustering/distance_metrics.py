import numpy as np

def euclidean_distance(data: np.ndarray, diagonal_value=np.inf):
    """Calculates pairwise Euclidean distance between vectors in data matrix."""
    squared_norms = np.sum(data ** 2, axis=1, keepdims=True)
    distances = np.sqrt(squared_norms - 2 * np.dot(data, data.T) + squared_norms.T)
    np.fill_diagonal(distances, diagonal_value)
    return distances
