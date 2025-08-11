import numpy as np
import scipy


def min_euclidean_distance(conditions: np.ndarray, candidates: np.ndarray, dist2prox: callable, metric: str='euclidean') -> np.ndarray:
    """
    Calculate the minimum Euclidean distance between two vectors.

    Args:
        conditions: Already existing datapoints, i.e. already sampled, shape (n, d).
        candidates: Sampled candidates for the next experiment, shape (c, d).

    Returns:
        distances: A 1d array including the distance to the closest condition datapoint for each candidate.

    """
    distance_matrix = scipy.spatial.distance.cdist(candidates, conditions, metric=metric) # shape (c, n)
    min_distances = np.min(distance_matrix, axis=1)  # shape (c,)
    return min_distances

def dist2prox_via_inverse(distances, epsilon=1e-5):
    """
    Convert distances to proximity values using an inverse function.

    Args:
        distances: A 1d array of distances.
        epsilon: A small value to avoid division by zero.

    Returns:
        proximities: A 1d array of proximity values.
    """
    return 1 / (distances + epsilon)  # shape (c,)

def proximity_gaussian_kernels(conditions: np.ndarray, candidates: np.ndarray, sigma:float=1., metric: str='euclidean') -> np.ndarray:
    """
    Calculate the minimum Euclidean distance between two vectors.

    Args:
        conditions: Already existing datapoints, i.e. already sampled, shape (n, d).
        candidates: Sampled candidates for the next experiment, shape (c, d).

    Returns:
        distances: A 1d array including the distance to the closest condition datapoint for each candidate.
    """
    distance_matrix = scipy.spatial.distance.cdist(candidates, conditions, metric=metric)  # shape (c, n)
    def gaussian_kernel(x_dist, sigma):
        #gaussian kernel, x_dist is the distance between mu and x
        #returns the probability of x given x_dist and sigma
        return np.exp(-0.5 * (x_dist / sigma) ** 2)
    gaussian_probabilities = gaussian_kernel(distance_matrix, sigma)  # shape (c, n)
    total_probabilities = np.sum(gaussian_probabilities, axis=1)  # shape (c,)
    return total_probabilities
