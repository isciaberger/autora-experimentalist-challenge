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
    conditions = np.atleast_2d(conditions)
    candidates = np.atleast_2d(candidates)

    if conditions.shape[1] != candidates.shape[1]:
        raise ValueError(f"Shape mismatch: conditions.shape = {conditions.shape}, candidates.shape = {candidates.shape}")

    distance_matrix = scipy.spatial.distance.cdist(candidates, conditions, metric=metric)
    min_distances = np.min(distance_matrix, axis=1)
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

def proximity_gaussian_kernels(conditions: np.ndarray, candidates: np.ndarray, sigma: float = 1., metric: str = 'euclidean') -> np.ndarray:
    """
    Calculate the minimum Euclidean distance between two vectors.

    Args:
        conditions: Already existing datapoints, i.e. already sampled, shape (n, d).
        candidates: Sampled candidates for the next experiment, shape (c, d).

    Returns:
        distances: A 1d array including the distance to the closest condition datapoint for each candidate.
    """
    #normalize axes
    def norm(x):
        x_mean = np.mean(x, axis=1, keepdims=True)
        x_std = np.std(x, axis=1, keepdims=True)
        return (x - x_mean) / (x_std + 1e-8)
    conditions = norm(conditions)
    candidates = norm(candidates)

    conditions = np.atleast_2d(conditions)
    candidates = np.atleast_2d(candidates)

    if conditions.shape[1] != candidates.shape[1]:
        raise ValueError(f"Shape mismatch: conditions.shape = {conditions.shape}, candidates.shape = {candidates.shape}")

    distance_matrix = scipy.spatial.distance.cdist(candidates, conditions, metric=metric)

    def gaussian_kernel(x_dist, sigma):
        return np.exp(-0.5 * (x_dist / sigma) ** 2)

    gaussian_probabilities = gaussian_kernel(distance_matrix, sigma)
    total_probabilities = np.sum(gaussian_probabilities, axis=1)
    return total_probabilities


def reweight_flavour(proximity, temperatur=1):
    """
    Converts a vector of proximity scores into a probability distribution 
    using a temperature-scaled Softmax.

    Args:
        proximity : proximity of candidates
        temperatur : temperature

    Returns:
        np.ndarray: Probability distribution over the items 
    """

    scaled= proximity / temperatur
    probs = np.exp(scaled) / np.sum(np.exp(scaled))

    return probs

