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



def perform_sampling(conditions, candidates, num_samples=1,
                   temperature=1.0, sigma=1.0, 
                   random_state=None):
    """
    Sample candidates based on their proximity to already sampled conditions.

    Args:
        conditions (np.ndarray): Already existing datapoints
        candidates (np.ndarray): Candidates
        num_samples (int): Number of points to sample
        temperature (float): Temperature for softmax reweighting
        sigma (float): Sigma for Gaussian kernel proximity
        random_state (int or None): Random seed for reproducibility

    Returns:
        np.ndarray: Selected candidate points
    """
    rng = np.random.default_rng(random_state)

    # Compute proximity
    distances = min_euclidean_distance(conditions, candidates, dist2prox=None)
    proximities_inverse = dist2prox_via_inverse(distances)
    proximities_gaussian = proximity_gaussian_kernels(conditions, candidates, sigma=sigma)
    
    # Convert to probability distribution
    probs_gaussian = reweight_flavour(proximities_gaussian, temperatur=temperature)
    probs_inverse = reweight_flavour(proximities_inverse, temperatur=temperature)

    # decide which proximity measure will be used?

    # Sample from candidates
    chosen_indices_gaussian = rng.choice(len(candidates), size=num_samples, replace=False, p=probs_gaussian)
    chosen_indices_inverse = rng.choice(len(candidates), size=num_samples, replace=False, p=probs_inverse)

    return candidates[chosen_indices_gaussian], candidates[chosen_indices_inverse]


