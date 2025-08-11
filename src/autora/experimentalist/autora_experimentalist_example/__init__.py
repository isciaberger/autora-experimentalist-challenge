"""
Example Experimentalist
"""
import numpy as np
from autora.experimentalist.autora_experimentalist_example.proximity_functions import dist2prox_via_inverse, min_euclidean_distance, proximity_gaussian_kernels, reweight_flavour


def sample(conditions, candidates, sampler="inverse", num_samples=1,
                   temperature=1.0, sigma=1.0, 
                   random_state=None):
    """
    Sample candidates based on their proximity to already sampled conditions.

    Args:
        conditions (np.ndarray): Already existing datapoints
        candidates (np.ndarray): Candidates
        sampler (string): "inverse" or "gaussian", proximity method
        num_samples (int): Number of points to sample
        temperature (float): Temperature for softmax reweighting
        sigma (float): Sigma for Gaussian kernel proximity
        random_state (int or None): Random seed for reproducibility

    Returns:
        np.ndarray: Selected candidate points
    """
    rng = np.random.default_rng(random_state)

    # Compute proximity
    if sampler == "inverse":
        distances = min_euclidean_distance(conditions, candidates, dist2prox=None)
        proximities = dist2prox_via_inverse(distances)
    elif sampler == "gaussian":
        proximities = proximity_gaussian_kernels(conditions, candidates, sigma=sigma)
    else:
        raise ValueError(f"Unknown sampler '{sampler}'. Expected one of: 'inverse', 'gaussian'.")
    
    # Convert to probability distribution
    probs = reweight_flavour(proximities, temperatur=temperature)

    # decide which proximity measure will be used?

    # Sample from candidates
    chosen_indices = rng.choice(len(candidates), size=num_samples, replace=False, p=probs)

    return candidates[chosen_indices]
