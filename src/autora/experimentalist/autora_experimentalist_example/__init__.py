"""
Example Experimentalist
"""
import numpy as np
from autora.experimentalist.autora_experimentalist_example.proximity_functions import dist2prox_via_inverse, min_euclidean_distance, proximity_gaussian_kernels, reweight_flavour
from autora.state import Delta

def sample(experiment_data,
            models_bms,
            models_lr,
            models_polyr,
            all_conditions,
            num_samples=1,
            random_state=None):
    """
    Sample new conditions based on existing conditions and models.
    Args:
        conditions:
        models:
        reference_conditions:
        num_samples:

    Returns:

    """
    args_dict = {
        "experiment_data": experiment_data,
        "models_bms": models_bms,
        "models_lr": models_lr,
        "models_polyr": models_polyr,
        "all_conditions": all_conditions,
        "num_samples": num_samples,
        "random_state": random_state
    }
    # Log the sampling process
    with open('sample_log.txt', 'a') as f:
        for key, value in args_dict.items():
            f.write(f"{key}: {value}\n")
    """
    experiment data is a pandas dataframe with the following columns:
    experiment_data: P_asymptotic
    trial
    performance
    0
    0.429293
    64.0
    0.916330
    1
    0.500000
    100.0
    0.975106
    """
    #take all the data except the first column, which is the index
    conditions = experiment_data.iloc[:, 1:].values  # Convert to numpy array
    # all_conditions is a pandas dataframe with the following columns:
    """
    all_conditions:      P_asymptotic  trial
    0              0.0    1.0
    1              0.0    2.0
    2              0.0    3.0
    3              0.0    4.0
    4              0.0    5.0
    ...            ...    ...
    9995           0.5   96.0
    9996           0.5   97.0
    9997           0.5   98.0
    9998           0.5   99.0
    9999           0.5  100.0
    """
    candidates = all_conditions.iloc[:, 1:].values  # Convert to numpy array
    # Sample candidates based on proximity to existing conditions
    #conditions = None
    #candidates = None #requires ndarray of shape candidates, dimensions
    num_samples = 1
    temperature = 10
    #sampler = None #, "inverse"  # or "gaussian"
    sampler = 'gaussian'  # or "inverse"
    sigma = 1.0
    random_state = 1312  # Set a random state for reproducibility
    alg_proposed_experiments = sample_flavour(
        conditions=conditions,
        candidates=candidates,
        sampler=sampler,
        num_samples=num_samples,
        temperature=temperature,
        sigma=sigma,
        random_state=random_state
    )
    print(f'!!!!!!!!!!!!!!!!!!!!!!1{alg_proposed_experiments}!!!!!!!!!!!!!!!!!!!!!!!!')
    return Delta(conditions=alg_proposed_experiments)


def sample_flavour(conditions, candidates, sampler="inverse", num_samples=1,
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
