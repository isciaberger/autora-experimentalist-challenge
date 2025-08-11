import numpy as np
from src.autora.experimentalist.autora_experimentalist_example.proximity_functions import min_euclidean_distance, dist2prox_via_inverse, proximity_gaussian_kernels

def test_min_euclidean_distance():
    conditions = np.array([[0, 0], [1, 1]])
    candidates = np.array([[0, 1], [2, 2]])

    # Erwartete minimale Distanzen:
    # Candidate [0, 1] -> min(1, sqrt(1^2 + 0^2)) = 1
    # Candidate [2, 2] -> min(sqrt(8), sqrt(2)) = sqrt(2)
    expected = np.array([1.0, np.sqrt(2)])

    result = min_euclidean_distance(conditions, candidates, dist2prox=None)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_dist2prox_via_inverse():
    distances = np.array([1.0, 2.0, 0.0])
    epsilon = 1e-5

    # Expected: 1 / (d + epsilon)
    expected = np.array([1/(1+epsilon), 1/(2+epsilon), 1/(0+epsilon)])

    result = dist2prox_via_inverse(distances, epsilon=epsilon)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_proximity_gaussian_kernels():
    conditions = np.array([[0, 0], [1, 1]])
    candidates = np.array([[0, 1], [2, 2]])
    sigma = 1.0

    # Manuell erwartete Werte berechnen
    def gaussian_kernel(x_dist, sigma):
        return np.exp(-0.5 * (x_dist / sigma) ** 2)

    distance_matrix = np.array([
        [np.linalg.norm([0, 1] - np.array([0, 0])),
         np.linalg.norm([0, 1] - np.array([1, 1]))],
        [np.linalg.norm([2, 2] - np.array([0, 0])),
         np.linalg.norm([2, 2] - np.array([1, 1]))]
    ])
    expected = np.sum(gaussian_kernel(distance_matrix, sigma), axis=1)

    result = proximity_gaussian_kernels(conditions, candidates, sigma=sigma)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
