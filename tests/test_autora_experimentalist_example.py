import numpy as np
from src.autora.experimentalist.autora_experimentalist.proximity_functions import (
    min_euclidean_distance,
    dist2prox_via_inverse,
    proximity_gaussian_kernels
)

def test_min_euclidean_distance():
    conditions = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0]
    ])
    candidates = np.array([
        [1.0, 0.0],
        [3.0, 3.0]
    ])
    expected = np.array([
        1.0,
        np.sqrt(2)
    ])
    output = min_euclidean_distance(conditions, candidates, dist2prox_via_inverse)
    np.testing.assert_allclose(output, expected, rtol=1e-5, err_msg="Min distances incorrect")

def test_dist2prox_via_inverse():
    distances = np.array([1.0, np.sqrt(2)])
    expected = 1 / (distances + 1e-5)
    output = dist2prox_via_inverse(distances)
    np.testing.assert_allclose(output, expected, rtol=1e-5, err_msg="Inverse proximity incorrect")

def test_proximity_gaussian_kernels():
    conditions = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0]
    ])
    candidates = np.array([
        [1.0, 0.0],
        [3.0, 3.0]
    ])
    output = proximity_gaussian_kernels(conditions, candidates, sigma=1.0)

    # Check the output shape
    assert output.shape == (2,), "Output shape should be (2,)"

    # Gaussian proximity should be higher for closer candidates
    assert output[0] > output[1], "Closer candidate should have higher proximity"
