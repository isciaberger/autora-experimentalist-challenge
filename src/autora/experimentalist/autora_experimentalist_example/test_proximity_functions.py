import unittest
import numpy as np
from proximity_functions import (
    min_euclidean_distance,
    dist2prox_via_inverse,
    proximity_gaussian_kernels
)


class TestProximityFunctions(unittest.TestCase):

    def setUp(self):
        # Sample data for tests
        self.conditions = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0]
        ])
        self.candidates = np.array([
            [1.0, 0.0],
            [3.0, 3.0]
        ])

    def test_min_euclidean_distance(self):
        expected = np.array([
            1.0,
            np.sqrt(2)
        ])
        output = min_euclidean_distance(self.conditions, self.candidates, dist2prox_via_inverse)
        np.testing.assert_allclose(output, expected, rtol=1e-5, err_msg="Min distances incorrect")

    def test_dist2prox_via_inverse(self):
        distances = np.array([1.0, np.sqrt(2)])
        expected = 1 / (distances + 1e-5)
        output = dist2prox_via_inverse(distances)
        np.testing.assert_allclose(output, expected, rtol=1e-5, err_msg="Inverse proximity incorrect")

    def test_proximity_gaussian_kernels(self):
        # Gaussian kernel should give higher values for closer candidates
        output = proximity_gaussian_kernels(self.conditions, self.candidates, sigma=1.0)
        self.assertEqual(output.shape, (2,))
        self.assertTrue(output[0] > output[1], "Closer candidate should have higher proximity")

if __name__ == '__main__':
    unittest.main()
