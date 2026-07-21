import unittest

import numpy as np

from data_preprocessing import Boundary
from deployment_results import align_uav_positions
from kmeans_deployment import KMeansConfig, fit_kmeans, run_kmeans_deployment
from random_deployment import run_random_deployment
from standard_pso import PSOConfig, run_standard_pso_baseline
from static_pso import run_static_pso


class BaselineAlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.boundary = Boundary(0.0, 100.0, 0.0, 100.0)
        first = np.array(
            [[18.0, 18.0], [20.0, 20.0], [22.0, 19.0], [80.0, 80.0]]
        )
        second = first + np.array([3.0, 2.0])
        third = second + np.array([2.0, 1.0])
        self.tensor = np.stack([first, second, third])
        self.slots = [0, 1, 2]
        self.pso = PSOConfig(particles=8, iterations=4)

    def test_alignment_preserves_nearest_uav_identity(self):
        previous = np.array([[10.0, 10.0], [90.0, 90.0]])
        current = np.array([[89.0, 91.0], [11.0, 9.0]])

        aligned = align_uav_positions(previous, current)

        np.testing.assert_allclose(aligned, current[[1, 0]])

    def test_random_deployment_is_deterministic_and_bounded(self):
        first = run_random_deployment(
            self.tensor, self.slots, self.boundary, uav_count=2, seed=5
        )
        second = run_random_deployment(
            self.tensor, self.slots, self.boundary, uav_count=2, seed=5
        )

        self.assertEqual(first.algorithm, "Random_Deployment")
        for left, right in zip(first.time_slices, second.time_slices):
            np.testing.assert_allclose(left.positions, right.positions)
            self.assertEqual(left.metrics.boundary_penalty, 0.0)

    def test_kmeans_finds_two_population_centers(self):
        points = np.array(
            [[9.0, 10.0], [11.0, 10.0], [89.0, 90.0], [91.0, 90.0]]
        )
        centers = fit_kmeans(
            points,
            2,
            self.boundary,
            np.random.default_rng(2),
            KMeansConfig(iterations=20),
        )
        centers = centers[np.argsort(centers[:, 0])]

        np.testing.assert_allclose(centers, [[10.0, 10.0], [90.0, 90.0]])

    def test_kmeans_runs_for_each_time_slice(self):
        result = run_kmeans_deployment(
            self.tensor, self.slots, self.boundary, uav_count=2, seed=7
        )

        self.assertEqual(result.algorithm, "KMeans")
        self.assertEqual(len(result.time_slices), 3)
        self.assertGreaterEqual(result.average_coverage_rate, 0.0)
        self.assertLessEqual(result.average_coverage_rate, 1.0)

    def test_standard_pso_restarts_for_each_time_slice(self):
        result = run_standard_pso_baseline(
            self.tensor,
            self.slots,
            self.boundary,
            uav_count=2,
            pso_config=self.pso,
            seed=9,
        )

        self.assertEqual(result.algorithm, "Standard_PSO")
        self.assertEqual(len(result.time_slices), 3)
        self.assertTrue(all(not item.warm_started for item in result.time_slices))
        self.assertTrue(
            all(len(item.convergence_history) == 5 for item in result.time_slices)
        )

    def test_static_pso_keeps_positions_fixed(self):
        result = run_static_pso(
            self.tensor,
            self.slots,
            self.boundary,
            uav_count=2,
            pso_config=self.pso,
            seed=10,
        )

        self.assertEqual(result.algorithm, "Static_PSO")
        reference = result.time_slices[0].positions
        for item in result.time_slices[1:]:
            np.testing.assert_allclose(item.positions, reference)
            self.assertEqual(item.metrics.movement_penalty, 0.0)
            self.assertEqual(item.metrics.movement_violations, 0)


if __name__ == "__main__":
    unittest.main()
