import unittest

import numpy as np

from data_preprocessing import Boundary
from standard_pso import PSOConfig, initialize_swarm, run_standard_pso


class StandardPSOTests(unittest.TestCase):
    def setUp(self):
        self.boundary = Boundary(0.0, 100.0, 0.0, 100.0)
        self.users = np.array(
            [
                [18.0, 18.0],
                [20.0, 20.0],
                [22.0, 19.0],
                [78.0, 79.0],
                [80.0, 80.0],
                [82.0, 78.0],
            ]
        )

    def test_default_parameters_match_v080_configuration(self):
        config = PSOConfig()

        self.assertEqual(config.particles, 32)
        self.assertEqual(config.iterations, 25)
        self.assertEqual(config.inertia, 0.72)
        self.assertEqual(config.c1, 1.45)
        self.assertEqual(config.c2, 1.45)

    def test_swarm_shape_and_boundary(self):
        swarm = initialize_swarm(
            np.random.default_rng(7), self.boundary, uav_count=5, particle_count=20
        )

        self.assertEqual(swarm.shape, (20, 5, 2))
        self.assertTrue((swarm[:, :, 0] >= self.boundary.xmin).all())
        self.assertTrue((swarm[:, :, 0] <= self.boundary.xmax).all())
        self.assertTrue((swarm[:, :, 1] >= self.boundary.ymin).all())
        self.assertTrue((swarm[:, :, 1] <= self.boundary.ymax).all())

    def test_result_is_deterministic_and_inside_boundary(self):
        config = PSOConfig(particles=8, iterations=5)
        first = run_standard_pso(
            self.users, self.boundary, uav_count=2, pso_config=config, seed=9
        )
        second = run_standard_pso(
            self.users, self.boundary, uav_count=2, pso_config=config, seed=9
        )

        np.testing.assert_allclose(first.positions, second.positions)
        self.assertAlmostEqual(first.metrics.fitness, second.metrics.fitness)
        self.assertTrue((first.positions[:, 0] >= self.boundary.xmin).all())
        self.assertTrue((first.positions[:, 0] <= self.boundary.xmax).all())
        self.assertTrue((first.positions[:, 1] >= self.boundary.ymin).all())
        self.assertTrue((first.positions[:, 1] <= self.boundary.ymax).all())

    def test_global_best_history_never_decreases(self):
        config = PSOConfig(particles=10, iterations=8)
        result = run_standard_pso(
            self.users, self.boundary, uav_count=2, pso_config=config, seed=12
        )

        history = np.asarray(result.convergence_history)
        self.assertEqual(len(history), config.iterations + 1)
        self.assertTrue((np.diff(history) >= -1e-12).all())
        self.assertGreaterEqual(result.metrics.fitness, result.initial_best_fitness)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            run_standard_pso(
                self.users,
                self.boundary,
                pso_config=PSOConfig(particles=0),
            )


if __name__ == "__main__":
    unittest.main()
