import unittest

import numpy as np

from data_preprocessing import Boundary
from dynamic_mas_pso import (
    WarmStartConfig,
    initialize_warm_swarm,
    run_dynamic_mas_pso,
)
from standard_pso import PSOConfig


class DynamicMASPSOTests(unittest.TestCase):
    def setUp(self):
        self.boundary = Boundary(0.0, 100.0, 0.0, 100.0)
        first = np.array(
            [[18.0, 18.0], [20.0, 20.0], [22.0, 19.0], [80.0, 80.0]]
        )
        second = first + np.array([3.0, 2.0])
        third = second + np.array([2.0, 1.0])
        self.tensor = np.stack([first, second, third])
        self.pso = PSOConfig(particles=10, iterations=5)

    def test_default_warm_start_parameters_remain_unchanged_in_v080(self):
        config = WarmStartConfig()

        self.assertEqual(config.noise_std, 5.0)
        self.assertEqual(config.random_restart_ratio, 0.20)

    def test_warm_start_keeps_previous_best_and_random_restarts(self):
        previous = np.array([[10.0, 10.0], [90.0, 90.0]])
        swarm = initialize_warm_swarm(
            np.random.default_rng(4),
            previous,
            self.boundary,
            particle_count=10,
            config=WarmStartConfig(noise_std=0.0, random_restart_ratio=0.20),
        )

        np.testing.assert_allclose(swarm[0], previous)
        np.testing.assert_allclose(swarm[1:8], np.broadcast_to(previous, (7, 2, 2)))
        self.assertFalse(np.allclose(swarm[8:], previous))

    def test_first_slice_is_random_and_later_slices_use_warm_start(self):
        result = run_dynamic_mas_pso(
            self.tensor,
            [0, 1, 2],
            self.boundary,
            uav_count=2,
            pso_config=self.pso,
            seed=8,
        )

        self.assertEqual(len(result.time_slices), 3)
        self.assertFalse(result.time_slices[0].warm_started)
        self.assertTrue(result.time_slices[1].warm_started)
        self.assertTrue(result.time_slices[2].warm_started)
        self.assertEqual([item.time_slot for item in result.time_slices], [0, 1, 2])

    def test_averages_match_time_slice_metrics(self):
        result = run_dynamic_mas_pso(
            self.tensor,
            [10, 11, 12],
            self.boundary,
            uav_count=2,
            pso_config=self.pso,
            seed=11,
        )

        expected_coverage = np.mean(
            [item.metrics.coverage_rate for item in result.time_slices]
        )
        expected_fitness = np.mean([item.metrics.fitness for item in result.time_slices])
        self.assertAlmostEqual(result.average_coverage_rate, expected_coverage)
        self.assertAlmostEqual(result.average_fitness, expected_fitness)

    def test_dynamic_result_is_deterministic_and_inside_boundary(self):
        first = run_dynamic_mas_pso(
            self.tensor,
            [0, 1, 2],
            self.boundary,
            uav_count=2,
            pso_config=self.pso,
            seed=15,
        )
        second = run_dynamic_mas_pso(
            self.tensor,
            [0, 1, 2],
            self.boundary,
            uav_count=2,
            pso_config=self.pso,
            seed=15,
        )

        for left, right in zip(first.time_slices, second.time_slices):
            np.testing.assert_allclose(left.positions, right.positions)
            self.assertEqual(left.metrics.boundary_penalty, 0.0)
            self.assertAlmostEqual(left.metrics.fitness, right.metrics.fitness)

    def test_time_slot_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            run_dynamic_mas_pso(
                self.tensor,
                [0, 1],
                self.boundary,
                uav_count=2,
                pso_config=self.pso,
            )


if __name__ == "__main__":
    unittest.main()
