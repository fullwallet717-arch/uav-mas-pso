import unittest

import numpy as np

from data_preprocessing import Boundary
from mas_coordination import (
    MASConfig,
    coverage_metrics,
    density_reward,
    evaluate_deployment,
    movement_metrics,
    overlap_metrics,
    safety_metrics,
    share_uav_state,
)


class MASCoordinationTests(unittest.TestCase):
    def test_default_weights_match_v080_configuration(self):
        config = MASConfig()

        self.assertEqual(config.density_weight, 0.03)
        self.assertEqual(config.safety_weight, 0.18)
        self.assertEqual(config.movement_weight, 0.12)
        self.assertEqual(config.overlap_weight, 0.04)

    def test_shared_state_builds_distance_based_adjacency(self):
        positions = np.array([[0.0, 0.0], [30.0, 0.0], [100.0, 0.0]])
        state = share_uav_state(positions, communication_range=60.0)

        expected = np.array(
            [[False, True, False], [True, False, False], [False, False, False]]
        )
        np.testing.assert_array_equal(state.communication_adjacency, expected)
        self.assertAlmostEqual(state.pairwise_distances[0, 1], 30.0)

    def test_coverage_and_density_reward(self):
        users = np.array([[0.0, 0.0], [10.0, 0.0], [40.0, 0.0]])
        uavs = np.array([[0.0, 0.0]])

        covered, rate = coverage_metrics(users, uavs, radius=25.0)
        reward = density_reward(users, uavs, radius=25.0)

        self.assertEqual(covered, 2)
        self.assertAlmostEqual(rate, 2.0 / 3.0)
        self.assertAlmostEqual(reward, (1.0 + 0.6 + 0.0) / 3.0)

    def test_constraint_and_overlap_metrics(self):
        current = np.array([[0.0, 0.0], [6.0, 0.0]])
        previous = np.array([[0.0, 0.0], [0.0, 0.0]])
        state = share_uav_state(current, communication_range=60.0)

        safety, safety_violations = safety_metrics(state, min_safe_distance=12.0)
        movement, movement_violations = movement_metrics(
            current, previous, max_move_distance=5.0
        )
        overlap, pairs = overlap_metrics(state, coverage_radius=25.0)

        self.assertAlmostEqual(safety, 0.5)
        self.assertEqual(safety_violations, 1)
        self.assertAlmostEqual(movement, 0.1)
        self.assertEqual(movement_violations, 1)
        self.assertAlmostEqual(overlap, 44.0 / 50.0)
        self.assertEqual(pairs, 1)

    def test_evaluate_deployment_matches_weighted_fitness(self):
        users = np.array([[10.0, 10.0], [30.0, 10.0]])
        uavs = np.array([[10.0, 10.0], [30.0, 10.0]])
        boundary = Boundary(0.0, 40.0, 0.0, 40.0)
        config = MASConfig()

        metrics = evaluate_deployment(users, uavs, boundary=boundary, config=config)
        expected = (
            metrics.coverage_rate
            + config.density_weight * metrics.density_reward
            - config.safety_weight * metrics.safety_penalty
            - config.movement_weight * metrics.movement_penalty
            - config.overlap_weight * metrics.overlap_penalty
            - metrics.boundary_penalty
        )

        self.assertAlmostEqual(metrics.fitness, expected)
        self.assertEqual(metrics.covered_users, 2)
        self.assertEqual(metrics.boundary_penalty, 0.0)


if __name__ == "__main__":
    unittest.main()
