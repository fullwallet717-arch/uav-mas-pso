import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_preprocessing import Boundary
from experiment_runner import (
    ALGORITHM_ORDER,
    ExperimentConfig,
    PreparedExperimentData,
    prepare_experiment_data,
    run_algorithm_suite,
)
from kmeans_deployment import KMeansConfig
from standard_pso import PSOConfig


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self):
        first = np.array(
            [[18.0, 18.0], [20.0, 20.0], [22.0, 19.0], [80.0, 80.0]]
        )
        second = first + np.array([3.0, 2.0])
        self.data = PreparedExperimentData(
            boundary=Boundary(0.0, 100.0, 0.0, 100.0),
            time_slots=(0, 1),
            user_ids=np.arange(4),
            positions_tensor=np.stack([first, second]),
            source_observations=8,
            selected_observations=8,
        )
        self.config = ExperimentConfig(
            t_start=0,
            t_end=1,
            max_users=10,
            uav_count=2,
            seed=7,
            pso=PSOConfig(particles=8, iterations=4),
            kmeans=KMeansConfig(iterations=10),
        )

    def test_default_data_filter_parameters_are_explicit(self):
        config = ExperimentConfig()

        self.assertEqual(config.boundary_quantile, 0.05)
        self.assertEqual(config.min_observation_ratio, 0.20)

    def test_suite_runs_all_algorithms_in_fixed_order(self):
        result = run_algorithm_suite(self.data, self.config)

        self.assertEqual(
            tuple(item.algorithm for item in result.algorithms), ALGORITHM_ORDER
        )
        for algorithm in result.algorithms:
            self.assertEqual(len(algorithm.time_slices), 2)
            self.assertEqual(
                tuple(item.time_slot for item in algorithm.time_slices), (0, 1)
            )
            self.assertGreaterEqual(algorithm.average_coverage_rate, 0.0)
            self.assertLessEqual(algorithm.average_coverage_rate, 1.0)

    def test_suite_is_deterministic(self):
        first = run_algorithm_suite(self.data, self.config)
        second = run_algorithm_suite(self.data, self.config)

        for left, right in zip(first.algorithms, second.algorithms):
            self.assertAlmostEqual(
                left.average_coverage_rate, right.average_coverage_rate
            )
            for left_slot, right_slot in zip(left.time_slices, right.time_slices):
                np.testing.assert_allclose(left_slot.positions, right_slot.positions)

    def test_algorithm_lookup(self):
        result = run_algorithm_suite(self.data, self.config)

        self.assertEqual(result.get_algorithm("MAS_PSO").algorithm, "MAS_PSO")
        with self.assertRaises(KeyError):
            result.get_algorithm("missing")

    def test_invalid_time_range_is_rejected(self):
        invalid = ExperimentConfig(t_start=3, t_end=2)

        with self.assertRaises(ValueError):
            run_algorithm_suite(self.data, invalid)

    def test_prepared_time_slots_must_match_config(self):
        mismatched = ExperimentConfig(
            t_start=0,
            t_end=2,
            uav_count=2,
            pso=PSOConfig(particles=8, iterations=4),
        )

        with self.assertRaises(ValueError):
            run_algorithm_suite(self.data, mismatched)

    def test_prepare_data_uses_the_shared_filtering_pipeline(self):
        rows = []
        for uid, base in enumerate([10.0, 20.0, 30.0, 40.0, 50.0]):
            rows.append({"uid": uid, "d": 0, "t": 0, "x": base, "y": base})
            rows.append(
                {"uid": uid, "d": 0, "t": 1, "x": base + 1.0, "y": base + 0.5}
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectories.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            prepared = prepare_experiment_data(path, self.config)

        self.assertEqual(prepared.time_slots, (0, 1))
        self.assertEqual(prepared.source_observations, 10)
        self.assertEqual(len(prepared.user_ids), 3)
        self.assertEqual(prepared.positions_tensor.shape, (2, 3, 2))


if __name__ == "__main__":
    unittest.main()
