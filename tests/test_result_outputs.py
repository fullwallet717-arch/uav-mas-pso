import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_preprocessing import Boundary
from experiment_runner import ExperimentConfig, PreparedExperimentData, run_algorithm_suite
from result_export import (
    build_comparison_frame,
    build_time_slice_frame,
    export_experiment_results,
)
from result_visualization import save_all_figures, select_representative_time_slices
from standard_pso import PSOConfig


class ResultOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(17)
        positions = rng.uniform(15.0, 85.0, size=(4, 24, 2))
        positions[1:] = np.clip(
            positions[0] + rng.normal(0.0, 7.0, size=(3, 24, 2)),
            0.0,
            100.0,
        )
        cls.config = ExperimentConfig(
            t_start=0,
            t_end=3,
            max_users=24,
            uav_count=3,
            seed=23,
            pso=PSOConfig(particles=8, iterations=4),
        )
        data = PreparedExperimentData(
            boundary=Boundary(0.0, 100.0, 0.0, 100.0),
            time_slots=(0, 1, 2, 3),
            user_ids=np.arange(24),
            positions_tensor=positions,
            source_observations=96,
            selected_observations=96,
        )
        cls.experiment = run_algorithm_suite(data, cls.config)

    def test_runtime_aggregates_match_time_slice_values(self) -> None:
        for algorithm in self.experiment.algorithms:
            runtimes = np.asarray(
                [item.runtime_seconds for item in algorithm.time_slices],
                dtype=float,
            )
            self.assertTrue(np.all(runtimes >= 0.0))
            self.assertAlmostEqual(algorithm.total_runtime_seconds, runtimes.sum())
            self.assertAlmostEqual(
                algorithm.average_runtime_seconds,
                runtimes.mean(),
            )

    def test_exported_tables_contain_required_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            paths = export_experiment_results(self.experiment, output)
            self.assertEqual(len(paths), 11)
            self.assertTrue(all(path.exists() for path in paths.values()))

            summary = pd.read_csv(output / "MAS_PSO_time_slot_summary.csv")
            self.assertTrue(
                {
                    "runtime_seconds",
                    "mean_uav_movement",
                    "overlap_metric",
                    "safety_distance_violations",
                    "movement_distance_violations",
                }.issubset(summary.columns)
            )
            comparison = pd.read_csv(output / "algorithm_comparison.csv")
            self.assertTrue(
                {
                    "average_coverage_rate",
                    "average_runtime_seconds",
                    "speed_rank",
                    "average_overlap_metric",
                }.issubset(comparison.columns)
            )

    def test_representative_slots_follow_exact_mas_pso_extrema(self) -> None:
        selected = select_representative_time_slices(self.experiment)
        mas_summary = build_time_slice_frame(
            self.experiment.get_algorithm("MAS_PSO")
        )
        expected = [
            int(mas_summary.loc[mas_summary["coverage_rate"].idxmin(), "time_slot"]),
            int(
                mas_summary.loc[
                    mas_summary["mean_uav_movement"].idxmax(),
                    "time_slot",
                ]
            ),
            int(mas_summary.loc[mas_summary["coverage_rate"].idxmax(), "time_slot"]),
        ]
        self.assertEqual(
            selected["criterion"].tolist(),
            [
                "lowest_mas_coverage",
                "largest_uav_movement",
                "highest_mas_coverage",
            ],
        )
        self.assertEqual(selected["time_slot"].tolist(), expected)

    def test_all_figures_are_generated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = save_all_figures(self.experiment, Path(directory))
            self.assertEqual(len(paths), 5)
            self.assertTrue(
                all(
                    path.exists() and path.stat().st_size > 0
                    for path in paths.values()
                )
            )

    def test_comparison_has_one_row_per_algorithm(self) -> None:
        comparison = build_comparison_frame(self.experiment)
        self.assertEqual(
            set(comparison["algorithm"]),
            {result.algorithm for result in self.experiment.algorithms},
        )


if __name__ == "__main__":
    unittest.main()
