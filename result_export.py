"""Export experiment results as analysis-ready CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from deployment_results import DeploymentRunResult

if TYPE_CHECKING:
    from experiment_runner import ExperimentResult


def _movement_distances(
    previous_positions: np.ndarray | None,
    current_positions: np.ndarray,
) -> np.ndarray:
    if previous_positions is None:
        return np.zeros(len(current_positions), dtype=float)
    return np.linalg.norm(current_positions - previous_positions, axis=1)


def _minimum_uav_distance(positions: np.ndarray) -> float:
    if len(positions) <= 1:
        return 0.0
    distances = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=2
    )
    return float(distances[np.triu_indices(len(positions), k=1)].min())


def build_position_frame(result: DeploymentRunResult) -> pd.DataFrame:
    rows = []
    previous: np.ndarray | None = None
    for time_slice in result.time_slices:
        movement = _movement_distances(previous, time_slice.positions)
        for uav_index, (position, distance) in enumerate(
            zip(time_slice.positions, movement), start=1
        ):
            rows.append(
                {
                    "algorithm": result.algorithm,
                    "time_slot": time_slice.time_slot,
                    "uav_id": uav_index,
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "move_from_previous": float(distance),
                }
            )
        previous = time_slice.positions
    return pd.DataFrame(rows)


def build_time_slice_frame(result: DeploymentRunResult) -> pd.DataFrame:
    rows = []
    previous: np.ndarray | None = None
    for time_slice in result.time_slices:
        metrics = time_slice.metrics
        movement = _movement_distances(previous, time_slice.positions)
        rows.append(
            {
                "algorithm": result.algorithm,
                "time_slot": time_slice.time_slot,
                "coverage_rate": metrics.coverage_rate,
                "covered_users": metrics.covered_users,
                "fitness": metrics.fitness,
                "density_reward": metrics.density_reward,
                "boundary_penalty": metrics.boundary_penalty,
                "safety_penalty": metrics.safety_penalty,
                "movement_penalty": metrics.movement_penalty,
                "overlap_metric": metrics.overlap_penalty,
                "safety_distance_violations": metrics.safety_violations,
                "movement_distance_violations": metrics.movement_violations,
                "communicating_pairs": metrics.communicating_pairs,
                "mean_uav_movement": float(movement.mean()),
                "max_uav_movement": float(movement.max(initial=0.0)),
                "min_inter_uav_distance": _minimum_uav_distance(
                    time_slice.positions
                ),
                "runtime_seconds": time_slice.runtime_seconds,
                "warm_started": time_slice.warm_started,
                "initial_best_fitness": time_slice.initial_best_fitness,
            }
        )
        previous = time_slice.positions
    return pd.DataFrame(rows)


def build_comparison_frame(experiment: ExperimentResult) -> pd.DataFrame:
    rows = []
    for result in experiment.algorithms:
        summary = build_time_slice_frame(result)
        rows.append(
            {
                "algorithm": result.algorithm,
                "average_coverage_rate": result.average_coverage_rate,
                "minimum_coverage_rate": float(summary["coverage_rate"].min()),
                "maximum_coverage_rate": float(summary["coverage_rate"].max()),
                "average_fitness": result.average_fitness,
                "average_uav_movement": float(
                    summary["mean_uav_movement"].mean()
                ),
                "safety_distance_violations_total": int(
                    summary["safety_distance_violations"].sum()
                ),
                "movement_distance_violations_total": int(
                    summary["movement_distance_violations"].sum()
                ),
                "average_overlap_metric": float(summary["overlap_metric"].mean()),
                "total_runtime_seconds": result.total_runtime_seconds,
                "average_runtime_seconds": result.average_runtime_seconds,
                "time_slots": len(result.time_slices),
                "uav_count": len(result.time_slices[0].positions),
            }
        )
    comparison = pd.DataFrame(rows).sort_values(
        "average_coverage_rate", ascending=False
    )
    comparison["coverage_rank"] = np.arange(1, len(comparison) + 1)
    speed_order = comparison.sort_values("average_runtime_seconds").copy()
    speed_order["speed_rank"] = np.arange(1, len(speed_order) + 1)
    return comparison.merge(
        speed_order[["algorithm", "speed_rank"]], on="algorithm", how="left"
    )


def export_experiment_results(
    experiment: ExperimentResult,
    output_dir: Path,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}

    for result in experiment.algorithms:
        positions_path = output / f"{result.algorithm}_uav_positions.csv"
        summary_path = output / f"{result.algorithm}_time_slot_summary.csv"
        build_position_frame(result).to_csv(
            positions_path, index=False, encoding="utf-8-sig"
        )
        build_time_slice_frame(result).to_csv(
            summary_path, index=False, encoding="utf-8-sig"
        )
        exported[f"{result.algorithm}_positions"] = positions_path
        exported[f"{result.algorithm}_summary"] = summary_path

    comparison_path = output / "algorithm_comparison.csv"
    build_comparison_frame(experiment).to_csv(
        comparison_path, index=False, encoding="utf-8-sig"
    )
    exported["algorithm_comparison"] = comparison_path
    return exported
