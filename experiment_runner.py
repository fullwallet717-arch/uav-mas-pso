"""Run all UAV deployment algorithms with one shared experiment setup."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from data_preprocessing import (
    BOUNDARY_QUANTILE,
    CSV_PATH,
    MAX_USERS,
    MIN_OBS_RATIO,
    Boundary,
    build_trajectory_tensor,
    determine_disaster_boundary,
    load_selected_data,
    select_stable_users,
)
from deployment_results import DeploymentRunResult, validate_positions_tensor
from dynamic_mas_pso import WarmStartConfig, run_dynamic_mas_pso
from kmeans_deployment import KMeansConfig, run_kmeans_deployment
from mas_coordination import MASConfig
from random_deployment import run_random_deployment
from standard_pso import PSOConfig, run_standard_pso_baseline
from static_pso import run_static_pso


ALGORITHM_ORDER = (
    "Random_Deployment",
    "KMeans",
    "Standard_PSO",
    "Static_PSO",
    "MAS_PSO",
)


@dataclass(frozen=True)
class ExperimentConfig:
    day: int = 0
    t_start: int = 0
    t_end: int = 23
    max_users: int = MAX_USERS
    uav_count: int = 5
    seed: int = 42
    boundary_quantile: float = BOUNDARY_QUANTILE
    min_observation_ratio: float = MIN_OBS_RATIO
    mas: MASConfig = field(default_factory=MASConfig)
    pso: PSOConfig = field(default_factory=PSOConfig)
    warm_start: WarmStartConfig = field(default_factory=WarmStartConfig)
    kmeans: KMeansConfig = field(default_factory=KMeansConfig)


@dataclass(frozen=True)
class PreparedExperimentData:
    boundary: Boundary
    time_slots: tuple[int, ...]
    user_ids: np.ndarray
    positions_tensor: np.ndarray
    source_observations: int
    selected_observations: int


@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    data: PreparedExperimentData
    algorithms: tuple[DeploymentRunResult, ...]

    def get_algorithm(self, name: str) -> DeploymentRunResult:
        for result in self.algorithms:
            if result.algorithm == name:
                return result
        raise KeyError(f"Unknown algorithm: {name}")


def _validate_config(config: ExperimentConfig) -> None:
    if config.t_start > config.t_end:
        raise ValueError("t_start cannot be greater than t_end.")
    if config.max_users <= 0:
        raise ValueError("max_users must be positive.")
    if config.uav_count <= 0:
        raise ValueError("uav_count must be positive.")
    if not 0.0 <= config.boundary_quantile < 0.5:
        raise ValueError("boundary_quantile must be in the range [0, 0.5).")
    if not 0.0 < config.min_observation_ratio <= 1.0:
        raise ValueError("min_observation_ratio must be in the range (0, 1].")


def prepare_experiment_data(
    csv_path: Path,
    config: ExperimentConfig = ExperimentConfig(),
) -> PreparedExperimentData:
    """Apply the shared filtering pipeline and build one trajectory tensor."""
    _validate_config(config)
    time_slots = list(range(config.t_start, config.t_end + 1))
    source = load_selected_data(Path(csv_path), config.day, time_slots)
    boundary = determine_disaster_boundary(source, config.boundary_quantile)
    selected, stable_users = select_stable_users(
        source,
        boundary,
        time_slots,
        config.max_users,
        config.min_observation_ratio,
    )
    user_ids, positions_tensor = build_trajectory_tensor(
        selected,
        stable_users,
        time_slots,
    )
    return PreparedExperimentData(
        boundary=boundary,
        time_slots=tuple(time_slots),
        user_ids=user_ids.copy(),
        positions_tensor=positions_tensor.copy(),
        source_observations=len(source),
        selected_observations=len(selected),
    )


def run_algorithm_suite(
    data: PreparedExperimentData,
    config: ExperimentConfig = ExperimentConfig(),
) -> ExperimentResult:
    """Run every algorithm with the same tensor, boundary, and seed."""
    _validate_config(config)
    positions = validate_positions_tensor(data.positions_tensor, data.time_slots)
    expected_slots = tuple(range(config.t_start, config.t_end + 1))
    if data.time_slots != expected_slots:
        raise ValueError("Prepared time slots do not match the experiment config.")
    if len(data.user_ids) != positions.shape[1]:
        raise ValueError("user_ids length must match the trajectory tensor.")
    common = {
        "positions_tensor": positions,
        "time_slots": data.time_slots,
        "boundary": data.boundary,
        "uav_count": config.uav_count,
        "mas_config": config.mas,
        "seed": config.seed,
    }

    algorithms = (
        run_random_deployment(**common),
        run_kmeans_deployment(**common, kmeans_config=config.kmeans),
        run_standard_pso_baseline(**common, pso_config=config.pso),
        run_static_pso(**common, pso_config=config.pso),
        run_dynamic_mas_pso(
            **common,
            pso_config=config.pso,
            warm_start_config=config.warm_start,
        ),
    )
    names = tuple(result.algorithm for result in algorithms)
    if names != ALGORITHM_ORDER:
        raise RuntimeError("Algorithm suite order does not match ALGORITHM_ORDER.")
    return ExperimentResult(config=config, data=data, algorithms=algorithms)


def run_experiment(
    csv_path: Path = CSV_PATH,
    config: ExperimentConfig = ExperimentConfig(),
) -> ExperimentResult:
    data = prepare_experiment_data(csv_path, config)
    return run_algorithm_suite(data, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the shared UAV experiment suite.")
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--day", type=int, default=0)
    parser.add_argument("--t-start", type=int, default=0)
    parser.add_argument("--t-end", type=int, default=23)
    parser.add_argument("--max-users", type=int, default=MAX_USERS)
    parser.add_argument("--uav-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boundary-quantile", type=float, default=BOUNDARY_QUANTILE)
    parser.add_argument("--min-observation-ratio", type=float, default=MIN_OBS_RATIO)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional directory for CSV results and thesis figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        day=args.day,
        t_start=args.t_start,
        t_end=args.t_end,
        max_users=args.max_users,
        uav_count=args.uav_count,
        seed=args.seed,
        boundary_quantile=args.boundary_quantile,
        min_observation_ratio=args.min_observation_ratio,
    )
    result = run_experiment(args.csv, config)

    print(f"Selected users: {len(result.data.user_ids)}")
    print(f"Time slots: {result.data.time_slots[0]}-{result.data.time_slots[-1]}")
    print(f"Boundary: {result.data.boundary}")
    print("Algorithm            Coverage    Fitness    Avg runtime (s)")
    for algorithm in result.algorithms:
        print(
            f"{algorithm.algorithm:<20} "
            f"{algorithm.average_coverage_rate:>8.4f}    "
            f"{algorithm.average_fitness:>7.4f}    "
            f"{algorithm.average_runtime_seconds:>15.6f}"
        )
    if args.output is not None:
        from result_export import export_experiment_results
        from result_visualization import save_all_figures

        export_experiment_results(result, args.output)
        save_all_figures(result, args.output)
        print(f"Results and figures: {args.output.resolve()}")


if __name__ == "__main__":
    main()
