"""Static PSO UAV deployment baseline."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from data_preprocessing import Boundary
from deployment_results import (
    DeploymentRunResult,
    TimeSliceResult,
    build_deployment_result,
    validate_positions_tensor,
)
from mas_coordination import MASConfig, evaluate_deployment
from standard_pso import PSOConfig, run_standard_pso


def run_static_pso(
    positions_tensor: np.ndarray,
    time_slots: list[int] | tuple[int, ...],
    boundary: Boundary,
    uav_count: int = 5,
    pso_config: PSOConfig = PSOConfig(),
    mas_config: MASConfig = MASConfig(),
    seed: int = 42,
) -> DeploymentRunResult:
    """Optimize once for the aggregate population and keep all UAVs fixed."""
    users_by_time = validate_positions_tensor(positions_tensor, time_slots)
    aggregate_users = users_by_time.reshape(-1, 2)
    optimization_start = perf_counter()
    optimized = run_standard_pso(
        aggregate_users,
        boundary,
        uav_count=uav_count,
        pso_config=pso_config,
        mas_config=mas_config,
        seed=seed,
    )
    optimization_runtime = perf_counter() - optimization_start

    static_positions = optimized.positions.copy()
    previous_positions: np.ndarray | None = None
    results: list[TimeSliceResult] = []
    for index, time_slot in enumerate(time_slots):
        slot_start = perf_counter()
        metrics = evaluate_deployment(
            users_by_time[index],
            static_positions,
            previous_positions=previous_positions,
            boundary=boundary,
            config=mas_config,
        )
        results.append(
            TimeSliceResult(
                time_slot=int(time_slot),
                positions=static_positions.copy(),
                metrics=metrics,
                convergence_history=(
                    optimized.convergence_history if index == 0 else ()
                ),
                initial_best_fitness=(
                    optimized.initial_best_fitness if index == 0 else None
                ),
                runtime_seconds=(
                    perf_counter() - slot_start
                    + (optimization_runtime if index == 0 else 0.0)
                ),
            )
        )
        previous_positions = static_positions

    return build_deployment_result("Static_PSO", results)
