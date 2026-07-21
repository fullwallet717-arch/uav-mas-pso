"""Random UAV deployment baseline."""

from __future__ import annotations

import numpy as np

from data_preprocessing import Boundary
from deployment_results import (
    DeploymentRunResult,
    TimeSliceResult,
    align_uav_positions,
    build_deployment_result,
    validate_positions_tensor,
)
from mas_coordination import MASConfig, evaluate_deployment


def random_uav_positions(
    rng: np.random.Generator,
    boundary: Boundary,
    uav_count: int,
) -> np.ndarray:
    if uav_count <= 0:
        raise ValueError("uav_count must be positive.")
    positions = np.empty((uav_count, 2), dtype=float)
    positions[:, 0] = rng.uniform(boundary.xmin, boundary.xmax, size=uav_count)
    positions[:, 1] = rng.uniform(boundary.ymin, boundary.ymax, size=uav_count)
    return positions


def run_random_deployment(
    positions_tensor: np.ndarray,
    time_slots: list[int] | tuple[int, ...],
    boundary: Boundary,
    uav_count: int = 5,
    mas_config: MASConfig = MASConfig(),
    seed: int = 42,
) -> DeploymentRunResult:
    """Sample an independent random deployment for every time slice."""
    users_by_time = validate_positions_tensor(positions_tensor, time_slots)
    rng = np.random.default_rng(seed)
    previous_positions: np.ndarray | None = None
    results: list[TimeSliceResult] = []

    for index, time_slot in enumerate(time_slots):
        positions = random_uav_positions(rng, boundary, uav_count)
        positions = align_uav_positions(previous_positions, positions)
        metrics = evaluate_deployment(
            users_by_time[index],
            positions,
            previous_positions=previous_positions,
            boundary=boundary,
            config=mas_config,
        )
        results.append(
            TimeSliceResult(
                time_slot=int(time_slot),
                positions=positions,
                metrics=metrics,
            )
        )
        previous_positions = positions.copy()

    return build_deployment_result("Random_Deployment", results)
