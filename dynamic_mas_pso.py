"""Dynamic MAS-PSO deployment with warm-start between time slices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data_preprocessing import Boundary
from deployment_results import (
    DeploymentRunResult,
    TimeSliceResult,
    build_deployment_result,
    validate_positions_tensor,
)
from mas_coordination import MASConfig
from standard_pso import PSOConfig, PSOResult, initialize_swarm, optimize_swarm


@dataclass(frozen=True)
class WarmStartConfig:
    noise_std: float = 5.0
    random_restart_ratio: float = 0.20


DynamicMASPSOResult = DeploymentRunResult


def _clip_swarm(swarm: np.ndarray, boundary: Boundary) -> None:
    swarm[:, :, 0] = np.clip(swarm[:, :, 0], boundary.xmin, boundary.xmax)
    swarm[:, :, 1] = np.clip(swarm[:, :, 1], boundary.ymin, boundary.ymax)


def initialize_warm_swarm(
    rng: np.random.Generator,
    previous_best: np.ndarray,
    boundary: Boundary,
    particle_count: int,
    config: WarmStartConfig = WarmStartConfig(),
) -> np.ndarray:
    """Keep the previous best, perturb nearby particles, and add random restarts."""
    previous = np.asarray(previous_best, dtype=float)
    if previous.ndim != 2 or previous.shape[1] != 2:
        raise ValueError("previous_best must have shape (UAVs, 2).")
    if not np.isfinite(previous).all():
        raise ValueError("previous_best must contain only finite values.")
    if particle_count <= 0:
        raise ValueError("particle_count must be positive.")
    if config.noise_std < 0:
        raise ValueError("noise_std cannot be negative.")
    if not 0.0 <= config.random_restart_ratio <= 1.0:
        raise ValueError("random_restart_ratio must be between 0 and 1.")

    swarm = previous[None, :, :] + rng.normal(
        0.0,
        config.noise_std,
        size=(particle_count, len(previous), 2),
    )
    swarm[0] = previous

    restart_count = min(
        particle_count - 1,
        int(round(particle_count * config.random_restart_ratio)),
    )
    if restart_count:
        swarm[-restart_count:] = initialize_swarm(
            rng,
            boundary,
            uav_count=len(previous),
            particle_count=restart_count,
        )
    _clip_swarm(swarm, boundary)
    return swarm


def run_dynamic_mas_pso(
    positions_tensor: np.ndarray,
    time_slots: list[int] | tuple[int, ...],
    boundary: Boundary,
    uav_count: int = 5,
    pso_config: PSOConfig = PSOConfig(),
    mas_config: MASConfig = MASConfig(),
    warm_start_config: WarmStartConfig = WarmStartConfig(),
    seed: int = 42,
) -> DynamicMASPSOResult:
    """Optimize consecutive time slices and reuse the preceding best deployment."""
    positions = validate_positions_tensor(positions_tensor, time_slots)
    rng = np.random.default_rng(seed)
    previous_best: np.ndarray | None = None
    slice_results: list[TimeSliceResult] = []

    for index, time_slot in enumerate(time_slots):
        warm_started = previous_best is not None
        if previous_best is None:
            initial_swarm = initialize_swarm(
                rng,
                boundary,
                uav_count=uav_count,
                particle_count=pso_config.particles,
            )
        else:
            initial_swarm = initialize_warm_swarm(
                rng,
                previous_best,
                boundary,
                particle_count=pso_config.particles,
                config=warm_start_config,
            )

        optimized: PSOResult = optimize_swarm(
            positions[index],
            boundary,
            initial_swarm,
            rng,
            previous_positions=previous_best,
            pso_config=pso_config,
            mas_config=mas_config,
        )
        slice_results.append(
            TimeSliceResult(
                time_slot=int(time_slot),
                positions=optimized.positions.copy(),
                metrics=optimized.metrics,
                convergence_history=optimized.convergence_history,
                initial_best_fitness=optimized.initial_best_fitness,
                warm_started=warm_started,
            )
        )
        previous_best = optimized.positions.copy()

    return build_deployment_result("MAS_PSO", slice_results)
