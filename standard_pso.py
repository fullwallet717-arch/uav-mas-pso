"""Standard PSO optimizer for a single UAV deployment time slice."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data_preprocessing import Boundary
from mas_coordination import DeploymentMetrics, MASConfig, evaluate_deployment


@dataclass(frozen=True)
class PSOConfig:
    particles: int = 20
    iterations: int = 20
    inertia: float = 0.70
    c1: float = 1.50
    c2: float = 1.50
    initial_velocity_std: float = 2.0


@dataclass(frozen=True)
class PSOResult:
    positions: np.ndarray
    metrics: DeploymentMetrics
    convergence_history: tuple[float, ...]
    initial_best_fitness: float


def _validate_inputs(
    user_positions: np.ndarray,
    boundary: Boundary,
    uav_count: int,
    config: PSOConfig,
) -> np.ndarray:
    users = np.asarray(user_positions, dtype=float)
    if users.ndim != 2 or users.shape[1] != 2:
        raise ValueError("user_positions must have shape (n, 2).")
    if not np.isfinite(users).all():
        raise ValueError("user_positions must contain only finite values.")
    if boundary.xmin >= boundary.xmax or boundary.ymin >= boundary.ymax:
        raise ValueError("boundary minimums must be smaller than maximums.")
    if uav_count <= 0:
        raise ValueError("uav_count must be positive.")
    if config.particles <= 0 or config.iterations < 0:
        raise ValueError("particles must be positive and iterations cannot be negative.")
    if config.inertia < 0 or config.c1 < 0 or config.c2 < 0:
        raise ValueError("inertia, c1, and c2 cannot be negative.")
    if config.initial_velocity_std < 0:
        raise ValueError("initial_velocity_std cannot be negative.")
    return users


def initialize_swarm(
    rng: np.random.Generator,
    boundary: Boundary,
    uav_count: int,
    particle_count: int,
) -> np.ndarray:
    """Create joint UAV deployment particles with shape (particles, UAVs, 2)."""
    swarm = np.empty((particle_count, uav_count, 2), dtype=float)
    swarm[:, :, 0] = rng.uniform(
        boundary.xmin, boundary.xmax, size=(particle_count, uav_count)
    )
    swarm[:, :, 1] = rng.uniform(
        boundary.ymin, boundary.ymax, size=(particle_count, uav_count)
    )
    return swarm


def _clip_swarm(swarm: np.ndarray, boundary: Boundary) -> None:
    swarm[:, :, 0] = np.clip(swarm[:, :, 0], boundary.xmin, boundary.xmax)
    swarm[:, :, 1] = np.clip(swarm[:, :, 1], boundary.ymin, boundary.ymax)


def run_standard_pso(
    user_positions: np.ndarray,
    boundary: Boundary,
    uav_count: int = 5,
    previous_positions: np.ndarray | None = None,
    pso_config: PSOConfig = PSOConfig(),
    mas_config: MASConfig = MASConfig(),
    seed: int = 42,
) -> PSOResult:
    """Optimize one time slice without warm-start or inter-slice state reuse."""
    users = _validate_inputs(user_positions, boundary, uav_count, pso_config)
    rng = np.random.default_rng(seed)
    swarm = initialize_swarm(rng, boundary, uav_count, pso_config.particles)
    velocities = rng.normal(
        0.0,
        pso_config.initial_velocity_std,
        size=swarm.shape,
    )

    personal_best = swarm.copy()
    personal_best_scores = np.full(pso_config.particles, -np.inf)
    global_best = swarm[0].copy()
    global_best_metrics: DeploymentMetrics | None = None

    def evaluate_swarm() -> None:
        nonlocal global_best, global_best_metrics
        for index, positions in enumerate(swarm):
            metrics = evaluate_deployment(
                users,
                positions,
                previous_positions=previous_positions,
                boundary=boundary,
                config=mas_config,
            )
            if metrics.fitness > personal_best_scores[index]:
                personal_best_scores[index] = metrics.fitness
                personal_best[index] = positions.copy()
            if global_best_metrics is None or metrics.fitness > global_best_metrics.fitness:
                global_best = positions.copy()
                global_best_metrics = metrics

    evaluate_swarm()
    assert global_best_metrics is not None
    initial_best_fitness = global_best_metrics.fitness
    convergence_history = [initial_best_fitness]

    for _ in range(pso_config.iterations):
        random_cognitive = rng.random(swarm.shape)
        random_social = rng.random(swarm.shape)
        velocities = (
            pso_config.inertia * velocities
            + pso_config.c1
            * random_cognitive
            * (personal_best - swarm)
            + pso_config.c2
            * random_social
            * (global_best[None, :, :] - swarm)
        )
        swarm += velocities
        _clip_swarm(swarm, boundary)
        evaluate_swarm()
        convergence_history.append(global_best_metrics.fitness)

    return PSOResult(
        positions=global_best.copy(),
        metrics=global_best_metrics,
        convergence_history=tuple(convergence_history),
        initial_best_fitness=float(initial_best_fitness),
    )
