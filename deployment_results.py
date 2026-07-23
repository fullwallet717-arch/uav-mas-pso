"""Shared result structures and helpers for multi-time-slice deployments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from mas_coordination import DeploymentMetrics


@dataclass(frozen=True)
class TimeSliceResult:
    time_slot: int
    positions: np.ndarray
    metrics: DeploymentMetrics
    convergence_history: tuple[float, ...] = ()
    initial_best_fitness: float | None = None
    warm_started: bool = False
    runtime_seconds: float = 0.0


@dataclass(frozen=True)
class DeploymentRunResult:
    algorithm: str
    time_slices: tuple[TimeSliceResult, ...]
    average_coverage_rate: float
    average_fitness: float
    total_runtime_seconds: float
    average_runtime_seconds: float


def validate_positions_tensor(
    positions_tensor: np.ndarray,
    time_slots: list[int] | tuple[int, ...],
) -> np.ndarray:
    positions = np.asarray(positions_tensor, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError("positions_tensor must have shape (time, users, 2).")
    if positions.shape[0] == 0 or positions.shape[1] == 0:
        raise ValueError("positions_tensor must contain time slices and users.")
    if positions.shape[0] != len(time_slots):
        raise ValueError("time_slots length must match positions_tensor.")
    if not np.isfinite(positions).all():
        raise ValueError("positions_tensor must contain only finite values.")
    return positions


def align_uav_positions(
    previous_positions: np.ndarray | None,
    current_positions: np.ndarray,
) -> np.ndarray:
    """Assign current coordinates to UAV IDs with minimum total movement."""
    current = np.asarray(current_positions, dtype=float)
    if current.ndim != 2 or current.shape[1] != 2:
        raise ValueError("current_positions must have shape (UAVs, 2).")
    if previous_positions is None:
        return current.copy()

    previous = np.asarray(previous_positions, dtype=float)
    if previous.shape != current.shape:
        raise ValueError("previous_positions must match current_positions.")
    if len(current) > 8:
        raise ValueError("exact UAV alignment supports at most eight UAVs.")

    best_order: tuple[int, ...] | None = None
    best_distance = np.inf
    for order in permutations(range(len(current))):
        candidate = current[list(order)]
        total_distance = float(np.linalg.norm(candidate - previous, axis=1).sum())
        if total_distance < best_distance:
            best_distance = total_distance
            best_order = order
    assert best_order is not None
    return current[list(best_order)].copy()


def build_deployment_result(
    algorithm: str,
    time_slices: list[TimeSliceResult],
) -> DeploymentRunResult:
    if not time_slices:
        raise ValueError("time_slices cannot be empty.")
    coverage = [item.metrics.coverage_rate for item in time_slices]
    fitness = [item.metrics.fitness for item in time_slices]
    runtimes = [item.runtime_seconds for item in time_slices]
    if any(runtime < 0 for runtime in runtimes):
        raise ValueError("runtime_seconds cannot be negative.")
    return DeploymentRunResult(
        algorithm=algorithm,
        time_slices=tuple(time_slices),
        average_coverage_rate=float(np.mean(coverage)),
        average_fitness=float(np.mean(fitness)),
        total_runtime_seconds=float(np.sum(runtimes)),
        average_runtime_seconds=float(np.mean(runtimes)),
    )
