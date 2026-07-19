"""Shared-state MAS coordination metrics for UAV deployment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data_preprocessing import Boundary


@dataclass(frozen=True)
class MASConfig:
    coverage_radius: float = 25.0
    communication_range: float = 60.0
    min_safe_distance: float = 12.0
    max_move_distance: float = 35.0
    density_weight: float = 0.05
    safety_weight: float = 0.20
    movement_weight: float = 0.15
    overlap_weight: float = 0.05


@dataclass(frozen=True)
class SharedUAVState:
    positions: np.ndarray
    pairwise_distances: np.ndarray
    communication_adjacency: np.ndarray


@dataclass(frozen=True)
class DeploymentMetrics:
    fitness: float
    coverage_rate: float
    covered_users: int
    density_reward: float
    boundary_penalty: float
    safety_penalty: float
    movement_penalty: float
    overlap_penalty: float
    safety_violations: int
    movement_violations: int
    communicating_pairs: int


def _points(values: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2).")
    return points


def pairwise_distances(uav_positions: np.ndarray) -> np.ndarray:
    positions = _points(uav_positions, "uav_positions")
    return np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)


def share_uav_state(
    uav_positions: np.ndarray,
    communication_range: float,
) -> SharedUAVState:
    positions = _points(uav_positions, "uav_positions")
    distances = pairwise_distances(positions)
    adjacency = distances <= communication_range
    np.fill_diagonal(adjacency, False)
    return SharedUAVState(
        positions=positions.copy(),
        pairwise_distances=distances,
        communication_adjacency=adjacency,
    )


def covered_user_mask(
    user_positions: np.ndarray,
    uav_positions: np.ndarray,
    radius: float,
) -> np.ndarray:
    users = _points(user_positions, "user_positions")
    uavs = _points(uav_positions, "uav_positions")
    if len(users) == 0:
        return np.zeros(0, dtype=bool)
    if len(uavs) == 0:
        return np.zeros(len(users), dtype=bool)
    distances = np.linalg.norm(users[:, None, :] - uavs[None, :, :], axis=2)
    return (distances <= radius).any(axis=1)


def coverage_metrics(
    user_positions: np.ndarray,
    uav_positions: np.ndarray,
    radius: float,
) -> tuple[int, float]:
    users = _points(user_positions, "user_positions")
    covered = int(covered_user_mask(users, uav_positions, radius).sum())
    rate = covered / len(users) if len(users) else 0.0
    return covered, float(rate)


def density_reward(
    user_positions: np.ndarray,
    uav_positions: np.ndarray,
    radius: float,
) -> float:
    users = _points(user_positions, "user_positions")
    uavs = _points(uav_positions, "uav_positions")
    if len(users) == 0 or len(uavs) == 0:
        return 0.0
    distances = np.linalg.norm(users[:, None, :] - uavs[None, :, :], axis=2)
    normalized = np.minimum(distances.min(axis=1) / max(radius, 1e-9), 1.0)
    return float(1.0 - normalized.mean())


def safety_metrics(
    state: SharedUAVState,
    min_safe_distance: float,
) -> tuple[float, int]:
    uav_count = len(state.positions)
    if uav_count <= 1:
        return 0.0, 0
    distances = state.pairwise_distances[np.triu_indices(uav_count, k=1)]
    shortfall = np.maximum(0.0, min_safe_distance - distances)
    penalty = float((shortfall / max(min_safe_distance, 1e-9)).mean())
    return penalty, int((distances < min_safe_distance).sum())


def movement_metrics(
    current_positions: np.ndarray,
    previous_positions: np.ndarray | None,
    max_move_distance: float,
) -> tuple[float, int]:
    current = _points(current_positions, "current_positions")
    if previous_positions is None:
        return 0.0, 0
    previous = _points(previous_positions, "previous_positions")
    if previous.shape != current.shape:
        raise ValueError("previous_positions must match current_positions.")
    movement = np.linalg.norm(current - previous, axis=1)
    excess = np.maximum(0.0, movement - max_move_distance)
    penalty = float(excess.mean() / max(max_move_distance, 1e-9))
    return penalty, int((movement > max_move_distance).sum())


def overlap_metrics(
    state: SharedUAVState,
    coverage_radius: float,
) -> tuple[float, int]:
    upper = np.triu(state.communication_adjacency, k=1)
    pair_indexes = np.argwhere(upper)
    pair_count = len(pair_indexes)
    if pair_count == 0:
        return 0.0, 0

    overlap = 0.0
    diameter = max(2.0 * coverage_radius, 1e-9)
    for first, second in pair_indexes:
        distance = state.pairwise_distances[first, second]
        overlap += max(0.0, (diameter - distance) / diameter)
    return float(overlap / pair_count), pair_count


def boundary_penalty(uav_positions: np.ndarray, boundary: Boundary | None) -> float:
    if boundary is None:
        return 0.0
    positions = _points(uav_positions, "uav_positions")
    inside = (
        (positions[:, 0] >= boundary.xmin)
        & (positions[:, 0] <= boundary.xmax)
        & (positions[:, 1] >= boundary.ymin)
        & (positions[:, 1] <= boundary.ymax)
    )
    return 0.0 if inside.all() else 1.0


def evaluate_deployment(
    user_positions: np.ndarray,
    uav_positions: np.ndarray,
    previous_positions: np.ndarray | None = None,
    boundary: Boundary | None = None,
    config: MASConfig = MASConfig(),
) -> DeploymentMetrics:
    state = share_uav_state(uav_positions, config.communication_range)
    covered, coverage = coverage_metrics(
        user_positions, state.positions, config.coverage_radius
    )
    density = density_reward(user_positions, state.positions, config.coverage_radius)
    safety, safety_violations = safety_metrics(state, config.min_safe_distance)
    movement, movement_violations = movement_metrics(
        state.positions, previous_positions, config.max_move_distance
    )
    overlap, communicating_pairs = overlap_metrics(state, config.coverage_radius)
    boundary_cost = boundary_penalty(state.positions, boundary)

    fitness = (
        coverage
        + config.density_weight * density
        - config.safety_weight * safety
        - config.movement_weight * movement
        - config.overlap_weight * overlap
        - boundary_cost
    )
    return DeploymentMetrics(
        fitness=float(fitness),
        coverage_rate=coverage,
        covered_users=covered,
        density_reward=density,
        boundary_penalty=boundary_cost,
        safety_penalty=safety,
        movement_penalty=movement,
        overlap_penalty=overlap,
        safety_violations=safety_violations,
        movement_violations=movement_violations,
        communicating_pairs=communicating_pairs,
    )
