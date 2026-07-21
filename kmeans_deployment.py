"""K-means UAV deployment baseline."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class KMeansConfig:
    iterations: int = 30
    tolerance: float = 1e-6


def fit_kmeans(
    user_positions: np.ndarray,
    cluster_count: int,
    boundary: Boundary,
    rng: np.random.Generator,
    config: KMeansConfig = KMeansConfig(),
) -> np.ndarray:
    points = np.asarray(user_positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("user_positions must have shape (n, 2) and cannot be empty.")
    if cluster_count <= 0:
        raise ValueError("cluster_count must be positive.")
    if config.iterations <= 0 or config.tolerance < 0:
        raise ValueError("iterations must be positive and tolerance cannot be negative.")

    unique_points = np.unique(points, axis=0)
    indexes = rng.choice(
        len(unique_points),
        size=cluster_count,
        replace=len(unique_points) < cluster_count,
    )
    centers = unique_points[indexes].astype(float)

    for _ in range(config.iterations):
        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        updated = centers.copy()
        for cluster in range(cluster_count):
            members = points[labels == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
            else:
                updated[cluster] = points[rng.integers(0, len(points))]
        updated[:, 0] = np.clip(updated[:, 0], boundary.xmin, boundary.xmax)
        updated[:, 1] = np.clip(updated[:, 1], boundary.ymin, boundary.ymax)
        if np.max(np.linalg.norm(updated - centers, axis=1)) <= config.tolerance:
            centers = updated
            break
        centers = updated
    return centers


def run_kmeans_deployment(
    positions_tensor: np.ndarray,
    time_slots: list[int] | tuple[int, ...],
    boundary: Boundary,
    uav_count: int = 5,
    kmeans_config: KMeansConfig = KMeansConfig(),
    mas_config: MASConfig = MASConfig(),
    seed: int = 42,
) -> DeploymentRunResult:
    """Fit new K-means centers to the population in every time slice."""
    users_by_time = validate_positions_tensor(positions_tensor, time_slots)
    rng = np.random.default_rng(seed)
    previous_positions: np.ndarray | None = None
    results: list[TimeSliceResult] = []

    for index, time_slot in enumerate(time_slots):
        positions = fit_kmeans(
            users_by_time[index], uav_count, boundary, rng, kmeans_config
        )
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

    return build_deployment_result("KMeans", results)
