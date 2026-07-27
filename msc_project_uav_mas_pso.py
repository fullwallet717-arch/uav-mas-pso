r"""
MSC Project: Dynamic MAS-PSO UAV Deployment

Run in PyCharm:
1. Open the project folder: D:\MSC Project Result\Project
2. Open this file: msc_project_uav_mas_pso.py
3. Select interpreter: .venv\Scripts\python.exe
4. Run this file. Results are written to D:\MSC Project Result\Result
"""

from __future__ import annotations

import argparse
from itertools import permutations
import time
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# Basic experiment settings

CSV_PATH = Path(r"D:\MSC Project Result\data\yjmob100k-dataset2.csv")
OUTPUT_DIR = Path(r"D:\MSC Project Result\Result")

DAY = 0
TIME_SLOTS = list(range(0, 24))  # 24 half-hour slots = 12 hours.
TIME_INTERVAL_MINUTES = 30

UID_COL = "uid"
DAY_COL = "d"
TIME_COL = "t"
X_COL = "x"
Y_COL = "y"

GRID_MIN = 1.0
GRID_MAX = 200.0
BOUNDARY_QUANTILE = 0.05
MIN_OBS_RATIO = 0.20

BASE_USER_COUNT = 400
BASE_UAV_COUNT = 5
BASE_COVERAGE_RADIUS = 25.0

RANDOM_SEED = 42
KMEANS_ITERATIONS = 40

ALGORITHMS = (
    "MAS_PSO",
    "Random_Deployment",
    "KMeans",
    "Standard_PSO",
    "Static_PSO",
)

FORMAL_OUTPUT_FILENAMES = (
    "selected_stable_users.csv",
    "selected_user_observations.csv",
    "processed_user_trajectories.csv",
    "boundary_and_selected_users.png",
    *(f"{name}_uav_positions.csv" for name in ALGORITHMS),
    *(f"{name}_time_slot_summary.csv" for name in ALGORITHMS),
    "algorithm_average_coverage_comparison.csv",
    "algorithm_average_coverage_comparison.png",
    "algorithm_runtime_comparison.png",
    "Figure_5_3_coverage_rate_by_time_slot.png",
    "Figure_5_4_mean_uav_movement_by_time_slot.png",
    "Figure_5_5_representative_time_slices.png",
    "Figure_5_5_selected_time_slots.csv",
    "Figure_5_6_constraint_violations_and_overlap.png",
    "scenario_results.csv",
    "A_Number_of_UAVs_line_chart.png",
    "B_Coverage_Radius_line_chart.png",
    "C_Number_of_Users_line_chart.png",
    "D_Disaster_Area_Size_line_chart.png",
)

OBSOLETE_OUTPUT_FILENAMES = (
    *(f"{name}_coverage_curve.png" for name in ALGORITHMS),
    "base_uav_positions.csv",
    "base_time_slot_summary.csv",
    "boundary_summary.json",
    "chosen_pso_parameters.json",
    "penalty_weight_grid_search.csv",
    "penalty_weight_grid_search.png",
    "pso_parameter_search.csv",
    "pso_parameter_search.png",
)



@dataclass
class Boundary:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return self.width * self.height

    def clip_points(self, points: np.ndarray) -> np.ndarray:
        clipped = points.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], self.xmin, self.xmax)
        clipped[:, 1] = np.clip(clipped[:, 1], self.ymin, self.ymax)
        return clipped


@dataclass
class PSOConfig:
    particles: int = 24
    iterations: int = 30
    inertia: float = 0.72
    c1: float = 1.45
    c2: float = 1.45
    warm_noise: float = 5.0
    min_safe_distance: float = 12.0
    max_move_distance: float = 35.0
    communication_range: float = 60.0
    # Fixed objective-function weights used by the main experiment.
    safe_penalty_weight: float = 0.18
    move_penalty_weight: float = 0.12
    overlap_penalty_weight: float = 0.04
    density_bonus_weight: float = 0.03


FORMAL_PSO_CONFIG = PSOConfig(
    particles=32,
    iterations=25,
    inertia=0.72,
    c1=1.45,
    c2=1.45,
)


def validate_formal_experiment() -> None:
    cfg = FORMAL_PSO_CONFIG
    if RANDOM_SEED < 0:
        raise ValueError("RANDOM_SEED cannot be negative.")
    if not TIME_SLOTS or TIME_SLOTS != list(range(0, 24)):
        raise ValueError("The formal experiment must use time slots 0-23.")
    if BASE_USER_COUNT <= 0 or BASE_UAV_COUNT <= 0:
        raise ValueError("User and UAV counts must be positive.")
    if BASE_COVERAGE_RADIUS <= 0.0:
        raise ValueError("Coverage radius must be positive.")
    if cfg.min_safe_distance <= 0.0 or cfg.max_move_distance <= 0.0:
        raise ValueError("Safety and movement limits must be positive.")
    if cfg.communication_range <= 0.0:
        raise ValueError("Communication range must be positive.")
    if cfg.particles <= 0 or cfg.iterations <= 0:
        raise ValueError("PSO particles and iterations must be positive.")
    if len(FORMAL_OUTPUT_FILENAMES) != len(set(FORMAL_OUTPUT_FILENAMES)):
        raise ValueError("Formal output filenames must be unique.")
    if set(FORMAL_OUTPUT_FILENAMES).intersection(OBSOLETE_OUTPUT_FILENAMES):
        raise ValueError("Obsolete files cannot appear in the formal output list.")


def remove_obsolete_outputs(output_dir: Path) -> tuple[Path, ...]:
    removed = []
    for filename in OBSOLETE_OUTPUT_FILENAMES:
        path = output_dir / filename
        if path.is_file():
            path.unlink()
            removed.append(path)
    return tuple(removed)


def verify_formal_outputs(output_dir: Path) -> None:
    missing = [
        filename
        for filename in FORMAL_OUTPUT_FILENAMES
        if not (output_dir / filename).is_file()
    ]
    obsolete = [
        filename
        for filename in OBSOLETE_OUTPUT_FILENAMES
        if (output_dir / filename).is_file()
    ]
    if missing or obsolete:
        raise RuntimeError(
            f"Formal output validation failed. Missing={missing}, obsolete={obsolete}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dynamic MAS-PSO UAV deployment experiment for YJMob100k data."
    )
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="Path to yjmob100k CSV.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--day", type=int, default=DAY, help="Selected dataset day.")
    parser.add_argument("--t-start", type=int, default=min(TIME_SLOTS), help="Start time slot.")
    parser.add_argument("--t-end", type=int, default=max(TIME_SLOTS), help="End time slot, included.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run an eight-slot smoke test; do not use these results in the paper.",
    )
    return parser.parse_args()


def load_selected_data(csv_path: Path, day: int, time_slots: list[int]) -> pd.DataFrame:
    usecols = [UID_COL, DAY_COL, TIME_COL, X_COL, Y_COL]
    df = pd.read_csv(csv_path, usecols=usecols)
    df = df[(df[DAY_COL] == day) & (df[TIME_COL].isin(time_slots))].copy()
    if df.empty:
        raise ValueError("No rows remain after filtering by day and time slots.")
    for col in [X_COL, Y_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[UID_COL, TIME_COL, X_COL, Y_COL])
    return df


def determine_disaster_boundary(df: pd.DataFrame, quantile: float = BOUNDARY_QUANTILE) -> Boundary:
    xmin = float(df[X_COL].quantile(quantile))
    xmax = float(df[X_COL].quantile(1 - quantile))
    ymin = float(df[Y_COL].quantile(quantile))
    ymax = float(df[Y_COL].quantile(1 - quantile))
    return Boundary(
        xmin=max(GRID_MIN, xmin),
        xmax=min(GRID_MAX, xmax),
        ymin=max(GRID_MIN, ymin),
        ymax=min(GRID_MAX, ymax),
    )


def centered_boundary(df: pd.DataFrame, size: float) -> Boundary:
    cx = float(df[X_COL].median())
    cy = float(df[Y_COL].median())
    half = size / 2.0
    xmin = max(GRID_MIN, cx - half)
    xmax = min(GRID_MAX, cx + half)
    ymin = max(GRID_MIN, cy - half)
    ymax = min(GRID_MAX, cy + half)
    return Boundary(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)


def select_stable_users(
    df: pd.DataFrame,
    boundary: Boundary,
    time_slots: list[int],
    max_users: int,
    min_obs_ratio: float = MIN_OBS_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inside = (
        df[X_COL].between(boundary.xmin, boundary.xmax)
        & df[Y_COL].between(boundary.ymin, boundary.ymax)
    )
    user_summary = (
        df.assign(in_boundary=inside)
        .groupby(UID_COL)
        .agg(
            observed_slots=(TIME_COL, "nunique"),
            all_observed_points_inside=("in_boundary", "all"),
            first_t=(TIME_COL, "min"),
            last_t=(TIME_COL, "max"),
        )
        .reset_index()
    )
    min_obs = max(1, int(np.ceil(len(time_slots) * min_obs_ratio)))
    stable_users = user_summary[
        (user_summary["observed_slots"] >= min_obs)
        & (user_summary["all_observed_points_inside"])
    ].copy()
    stable_users = stable_users.sort_values(
        ["observed_slots", UID_COL], ascending=[False, True]
    ).head(max_users)
    stable_users["required_time_slots"] = len(time_slots)
    stable_users["observed_ratio"] = stable_users["observed_slots"] / max(len(time_slots), 1)
    stable_users["imputed_slots"] = len(time_slots) - stable_users["observed_slots"]
    selected = df[df[UID_COL].isin(stable_users[UID_COL])].copy()
    selected = selected[
        selected[X_COL].between(boundary.xmin, boundary.xmax)
        & selected[Y_COL].between(boundary.ymin, boundary.ymax)
    ]
    if selected[UID_COL].nunique() == 0:
        raise ValueError("No stable users found. Try a larger boundary or lower MIN_OBS_RATIO.")
    return selected, stable_users


def build_trajectory_tensor(
    selected: pd.DataFrame,
    stable_users: pd.DataFrame,
    time_slots: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    user_ids = stable_users[UID_COL].to_numpy()
    selected = selected.sort_values([UID_COL, TIME_COL])
    selected = selected.drop_duplicates([UID_COL, TIME_COL], keep="last")

    full_index = pd.MultiIndex.from_product([user_ids, time_slots], names=[UID_COL, TIME_COL])
    xy = selected.set_index([UID_COL, TIME_COL])[[X_COL, Y_COL]].reindex(full_index)
    xy = xy.groupby(level=0, group_keys=False).ffill()
    xy = xy.groupby(level=0, group_keys=False).bfill()
    xy = xy.dropna()

    positions_by_time = []
    valid_users = xy.index.get_level_values(UID_COL).unique().to_numpy()
    for t in time_slots:
        frame = xy.xs(t, level=TIME_COL).loc[valid_users, [X_COL, Y_COL]]
        positions_by_time.append(frame.to_numpy(dtype=float))
    return valid_users, np.stack(positions_by_time, axis=0)


def build_processed_trajectory_frame(
    selected: pd.DataFrame,
    user_ids: np.ndarray,
    tensor: np.ndarray,
    time_slots: list[int],
) -> pd.DataFrame:
    """Create an auditable model-input table that distinguishes observed and imputed positions."""
    user_major_positions = np.transpose(tensor, (1, 0, 2)).reshape(-1, 2)
    frame = pd.DataFrame(
        {
            UID_COL: np.repeat(user_ids, len(time_slots)),
            TIME_COL: np.tile(time_slots, len(user_ids)),
            X_COL: user_major_positions[:, 0],
            Y_COL: user_major_positions[:, 1],
        }
    )
    observed_index = pd.MultiIndex.from_frame(
        selected.drop_duplicates([UID_COL, TIME_COL], keep="last")[[UID_COL, TIME_COL]]
    )
    frame_index = pd.MultiIndex.from_frame(frame[[UID_COL, TIME_COL]])
    frame["is_observed"] = frame_index.isin(observed_index)
    frame["is_imputed"] = ~frame["is_observed"]
    frame["position_source"] = np.where(
        frame["is_observed"], "observed", "forward_or_backward_filled"
    )
    return frame


def covered_user_count(user_positions: np.ndarray, uav_positions: np.ndarray, radius: float) -> int:
    if len(user_positions) == 0:
        return 0
    distances = np.linalg.norm(
        user_positions[:, None, :] - uav_positions[None, :, :],
        axis=2,
    )
    return int((distances <= radius).any(axis=1).sum())


def align_uav_ids(previous_positions: np.ndarray, current_positions: np.ndarray) -> np.ndarray:
    """Match current UAV positions to previous IDs by minimum total displacement."""
    if previous_positions.shape != current_positions.shape or len(current_positions) <= 1:
        return current_positions
    m = len(current_positions)
    distance_matrix = np.linalg.norm(
        previous_positions[:, None, :] - current_positions[None, :, :], axis=2
    )
    best_order = min(
        permutations(range(m)),
        key=lambda order: sum(distance_matrix[i, order[i]] for i in range(m)),
    )
    return current_positions[np.asarray(best_order, dtype=int)]


def uav_motion_and_safety_metrics(
    current_positions: np.ndarray,
    previous_positions: np.ndarray | None,
    cfg: PSOConfig,
) -> tuple[np.ndarray, float, float, int, int]:
    if previous_positions is None or previous_positions.shape != current_positions.shape:
        moves = np.zeros(len(current_positions), dtype=float)
    else:
        moves = np.linalg.norm(current_positions - previous_positions, axis=1)

    if len(current_positions) <= 1:
        min_distance = float("nan")
        safe_violations = 0
    else:
        distances = np.linalg.norm(
            current_positions[:, None, :] - current_positions[None, :, :], axis=2
        )
        upper = distances[np.triu_indices(len(current_positions), k=1)]
        min_distance = float(upper.min())
        safe_violations = int((upper < cfg.min_safe_distance).sum())

    movement_violations = int((moves > cfg.max_move_distance).sum())
    return moves, float(moves.max(initial=0.0)), min_distance, safe_violations, movement_violations


def coverage_rate(user_positions: np.ndarray, uav_positions: np.ndarray, radius: float) -> float:
    if len(user_positions) == 0:
        return 0.0
    return float(covered_user_count(user_positions, uav_positions, radius) / len(user_positions))


def local_density_bonus(user_positions: np.ndarray, uav_positions: np.ndarray, radius: float) -> float:
    if len(user_positions) == 0:
        return 0.0
    distances = np.linalg.norm(user_positions[:, None, :] - uav_positions[None, :, :], axis=2)
    return float(np.minimum(distances.min(axis=1) / max(radius, 1e-9), 1.0).mean())


def mas_overlap_penalty(uav_positions: np.ndarray, communication_range: float, radius: float) -> float:
    m = len(uav_positions)
    if m <= 1:
        return 0.0
    penalty = 0.0
    pairs = 0
    for i in range(m):
        for j in range(i + 1, m):
            dist = float(np.linalg.norm(uav_positions[i] - uav_positions[j]))
            if dist <= communication_range:
                pairs += 1
                penalty += max(0.0, (2.0 * radius - dist) / max(2.0 * radius, 1e-9))
    return penalty / max(pairs, 1)


def constraint_penalty(
    uav_positions: np.ndarray,
    previous_positions: np.ndarray | None,
    cfg: PSOConfig,
) -> tuple[float, float]:
    m = len(uav_positions)
    safe_penalty = 0.0
    pair_count = 0
    for i in range(m):
        for j in range(i + 1, m):
            pair_count += 1
            dist = float(np.linalg.norm(uav_positions[i] - uav_positions[j]))
            safe_penalty += max(0.0, cfg.min_safe_distance - dist) / cfg.min_safe_distance
    safe_penalty /= max(pair_count, 1)

    move_penalty = 0.0
    if previous_positions is not None and previous_positions.shape == uav_positions.shape:
        moves = np.linalg.norm(uav_positions - previous_positions, axis=1)
        move_penalty = float(np.maximum(0.0, moves - cfg.max_move_distance).mean())
        move_penalty /= max(cfg.max_move_distance, 1e-9)
    return safe_penalty, move_penalty


def evaluate_particle(
    flat_particle: np.ndarray,
    user_positions: np.ndarray,
    previous_positions: np.ndarray | None,
    boundary: Boundary,
    radius: float,
    m: int,
    cfg: PSOConfig,
) -> tuple[float, float]:
    uav_positions = flat_particle.reshape(m, 2)
    cover = coverage_rate(user_positions, uav_positions, radius)
    safe_penalty, move_penalty = constraint_penalty(uav_positions, previous_positions, cfg)
    overlap_penalty = mas_overlap_penalty(uav_positions, cfg.communication_range, radius)
    density_bonus = 1.0 - local_density_bonus(user_positions, uav_positions, radius)
    fitness = (
        cover
        + cfg.density_bonus_weight * density_bonus
        - cfg.safe_penalty_weight * safe_penalty
        - cfg.move_penalty_weight * move_penalty
        - cfg.overlap_penalty_weight * overlap_penalty
    )
    clipped = boundary.clip_points(uav_positions)
    if not np.allclose(clipped, uav_positions):
        fitness -= 1.0
    return fitness, cover


def initialize_particles(
    rng: np.random.Generator,
    m: int,
    n_particles: int,
    boundary: Boundary,
    previous_best: np.ndarray | None,
    cfg: PSOConfig,
) -> np.ndarray:
    if previous_best is None or previous_best.shape != (m, 2):
        xs = rng.uniform(boundary.xmin, boundary.xmax, size=(n_particles, m, 1))
        ys = rng.uniform(boundary.ymin, boundary.ymax, size=(n_particles, m, 1))
        particles = np.concatenate([xs, ys], axis=2)
    else:
        particles = previous_best[None, :, :] + rng.normal(
            0.0, cfg.warm_noise, size=(n_particles, m, 2)
        )
        particles[0] = previous_best
        for idx in range(1, max(2, n_particles // 5)):
            xs = rng.uniform(boundary.xmin, boundary.xmax, size=(m, 1))
            ys = rng.uniform(boundary.ymin, boundary.ymax, size=(m, 1))
            particles[idx] = np.concatenate([xs, ys], axis=1)
    for idx in range(n_particles):
        particles[idx] = boundary.clip_points(particles[idx])
    return particles.reshape(n_particles, m * 2)


def run_pso_for_time_slot(
    user_positions: np.ndarray,
    boundary: Boundary,
    radius: float,
    m: int,
    cfg: PSOConfig,
    rng: np.random.Generator,
    previous_best: np.ndarray | None,
) -> tuple[np.ndarray, float, float, int]:
    dim = m * 2
    particles = initialize_particles(rng, m, cfg.particles, boundary, previous_best, cfg)
    velocities = rng.normal(0.0, 2.0, size=(cfg.particles, dim))
    pbest = particles.copy()
    pbest_scores = np.full(cfg.particles, -np.inf)
    pbest_cover = np.zeros(cfg.particles)

    gbest = particles[0].copy()
    gbest_score = -np.inf
    gbest_cover = 0.0
    best_score_history = []

    for _ in range(cfg.iterations):
        for i in range(cfg.particles):
            score, cover = evaluate_particle(
                particles[i], user_positions, previous_best, boundary, radius, m, cfg
            )
            if score > pbest_scores[i]:
                pbest_scores[i] = score
                pbest_cover[i] = cover
                pbest[i] = particles[i].copy()
            if score > gbest_score:
                gbest_score = score
                gbest_cover = cover
                gbest = particles[i].copy()

        r1 = rng.random((cfg.particles, dim))
        r2 = rng.random((cfg.particles, dim))
        velocities = cfg.inertia * velocities + cfg.c1 * r1 * (pbest - particles) + cfg.c2 * r2 * (gbest - particles)
        particles = particles + velocities
        shaped = particles.reshape(cfg.particles, m, 2)
        for idx in range(cfg.particles):
            shaped[idx] = boundary.clip_points(shaped[idx])
        particles = shaped.reshape(cfg.particles, dim)

        best_score_history.append(float(gbest_score))

    tolerance = 0.01 * max(1.0, abs(float(gbest_score)))
    convergence_iteration = next(
        idx + 1
        for idx, historical_score in enumerate(best_score_history)
        if historical_score >= float(gbest_score) - tolerance
    )
    return gbest.reshape(m, 2), float(gbest_score), float(gbest_cover), convergence_iteration


def run_dynamic_mas_pso(
    positions_tensor: np.ndarray,
    time_slots: list[int],
    boundary: Boundary,
    m: int,
    radius: float,
    cfg: PSOConfig,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    previous_best = None
    rows = []
    summary = []
    start = time.perf_counter()
    for time_index, t in enumerate(time_slots):
        slot_start = time.perf_counter()
        best, score, cover, convergence_iteration = run_pso_for_time_slot(
            positions_tensor[time_index],
            boundary,
            radius,
            m,
            cfg,
            rng,
            previous_best,
        )
        slot_elapsed = time.perf_counter() - slot_start
        covered_users = covered_user_count(positions_tensor[time_index], best, radius)
        total_users = int(len(positions_tensor[time_index]))
        moves, max_move, min_distance, safe_violations, movement_violations = (
            uav_motion_and_safety_metrics(best, previous_best, cfg)
        )
        move_mean = float(moves.mean())
        overlap_metric = mas_overlap_penalty(best, cfg.communication_range, radius)
        previous_best = best
        for uav_id, ((x, y), move_distance) in enumerate(zip(best, moves), start=1):
            rows.append(
                {
                    "time_slot": t,
                    "uav_id": uav_id,
                    "x": x,
                    "y": y,
                    "move_from_previous": move_distance,
                }
            )
        summary.append(
            {
                "time_slot": t,
                "coverage_rate": cover,
                "covered_users": covered_users,
                "total_users": total_users,
                "fitness": score,
                "mean_move_from_previous": move_mean,
                "max_move_from_previous": max_move,
                "min_inter_uav_distance": min_distance,
                "safe_distance_violations": safe_violations,
                "movement_violations": movement_violations,
                "overlap_metric": overlap_metric,
                "configured_min_safe_distance": cfg.min_safe_distance,
                "configured_max_move_distance": cfg.max_move_distance,
                "uav_count": m,
                "coverage_radius": radius,
                "runtime_seconds": slot_elapsed,
                "convergence_iteration": convergence_iteration,
            }
        )
    elapsed = time.perf_counter() - start
    summary_df = pd.DataFrame(summary)
    return pd.DataFrame(rows), summary_df


def random_uav_positions(rng: np.random.Generator, boundary: Boundary, m: int) -> np.ndarray:
    xs = rng.uniform(boundary.xmin, boundary.xmax, size=(m, 1))
    ys = rng.uniform(boundary.ymin, boundary.ymax, size=(m, 1))
    return np.concatenate([xs, ys], axis=1)


def simple_kmeans(
    points: np.ndarray,
    k: int,
    boundary: Boundary,
    rng: np.random.Generator,
    iterations: int = KMEANS_ITERATIONS,
) -> np.ndarray:
    if len(points) == 0:
        return random_uav_positions(rng, boundary, k)

    unique_points = np.unique(points, axis=0)
    replace = len(unique_points) < k
    init_idx = rng.choice(len(unique_points), size=k, replace=replace)
    centers = unique_points[init_idx].astype(float)

    for _ in range(iterations):
        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        new_centers = centers.copy()
        for cluster_id in range(k):
            cluster_points = points[labels == cluster_id]
            if len(cluster_points) == 0:
                new_centers[cluster_id] = points[rng.integers(0, len(points))]
            else:
                new_centers[cluster_id] = cluster_points.mean(axis=0)
        new_centers = boundary.clip_points(new_centers)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return boundary.clip_points(centers)


def run_standard_pso_for_positions(
    user_positions: np.ndarray,
    boundary: Boundary,
    radius: float,
    m: int,
    cfg: PSOConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    dim = m * 2
    particles = initialize_particles(rng, m, cfg.particles, boundary, None, cfg)
    velocities = rng.normal(0.0, 2.0, size=(cfg.particles, dim))
    pbest = particles.copy()
    pbest_scores = np.full(cfg.particles, -np.inf)
    gbest = particles[0].copy()
    gbest_score = -np.inf

    for _ in range(cfg.iterations):
        for i in range(cfg.particles):
            score = coverage_rate(user_positions, particles[i].reshape(m, 2), radius)
            if score > pbest_scores[i]:
                pbest_scores[i] = score
                pbest[i] = particles[i].copy()
            if score > gbest_score:
                gbest_score = score
                gbest = particles[i].copy()

        r1 = rng.random((cfg.particles, dim))
        r2 = rng.random((cfg.particles, dim))
        velocities = cfg.inertia * velocities + cfg.c1 * r1 * (pbest - particles) + cfg.c2 * r2 * (gbest - particles)
        particles = particles + velocities
        shaped = particles.reshape(cfg.particles, m, 2)
        for idx in range(cfg.particles):
            shaped[idx] = boundary.clip_points(shaped[idx])
        particles = shaped.reshape(cfg.particles, dim)

    return gbest.reshape(m, 2), float(gbest_score)


def build_algorithm_frames(
    algorithm: str,
    positions_by_time: list[np.ndarray],
    positions_tensor: np.ndarray,
    time_slots: list[int],
    radius: float,
    elapsed: float,
    cfg: PSOConfig,
    slot_runtimes: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary = []
    previous_positions = None
    if slot_runtimes is None:
        slot_runtimes = [elapsed / max(len(time_slots), 1)] * len(time_slots)
    for time_index, t in enumerate(time_slots):
        uav_positions = positions_by_time[time_index]
        if previous_positions is not None:
            uav_positions = align_uav_ids(previous_positions, uav_positions)
        user_positions = positions_tensor[time_index]
        runtime_seconds = float(slot_runtimes[time_index])
        cover = coverage_rate(user_positions, uav_positions, radius)
        covered_users = covered_user_count(user_positions, uav_positions, radius)
        total_users = int(len(user_positions))
        moves, max_move, min_distance, safe_violations, movement_violations = (
            uav_motion_and_safety_metrics(uav_positions, previous_positions, cfg)
        )
        move_mean = float(moves.mean())
        overlap_metric = mas_overlap_penalty(
            uav_positions, cfg.communication_range, radius
        )
        previous_positions = uav_positions
        for uav_id, ((x, y), move_distance) in enumerate(zip(uav_positions, moves), start=1):
            rows.append(
                {
                    "algorithm": algorithm,
                    "time_slot": t,
                    "uav_id": uav_id,
                    "x": x,
                    "y": y,
                    "move_from_previous": move_distance,
                }
            )
        summary.append(
            {
                "algorithm": algorithm,
                "time_slot": t,
                "coverage_rate": cover,
                "covered_users": covered_users,
                "total_users": total_users,
                "mean_move_from_previous": move_mean,
                "max_move_from_previous": max_move,
                "min_inter_uav_distance": min_distance,
                "safe_distance_violations": safe_violations,
                "movement_violations": movement_violations,
                "overlap_metric": overlap_metric,
                "configured_min_safe_distance": cfg.min_safe_distance,
                "configured_max_move_distance": cfg.max_move_distance,
                "uav_count": len(uav_positions),
                "coverage_radius": radius,
                "runtime_seconds": runtime_seconds,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary)


def run_random_deployment(
    positions_tensor: np.ndarray,
    time_slots: list[int],
    boundary: Boundary,
    m: int,
    radius: float,
    cfg: PSOConfig,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    positions_by_time = []
    slot_runtimes = []
    for _ in time_slots:
        slot_start = time.perf_counter()
        positions_by_time.append(random_uav_positions(rng, boundary, m))
        slot_runtimes.append(time.perf_counter() - slot_start)
    elapsed = time.perf_counter() - start
    return build_algorithm_frames(
        "Random_Deployment", positions_by_time, positions_tensor, time_slots, radius, elapsed, cfg, slot_runtimes
    )


def run_kmeans_deployment(
    positions_tensor: np.ndarray,
    time_slots: list[int],
    boundary: Boundary,
    m: int,
    radius: float,
    cfg: PSOConfig,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    positions_by_time = []
    slot_runtimes = []
    for time_index, _ in enumerate(time_slots):
        slot_start = time.perf_counter()
        positions_by_time.append(simple_kmeans(positions_tensor[time_index], m, boundary, rng))
        slot_runtimes.append(time.perf_counter() - slot_start)
    elapsed = time.perf_counter() - start
    return build_algorithm_frames(
        "KMeans", positions_by_time, positions_tensor, time_slots, radius, elapsed, cfg, slot_runtimes
    )


def run_standard_pso_deployment(
    positions_tensor: np.ndarray,
    time_slots: list[int],
    boundary: Boundary,
    m: int,
    radius: float,
    cfg: PSOConfig,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    positions_by_time = []
    slot_runtimes = []
    for time_index, _ in enumerate(time_slots):
        slot_start = time.perf_counter()
        positions_by_time.append(
            run_standard_pso_for_positions(positions_tensor[time_index], boundary, radius, m, cfg, rng)[0]
        )
        slot_runtimes.append(time.perf_counter() - slot_start)
    elapsed = time.perf_counter() - start
    return build_algorithm_frames(
        "Standard_PSO", positions_by_time, positions_tensor, time_slots, radius, elapsed, cfg, slot_runtimes
    )


def run_static_pso_deployment(
    positions_tensor: np.ndarray,
    time_slots: list[int],
    boundary: Boundary,
    m: int,
    radius: float,
    cfg: PSOConfig,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    aggregate_positions = positions_tensor.reshape(-1, 2)
    start = time.perf_counter()
    static_positions, _ = run_standard_pso_for_positions(
        aggregate_positions, boundary, radius, m, cfg, rng
    )
    optimization_elapsed = time.perf_counter() - start
    positions_by_time = [static_positions.copy() for _ in time_slots]
    elapsed = time.perf_counter() - start
    per_slot_runtime = optimization_elapsed / max(len(time_slots), 1)
    slot_runtimes = [per_slot_runtime] * len(time_slots)
    return build_algorithm_frames(
        "Static_PSO", positions_by_time, positions_tensor, time_slots, radius, elapsed, cfg, slot_runtimes
    )


def save_algorithm_outputs(
    algorithm: str,
    positions_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    positions_df.to_csv(output_dir / f"{algorithm}_uav_positions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / f"{algorithm}_time_slot_summary.csv", index=False, encoding="utf-8-sig")
    avg_row = {
        "algorithm": algorithm,
        "avg_coverage": float(summary_df["coverage_rate"].mean()),
        "min_coverage": float(summary_df["coverage_rate"].min()),
        "max_coverage": float(summary_df["coverage_rate"].max()),
        "std_coverage": float(summary_df["coverage_rate"].std(ddof=0)),
        "avg_covered_users": float(summary_df["covered_users"].mean()),
        "avg_move_distance": float(summary_df["mean_move_from_previous"].mean()),
        "max_move_distance_observed": float(summary_df["max_move_from_previous"].max()),
        "min_inter_uav_distance_observed": float(summary_df["min_inter_uav_distance"].min()),
        "safe_distance_violations_total": int(summary_df["safe_distance_violations"].sum()),
        "movement_violations_total": int(summary_df["movement_violations"].sum()),
        "avg_overlap_metric": float(summary_df["overlap_metric"].mean()),
        "configured_min_safe_distance": float(summary_df["configured_min_safe_distance"].iloc[0]),
        "configured_max_move_distance": float(summary_df["configured_max_move_distance"].iloc[0]),
        "avg_runtime_seconds": float(summary_df["runtime_seconds"].mean()),
        "min_runtime_seconds": float(summary_df["runtime_seconds"].min()),
        "max_runtime_seconds": float(summary_df["runtime_seconds"].max()),
        "time_slots": int(len(summary_df)),
        "uav_count": int(summary_df["uav_count"].iloc[0]),
        "coverage_radius": float(summary_df["coverage_radius"].iloc[0]),
        "elapsed_seconds_total": float(summary_df["runtime_seconds"].sum()),
    }
    return avg_row


def save_algorithm_comparison(avg_rows: list[dict], output_dir: Path) -> pd.DataFrame:
    comparison_df = pd.DataFrame(avg_rows).sort_values("avg_coverage", ascending=False)
    comparison_df["rank"] = np.arange(1, len(comparison_df) + 1)
    speed_df = comparison_df.sort_values("avg_runtime_seconds", ascending=True).copy()
    speed_df["speed_rank"] = np.arange(1, len(speed_df) + 1)
    comparison_df = comparison_df.merge(
        speed_df[["algorithm", "speed_rank"]],
        on="algorithm",
        how="left",
    )
    comparison_df.to_csv(
        output_dir / "algorithm_average_coverage_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plt.figure(figsize=(9.2, 5.2), dpi=140)
    ordered = comparison_df.sort_values("avg_coverage", ascending=True)
    plt.barh(ordered["algorithm"], ordered["avg_coverage"], color="#2f6f9f")
    plt.xlabel("Average coverage rate")
    plt.title("Average coverage comparison")
    plt.xlim(0, min(1.05, max(0.2, float(ordered["avg_coverage"].max()) + 0.12)))
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "algorithm_average_coverage_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9.2, 5.2), dpi=140)
    runtime_ordered = speed_df.sort_values("avg_runtime_seconds", ascending=False)
    plt.barh(runtime_ordered["algorithm"], runtime_ordered["avg_runtime_seconds"], color="#6a8f3f")
    plt.xlabel("Average runtime per time slot (seconds)")
    plt.title("Average runtime comparison")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "algorithm_runtime_comparison.png", dpi=300)
    plt.close()
    return comparison_df


def save_combined_coverage_plot(
    summaries: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    colors = {
        "MAS_PSO": "#c43d3d",
        "Standard_PSO": "#2f6f9f",
        "Static_PSO": "#6a8f3f",
        "KMeans": "#8b5fbf",
        "Random_Deployment": "#d28b26",
    }
    labels = {
        "MAS_PSO": "MAS-PSO",
        "Standard_PSO": "Standard PSO",
        "Static_PSO": "Static PSO",
        "KMeans": "K-means",
        "Random_Deployment": "Random deployment",
    }
    plt.figure(figsize=(10.0, 5.8), dpi=150)
    for algorithm in [
        "MAS_PSO",
        "Standard_PSO",
        "Static_PSO",
        "KMeans",
        "Random_Deployment",
    ]:
        summary = summaries[algorithm].sort_values("time_slot")
        plt.plot(
            summary["time_slot"],
            summary["coverage_rate"],
            marker="o",
            markersize=4,
            linewidth=2,
            color=colors[algorithm],
            label=labels[algorithm],
        )
    plt.xlabel("Time slot")
    plt.ylabel("Coverage rate")
    plt.ylim(0.0, 1.0)
    plt.title("Coverage Rate across Algorithms")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / "Figure_5_3_coverage_rate_by_time_slot.png", dpi=300)
    plt.close()


def save_movement_comparison_plot(
    summaries: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    styles = {
        "MAS_PSO": ("MAS-PSO", "#c43d3d"),
        "Standard_PSO": ("Standard PSO", "#2f6f9f"),
        "KMeans": ("K-means", "#8b5fbf"),
        "Random_Deployment": ("Random deployment", "#d28b26"),
    }
    plt.figure(figsize=(10.0, 5.8), dpi=150)
    for algorithm, (label, color) in styles.items():
        summary = summaries[algorithm].sort_values("time_slot")
        plt.plot(
            summary["time_slot"],
            summary["mean_move_from_previous"],
            marker="o",
            markersize=4,
            linewidth=2,
            color=color,
            label=label,
        )
    plt.xlabel("Time slot")
    plt.ylabel("Mean UAV movement distance")
    plt.title("Mean UAV Movement Distance across Dynamic Algorithms")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(
        output_dir / "Figure_5_4_mean_uav_movement_by_time_slot.png", dpi=300
    )
    plt.close()


def select_representative_time_slots(
    positions_tensor: np.ndarray,
    time_slots: list[int],
    mas_summary: pd.DataFrame,
    static_summary: pd.DataFrame,
) -> pd.DataFrame:
    mas = mas_summary.set_index("time_slot")
    static = static_summary.set_index("time_slot")
    user_movements = np.zeros(len(time_slots), dtype=float)
    if len(time_slots) > 1:
        displacement = np.linalg.norm(
            positions_tensor[1:] - positions_tensor[:-1], axis=2
        )
        user_movements[1:] = displacement.mean(axis=1)
    movement_by_slot = dict(zip(time_slots, user_movements))

    criteria = [
        (
            "lowest_mas_coverage",
            sorted(time_slots, key=lambda slot: float(mas.loc[slot, "coverage_rate"])),
        ),
        (
            "largest_uav_movement",
            sorted(
                time_slots,
                key=lambda slot: float(
                    mas.loc[slot, "mean_move_from_previous"]
                ),
                reverse=True,
            ),
        ),
        (
            "highest_mas_coverage",
            sorted(
                time_slots,
                key=lambda slot: float(mas.loc[slot, "coverage_rate"]),
                reverse=True,
            ),
        ),
    ]

    selected_rows = []
    for criterion, candidates in criteria:
        slot = candidates[0]
        mas_coverage = float(mas.loc[slot, "coverage_rate"])
        static_coverage = float(static.loc[slot, "coverage_rate"])
        if criterion in {"lowest_mas_coverage", "highest_mas_coverage"}:
            selection_score = mas_coverage
        elif criterion == "largest_uav_movement":
            selection_score = float(mas.loc[slot, "mean_move_from_previous"])
        selected_rows.append(
            {
                "criterion": criterion,
                "time_slot": slot,
                "selection_score": selection_score,
                "mean_user_movement": movement_by_slot[slot],
                "mean_uav_movement": float(
                    mas.loc[slot, "mean_move_from_previous"]
                ),
                "mas_coverage_rate": mas_coverage,
                "mas_covered_users": int(mas.loc[slot, "covered_users"]),
                "static_coverage_rate": static_coverage,
                "static_covered_users": int(static.loc[slot, "covered_users"]),
                "mas_minus_static_coverage": mas_coverage - static_coverage,
            }
        )
    return pd.DataFrame(selected_rows)


def save_representative_time_slices_plot(
    positions_tensor: np.ndarray,
    time_slots: list[int],
    boundary: Boundary,
    mas_positions: pd.DataFrame,
    mas_summary: pd.DataFrame,
    selected_slots: pd.DataFrame,
    radius: float,
    output_dir: Path,
) -> None:
    display_names = {
        "lowest_mas_coverage": "Lowest MAS-PSO coverage",
        "largest_uav_movement": "Largest MAS-PSO UAV movement",
        "highest_mas_coverage": "Highest MAS-PSO coverage",
    }
    time_to_index = {slot: index for index, slot in enumerate(time_slots)}
    summary = mas_summary.set_index("time_slot")
    panel_count = len(selected_slots)
    fig, axes = plt.subplots(
        1,
        panel_count,
        figsize=(6.0 * panel_count, 5.8),
        dpi=150,
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for panel_index, (axis, (_, selected)) in enumerate(
        zip(axes, selected_slots.iterrows())
    ):
        slot = int(selected["time_slot"])
        users = positions_tensor[time_to_index[slot]]
        uavs = mas_positions[mas_positions["time_slot"] == slot].sort_values("uav_id")
        axis.scatter(
            users[:, 0],
            users[:, 1],
            s=13,
            alpha=0.42,
            color="#2f6f9f",
            edgecolors="none",
            label="Users",
            zorder=2,
        )
        for uav_index, (_, uav) in enumerate(uavs.iterrows()):
            coverage_circle = plt.Circle(
                (uav["x"], uav["y"]),
                radius,
                facecolor="#25a18e",
                edgecolor="#157f73",
                linewidth=1.2,
                alpha=0.13,
                label=(
                    "Coverage radius"
                    if panel_index == 0 and uav_index == 0
                    else None
                ),
                zorder=1,
            )
            axis.add_patch(coverage_circle)
            axis.scatter(
                uav["x"],
                uav["y"],
                marker="^",
                s=85,
                color="#c43d3d",
                edgecolor="white",
                linewidth=0.7,
                label=(
                    "MAS-PSO UAV"
                    if panel_index == 0 and uav_index == 0
                    else None
                ),
                zorder=4,
            )
            axis.annotate(
                str(int(uav["uav_id"])),
                (uav["x"], uav["y"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="#7a1f1f",
                zorder=5,
            )
        criterion_key = str(selected["criterion"])
        criterion = display_names[criterion_key]
        coverage = float(summary.loc[slot, "coverage_rate"])
        covered = int(summary.loc[slot, "covered_users"])
        total = int(summary.loc[slot, "total_users"])
        axis.set_title(
            f"{criterion}\nTime slot {slot}: {covered}/{total} covered ({coverage:.3f})",
            fontsize=10,
        )
        axis.set_xlim(boundary.xmin, boundary.xmax)
        axis.set_ylim(boundary.ymin, boundary.ymax)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.18)
        axis.set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(loc="lower left", frameon=False)
    fig.suptitle("Representative Dynamic MAS-PSO Deployments", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(
        output_dir / "Figure_5_5_representative_time_slices.png", dpi=300
    )
    plt.close(fig)


def save_constraint_and_overlap_plot(
    summaries: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    algorithms = [
        "MAS_PSO",
        "Standard_PSO",
        "Static_PSO",
        "KMeans",
        "Random_Deployment",
    ]
    labels = ["MAS-PSO", "Standard PSO", "Static PSO", "K-means", "Random"]
    safety = [
        int(summaries[name]["safe_distance_violations"].sum())
        for name in algorithms
    ]
    movement = [
        int(summaries[name]["movement_violations"].sum()) for name in algorithms
    ]
    overlap = [float(summaries[name]["overlap_metric"].mean()) for name in algorithms]

    x = np.arange(len(algorithms))
    width = 0.24
    fig, left_axis = plt.subplots(figsize=(11.0, 6.0), dpi=150)
    right_axis = left_axis.twinx()
    safety_bars = left_axis.bar(
        x - width,
        safety,
        width,
        color="#2f6f9f",
        label="Safety-distance violations",
    )
    movement_bars = left_axis.bar(
        x,
        movement,
        width,
        color="#c43d3d",
        label="Movement-distance violations",
    )
    overlap_bars = right_axis.bar(
        x + width,
        overlap,
        width,
        color="#6a8f3f",
        label="Mean overlap metric",
    )
    left_axis.set_xticks(x, labels)
    left_axis.set_ylabel("Violation count")
    right_axis.set_ylabel("Mean overlap metric")
    right_axis.set_ylim(0.0, max(0.1, max(overlap) * 1.25))
    left_axis.set_title("Constraint Violations and Coverage Overlap across Algorithms")
    left_axis.grid(axis="y", alpha=0.22)
    left_axis.bar_label(safety_bars, padding=3, fontsize=8)
    left_axis.bar_label(movement_bars, padding=3, fontsize=8)
    right_axis.bar_label(overlap_bars, padding=3, fmt="%.3f", fontsize=8)
    handles_left, labels_left = left_axis.get_legend_handles_labels()
    handles_right, labels_right = right_axis.get_legend_handles_labels()
    left_axis.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper left",
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "Figure_5_6_constraint_violations_and_overlap.png", dpi=300
    )
    plt.close(fig)


def run_baseline_suite(
    positions_tensor: np.ndarray,
    time_slots: list[int],
    boundary: Boundary,
    output_dir: Path,
    cfg: PSOConfig,
) -> tuple[list[dict], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    baseline_runs = [
        run_random_deployment(
            positions_tensor,
            time_slots,
            boundary,
            BASE_UAV_COUNT,
            BASE_COVERAGE_RADIUS,
            cfg,
            seed=RANDOM_SEED + 501,
        ),
        run_kmeans_deployment(
            positions_tensor,
            time_slots,
            boundary,
            BASE_UAV_COUNT,
            BASE_COVERAGE_RADIUS,
            cfg,
            seed=RANDOM_SEED + 502,
        ),
        run_standard_pso_deployment(
            positions_tensor,
            time_slots,
            boundary,
            BASE_UAV_COUNT,
            BASE_COVERAGE_RADIUS,
            cfg,
            seed=RANDOM_SEED + 503,
        ),
        run_static_pso_deployment(
            positions_tensor,
            time_slots,
            boundary,
            BASE_UAV_COUNT,
            BASE_COVERAGE_RADIUS,
            cfg,
            seed=RANDOM_SEED + 504,
        ),
    ]
    avg_rows = []
    positions_by_algorithm = {}
    summaries_by_algorithm = {}
    for positions_df, summary_df in baseline_runs:
        algorithm = str(summary_df["algorithm"].iloc[0])
        avg_rows.append(save_algorithm_outputs(algorithm, positions_df, summary_df, output_dir))
        positions_by_algorithm[algorithm] = positions_df
        summaries_by_algorithm[algorithm] = summary_df
    return avg_rows, positions_by_algorithm, summaries_by_algorithm


def make_line_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, path: Path) -> None:
    plt.figure(figsize=(8.2, 5.0), dpi=140)
    plt.plot(df[x_col], df[y_col], marker="o", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Average coverage rate")
    plt.ylim(0, min(1.05, max(0.2, float(df[y_col].max()) + 0.12)))
    plt.grid(alpha=0.28)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()




def save_boundary_plot(df: pd.DataFrame, selected: pd.DataFrame, boundary: Boundary, path: Path) -> None:
    sample_all = df.sample(min(len(df), 2500), random_state=RANDOM_SEED)
    sample_selected = selected.sample(min(len(selected), 1500), random_state=RANDOM_SEED)
    plt.figure(figsize=(7.0, 7.0), dpi=150)
    plt.scatter(sample_all[X_COL], sample_all[Y_COL], s=6, alpha=0.16, label="all observations")
    plt.scatter(sample_selected[X_COL], sample_selected[Y_COL], s=8, alpha=0.55, label="selected stable users")
    plt.plot(
        [boundary.xmin, boundary.xmax, boundary.xmax, boundary.xmin, boundary.xmin],
        [boundary.ymin, boundary.ymin, boundary.ymax, boundary.ymax, boundary.ymin],
        color="#d62728",
        linewidth=2,
        label="disaster boundary",
    )
    plt.xlim(GRID_MIN, GRID_MAX)
    plt.ylim(GRID_MIN, GRID_MAX)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Disaster boundary and selected users")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()




def run_scenario_suite(
    raw_df: pd.DataFrame,
    base_tensor: np.ndarray,
    time_slots: list[int],
    base_boundary: Boundary,
    output_dir: Path,
    cfg: PSOConfig,
    quick: bool,
) -> pd.DataFrame:
    scenario_rows = []

    uav_values = [2, 3, 4] if quick else [2, 3, 4, 5, 6, 7, 8]
    radius_values = [15, 25, 35] if quick else [10, 15, 20, 25, 30, 35, 40]
    user_values = [100, 200] if quick else [100, 200, 300, 400]
    area_values = [120, 160] if quick else [100, 120, 140, 160, 180]

    def record(scenario: str, variable_name: str, variable_value: float, summary: pd.DataFrame) -> None:
        scenario_rows.append(
            {
                "scenario": scenario,
                "variable_name": variable_name,
                "variable_value": variable_value,
                "avg_coverage": float(summary["coverage_rate"].mean()),
                "min_coverage": float(summary["coverage_rate"].min()),
                "max_coverage": float(summary["coverage_rate"].max()),
                "avg_move_distance": float(summary["mean_move_from_previous"].mean()),
                "max_move_distance": float(summary["max_move_from_previous"].max()),
                "min_inter_uav_distance": float(summary["min_inter_uav_distance"].min()),
                "safe_distance_violations": int(summary["safe_distance_violations"].sum()),
                "movement_violations": int(summary["movement_violations"].sum()),
                "time_slots": len(summary),
            }
        )

    for m in uav_values:
        _, summary = run_dynamic_mas_pso(
            base_tensor,
            time_slots,
            base_boundary,
            m,
            BASE_COVERAGE_RADIUS,
            cfg,
            seed=RANDOM_SEED + 100 + m,
        )
        record("A_Number_of_UAVs", "uav_count", m, summary)

    for radius in radius_values:
        _, summary = run_dynamic_mas_pso(
            base_tensor,
            time_slots,
            base_boundary,
            BASE_UAV_COUNT,
            radius,
            cfg,
            seed=RANDOM_SEED + 200 + int(radius),
        )
        record("B_Coverage_Radius", "coverage_radius", radius, summary)

    for user_count in user_values:
        n = min(user_count, base_tensor.shape[1])
        _, summary = run_dynamic_mas_pso(
            base_tensor[:, :n, :],
            time_slots,
            base_boundary,
            BASE_UAV_COUNT,
            BASE_COVERAGE_RADIUS,
            cfg,
            seed=RANDOM_SEED + 300 + n,
        )
        record("C_Number_of_Users", "user_count", n, summary)

    for area_size in area_values:
        boundary = centered_boundary(raw_df, area_size)
        selected, stable = select_stable_users(
            raw_df, boundary, time_slots, BASE_USER_COUNT, min_obs_ratio=MIN_OBS_RATIO
        )
        _, tensor = build_trajectory_tensor(selected, stable, time_slots)
        _, summary = run_dynamic_mas_pso(
            tensor,
            time_slots,
            boundary,
            BASE_UAV_COUNT,
            BASE_COVERAGE_RADIUS,
            cfg,
            seed=RANDOM_SEED + 400 + int(area_size),
        )
        record("D_Disaster_Area_Size", "area_size", area_size, summary)

    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(output_dir / "scenario_results.csv", index=False, encoding="utf-8-sig")

    plot_specs = [
        ("A_Number_of_UAVs", "uav_count", "Scenario A: Number of UAVs", "Number of UAVs"),
        ("B_Coverage_Radius", "coverage_radius", "Scenario B: Coverage Radius", "Coverage radius"),
        ("C_Number_of_Users", "user_count", "Scenario C: Number of Users", "Number of selected users"),
        ("D_Disaster_Area_Size", "area_size", "Scenario D: Disaster Area Size", "Boundary side length"),
    ]
    for scenario, x_col, title, xlabel in plot_specs:
        subset = scenario_df[scenario_df["scenario"] == scenario].sort_values("variable_value")
        plot_df = subset.rename(columns={"variable_value": x_col})
        make_line_plot(
            plot_df,
            x_col=x_col,
            y_col="avg_coverage",
            title=title,
            xlabel=xlabel,
            path=output_dir / f"{scenario}_line_chart.png",
        )
    return scenario_df


def main() -> None:
    args = parse_args()
    validate_formal_experiment()
    time_slots = list(range(args.t_start, args.t_end + 1))
    if args.quick:
        time_slots = time_slots[: min(len(time_slots), 8)]

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    removed_outputs = remove_obsolete_outputs(output_dir)

    raw_df = load_selected_data(args.csv, args.day, time_slots)
    boundary = determine_disaster_boundary(raw_df)
    max_users = 180 if args.quick else BASE_USER_COUNT
    selected, stable_users = select_stable_users(raw_df, boundary, time_slots, max_users=max_users)
    user_ids, tensor = build_trajectory_tensor(selected, stable_users, time_slots)
    processed_trajectories = build_processed_trajectory_frame(
        selected, user_ids, tensor, time_slots
    )

    stable_users.to_csv(output_dir / "selected_stable_users.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(output_dir / "selected_user_observations.csv", index=False, encoding="utf-8-sig")
    processed_trajectories.to_csv(
        output_dir / "processed_user_trajectories.csv", index=False, encoding="utf-8-sig"
    )
    save_boundary_plot(raw_df, selected, boundary, output_dir / "boundary_and_selected_users.png")

    cfg = FORMAL_PSO_CONFIG
    uav_positions, base_summary = run_dynamic_mas_pso(
        tensor,
        time_slots,
        boundary,
        BASE_UAV_COUNT,
        BASE_COVERAGE_RADIUS,
        cfg,
        seed=RANDOM_SEED + 999,
    )
    mas_positions = uav_positions.copy()
    mas_summary = base_summary.copy()
    mas_positions.insert(0, "algorithm", "MAS_PSO")
    mas_summary.insert(0, "algorithm", "MAS_PSO")
    avg_rows = [
        save_algorithm_outputs("MAS_PSO", mas_positions, mas_summary, output_dir)
    ]

    baseline_avg_rows, baseline_positions, baseline_summaries = run_baseline_suite(
        tensor, time_slots, boundary, output_dir, cfg
    )
    avg_rows.extend(baseline_avg_rows)
    comparison_df = save_algorithm_comparison(avg_rows, output_dir)

    summaries_by_algorithm = {"MAS_PSO": mas_summary, **baseline_summaries}
    positions_by_algorithm = {"MAS_PSO": mas_positions, **baseline_positions}
    save_combined_coverage_plot(summaries_by_algorithm, output_dir)
    save_movement_comparison_plot(summaries_by_algorithm, output_dir)

    representative_slots = select_representative_time_slots(
        tensor,
        time_slots,
        mas_summary,
        summaries_by_algorithm["Static_PSO"],
    )
    representative_slots.to_csv(
        output_dir / "Figure_5_5_selected_time_slots.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_representative_time_slices_plot(
        tensor,
        time_slots,
        boundary,
        positions_by_algorithm["MAS_PSO"],
        mas_summary,
        representative_slots,
        BASE_COVERAGE_RADIUS,
        output_dir,
    )
    save_constraint_and_overlap_plot(summaries_by_algorithm, output_dir)

    run_scenario_suite(raw_df, tensor, time_slots, boundary, output_dir, cfg, quick=args.quick)
    verify_formal_outputs(output_dir)

    print("Done.")
    print(f"Output directory: {output_dir}")
    print(f"Selected users: {len(user_ids)}")
    print(f"Boundary: {boundary}")
    print("Formal experiment parameters:")
    print(cfg)
    print(f"Removed obsolete outputs: {len(removed_outputs)}")
    print(f"Validated formal outputs: {len(FORMAL_OUTPUT_FILENAMES)}")
    print("Average coverage comparison:")
    print(
        comparison_df[
            ["rank", "speed_rank", "algorithm", "avg_coverage", "avg_covered_users", "avg_runtime_seconds"]
        ].to_string(index=False)
    )
    print("Representative time slots:")
    print(representative_slots.to_string(index=False))


if __name__ == "__main__":
    main()
