"""Preprocess YJMob100K trajectories for the UAV deployment experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
CSV_PATH = ROOT_DIR / "data" / "yjmob100k-dataset2.csv"
OUTPUT_DIR = ROOT_DIR / "results" / "data_processing"

UID_COL = "uid"
DAY_COL = "d"
TIME_COL = "t"
X_COL = "x"
Y_COL = "y"

GRID_MIN = 1.0
GRID_MAX = 200.0
BOUNDARY_QUANTILE = 0.05
MIN_OBS_RATIO = 0.20
MAX_USERS = 400
RANDOM_SEED = 42


@dataclass
class Boundary:
    xmin: float
    xmax: float
    ymin: float
    ymax: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess the YJMob100K trajectory data.")
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--day", type=int, default=0)
    parser.add_argument("--t-start", type=int, default=0)
    parser.add_argument("--t-end", type=int, default=23)
    parser.add_argument("--max-users", type=int, default=MAX_USERS)
    return parser.parse_args()


def load_selected_data(csv_path: Path, day: int, time_slots: list[int]) -> pd.DataFrame:
    columns = [UID_COL, DAY_COL, TIME_COL, X_COL, Y_COL]
    data = pd.read_csv(csv_path, usecols=columns)
    data = data[(data[DAY_COL] == day) & data[TIME_COL].isin(time_slots)].copy()

    for column in [X_COL, Y_COL]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=columns)

    if data.empty:
        raise ValueError("No data remains after filtering the day and time range.")
    return data


def determine_disaster_boundary(
    data: pd.DataFrame,
    quantile: float = BOUNDARY_QUANTILE,
) -> Boundary:
    return Boundary(
        xmin=max(GRID_MIN, float(data[X_COL].quantile(quantile))),
        xmax=min(GRID_MAX, float(data[X_COL].quantile(1 - quantile))),
        ymin=max(GRID_MIN, float(data[Y_COL].quantile(quantile))),
        ymax=min(GRID_MAX, float(data[Y_COL].quantile(1 - quantile))),
    )


def select_stable_users(
    data: pd.DataFrame,
    boundary: Boundary,
    time_slots: list[int],
    max_users: int,
    min_obs_ratio: float = MIN_OBS_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inside = (
        data[X_COL].between(boundary.xmin, boundary.xmax)
        & data[Y_COL].between(boundary.ymin, boundary.ymax)
    )
    summary = (
        data.assign(in_boundary=inside)
        .groupby(UID_COL)
        .agg(
            observed_slots=(TIME_COL, "nunique"),
            all_observed_points_inside=("in_boundary", "all"),
            first_t=(TIME_COL, "min"),
            last_t=(TIME_COL, "max"),
        )
        .reset_index()
    )

    min_observations = max(1, int(np.ceil(len(time_slots) * min_obs_ratio)))
    stable_users = summary[
        (summary["observed_slots"] >= min_observations)
        & summary["all_observed_points_inside"]
    ].copy()
    stable_users = stable_users.sort_values(
        ["observed_slots", UID_COL], ascending=[False, True]
    ).head(max_users)
    stable_users["required_time_slots"] = len(time_slots)
    stable_users["observed_ratio"] = stable_users["observed_slots"] / len(time_slots)
    stable_users["imputed_slots"] = len(time_slots) - stable_users["observed_slots"]

    selected = data[data[UID_COL].isin(stable_users[UID_COL])].copy()
    selected = selected[
        selected[X_COL].between(boundary.xmin, boundary.xmax)
        & selected[Y_COL].between(boundary.ymin, boundary.ymax)
    ]
    if selected.empty:
        raise ValueError("No stable users were selected.")
    return selected, stable_users


def build_trajectory_tensor(
    selected: pd.DataFrame,
    stable_users: pd.DataFrame,
    time_slots: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    user_ids = stable_users[UID_COL].to_numpy()
    observations = selected.sort_values([UID_COL, TIME_COL]).drop_duplicates(
        [UID_COL, TIME_COL], keep="last"
    )

    full_index = pd.MultiIndex.from_product(
        [user_ids, time_slots], names=[UID_COL, TIME_COL]
    )
    positions = observations.set_index([UID_COL, TIME_COL])[[X_COL, Y_COL]].reindex(
        full_index
    )
    positions = positions.groupby(level=0, group_keys=False).ffill()
    positions = positions.groupby(level=0, group_keys=False).bfill().dropna()

    valid_users = positions.index.get_level_values(UID_COL).unique().to_numpy()
    positions_by_time = [
        positions.xs(time_slot, level=TIME_COL)
        .loc[valid_users, [X_COL, Y_COL]]
        .to_numpy(dtype=float)
        for time_slot in time_slots
    ]
    return valid_users, np.stack(positions_by_time, axis=0)


def build_processed_trajectory_frame(
    selected: pd.DataFrame,
    user_ids: np.ndarray,
    tensor: np.ndarray,
    time_slots: list[int],
) -> pd.DataFrame:
    positions = np.transpose(tensor, (1, 0, 2)).reshape(-1, 2)
    result = pd.DataFrame(
        {
            UID_COL: np.repeat(user_ids, len(time_slots)),
            TIME_COL: np.tile(time_slots, len(user_ids)),
            X_COL: positions[:, 0],
            Y_COL: positions[:, 1],
        }
    )

    observed_index = pd.MultiIndex.from_frame(
        selected.drop_duplicates([UID_COL, TIME_COL], keep="last")[
            [UID_COL, TIME_COL]
        ]
    )
    result_index = pd.MultiIndex.from_frame(result[[UID_COL, TIME_COL]])
    result["is_observed"] = result_index.isin(observed_index)
    result["is_imputed"] = ~result["is_observed"]
    result["position_source"] = np.where(
        result["is_observed"], "observed", "forward_or_backward_filled"
    )
    return result


def save_boundary_plot(
    data: pd.DataFrame,
    selected: pd.DataFrame,
    boundary: Boundary,
    path: Path,
) -> None:
    all_sample = data.sample(min(len(data), 2500), random_state=RANDOM_SEED)
    selected_sample = selected.sample(min(len(selected), 1500), random_state=RANDOM_SEED)

    plt.figure(figsize=(7, 7), dpi=150)
    plt.scatter(all_sample[X_COL], all_sample[Y_COL], s=6, alpha=0.16, label="all observations")
    plt.scatter(
        selected_sample[X_COL],
        selected_sample[Y_COL],
        s=8,
        alpha=0.55,
        label="selected users",
    )
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
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    time_slots = list(range(args.t_start, args.t_end + 1))
    args.output.mkdir(parents=True, exist_ok=True)

    data = load_selected_data(args.csv, args.day, time_slots)
    boundary = determine_disaster_boundary(data)
    selected, stable_users = select_stable_users(
        data, boundary, time_slots, args.max_users
    )
    user_ids, tensor = build_trajectory_tensor(selected, stable_users, time_slots)
    trajectories = build_processed_trajectory_frame(
        selected, user_ids, tensor, time_slots
    )

    stable_users.to_csv(
        args.output / "selected_stable_users.csv", index=False, encoding="utf-8-sig"
    )
    selected.to_csv(
        args.output / "selected_user_observations.csv", index=False, encoding="utf-8-sig"
    )
    trajectories.to_csv(
        args.output / "processed_user_trajectories.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_boundary_plot(
        data, selected, boundary, args.output / "boundary_and_selected_users.png"
    )

    print(f"Selected users: {len(user_ids)}")
    print(f"Boundary: {boundary}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
