"""Create thesis figures from one unified experiment result."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deployment_results import DeploymentRunResult
from result_export import build_time_slice_frame

if TYPE_CHECKING:
    from experiment_runner import ExperimentResult


COLORS = {
    "MAS_PSO": "#c43d3d",
    "Standard_PSO": "#2f6f9f",
    "Static_PSO": "#6a8f3f",
    "KMeans": "#8b5fbf",
    "Random_Deployment": "#d28b26",
}
LABELS = {
    "MAS_PSO": "MAS-PSO",
    "Standard_PSO": "Standard PSO",
    "Static_PSO": "Static PSO",
    "KMeans": "K-means",
    "Random_Deployment": "Random deployment",
}


def _summary_map(experiment: ExperimentResult) -> dict[str, pd.DataFrame]:
    return {
        result.algorithm: build_time_slice_frame(result)
        for result in experiment.algorithms
    }


def save_coverage_figure(experiment: ExperimentResult, output_dir: Path) -> Path:
    summaries = _summary_map(experiment)
    order = ["MAS_PSO", "Standard_PSO", "Static_PSO", "KMeans", "Random_Deployment"]
    fig, axis = plt.subplots(figsize=(10.0, 5.8), dpi=150)
    for name in order:
        summary = summaries[name]
        axis.plot(
            summary["time_slot"],
            summary["coverage_rate"],
            marker="o",
            markersize=4,
            linewidth=2,
            color=COLORS[name],
            label=LABELS[name],
        )
    axis.set(xlabel="Time slot", ylabel="Coverage rate", ylim=(0.0, 1.0))
    axis.set_title("Coverage Rate across Algorithms")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, frameon=False)
    fig.tight_layout()
    path = Path(output_dir) / "Figure_5_3_coverage_rate_by_time_slot.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def save_movement_figure(experiment: ExperimentResult, output_dir: Path) -> Path:
    summaries = _summary_map(experiment)
    order = ["MAS_PSO", "Standard_PSO", "KMeans", "Random_Deployment"]
    fig, axis = plt.subplots(figsize=(10.0, 5.8), dpi=150)
    for name in order:
        summary = summaries[name]
        axis.plot(
            summary["time_slot"],
            summary["mean_uav_movement"],
            marker="o",
            markersize=4,
            linewidth=2,
            color=COLORS[name],
            label=LABELS[name],
        )
    axis.set(xlabel="Time slot", ylabel="Mean UAV movement distance")
    axis.set_title("Mean UAV Movement Distance across Dynamic Algorithms")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    path = Path(output_dir) / "Figure_5_4_mean_uav_movement_by_time_slot.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def select_representative_time_slices(
    experiment: ExperimentResult,
) -> pd.DataFrame:
    mas = experiment.get_algorithm("MAS_PSO")
    summary = build_time_slice_frame(mas).set_index("time_slot")
    slots = list(experiment.data.time_slots)
    user_movement = np.zeros(len(slots), dtype=float)
    if len(slots) > 1:
        displacement = np.linalg.norm(
            experiment.data.positions_tensor[1:]
            - experiment.data.positions_tensor[:-1],
            axis=2,
        )
        user_movement[1:] = displacement.mean(axis=1)
    user_movement_by_slot = dict(zip(slots, user_movement))

    selections = (
        ("lowest_mas_coverage", summary["coverage_rate"].idxmin()),
        ("largest_uav_movement", summary["mean_uav_movement"].idxmax()),
        ("highest_mas_coverage", summary["coverage_rate"].idxmax()),
    )
    rows = []
    for criterion, slot_value in selections:
        slot = int(slot_value)
        if criterion == "largest_uav_movement":
            score = float(summary.loc[slot, "mean_uav_movement"])
        else:
            score = float(summary.loc[slot, "coverage_rate"])
        rows.append(
            {
                "criterion": criterion,
                "time_slot": slot,
                "selection_score": score,
                "mean_user_movement": user_movement_by_slot[slot],
                "mean_uav_movement": float(
                    summary.loc[slot, "mean_uav_movement"]
                ),
                "mas_coverage_rate": float(summary.loc[slot, "coverage_rate"]),
                "mas_covered_users": int(summary.loc[slot, "covered_users"]),
            }
        )
    return pd.DataFrame(rows)


def save_representative_figure(
    experiment: ExperimentResult,
    output_dir: Path,
) -> tuple[Path, Path]:
    selected = select_representative_time_slices(experiment)
    mas = experiment.get_algorithm("MAS_PSO")
    mas_by_slot = {item.time_slot: item for item in mas.time_slices}
    time_to_index = {
        slot: index for index, slot in enumerate(experiment.data.time_slots)
    }
    titles = {
        "lowest_mas_coverage": "Lowest MAS-PSO coverage",
        "largest_uav_movement": "Largest MAS-PSO UAV movement",
        "highest_mas_coverage": "Highest MAS-PSO coverage",
    }
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8), dpi=150, sharex=True, sharey=True)
    radius = experiment.config.mas.coverage_radius
    boundary = experiment.data.boundary
    total_users = experiment.data.positions_tensor.shape[1]
    for panel, (axis, (_, row)) in enumerate(zip(axes, selected.iterrows())):
        slot = int(row["time_slot"])
        users = experiment.data.positions_tensor[time_to_index[slot]]
        time_slice = mas_by_slot[slot]
        axis.scatter(
            users[:, 0], users[:, 1], s=13, alpha=0.42, color="#2f6f9f",
            edgecolors="none", label="Users", zorder=2,
        )
        for uav_index, position in enumerate(time_slice.positions, start=1):
            circle = plt.Circle(
                position, radius, facecolor="#25a18e", edgecolor="#157f73",
                linewidth=1.2, alpha=0.13,
                label="Coverage radius" if panel == 0 and uav_index == 1 else None,
                zorder=1,
            )
            axis.add_patch(circle)
            axis.scatter(
                position[0], position[1], marker="^", s=85, color="#c43d3d",
                edgecolor="white", linewidth=0.7,
                label="MAS-PSO UAV" if panel == 0 and uav_index == 1 else None,
                zorder=4,
            )
            axis.annotate(
                str(uav_index), position, xytext=(5, 5), textcoords="offset points",
                fontsize=8, color="#7a1f1f", zorder=5,
            )
        axis.set_title(
            f"{titles[row['criterion']]}\nTime slot {slot}: "
            f"{int(row['mas_covered_users'])}/{total_users} covered "
            f"({row['mas_coverage_rate']:.3f})",
            fontsize=10,
        )
        axis.set_xlim(boundary.xmin, boundary.xmax)
        axis.set_ylim(boundary.ymin, boundary.ymax)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x")
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("y")
    axes[0].legend(loc="lower left", frameon=False)
    fig.suptitle("Representative Dynamic MAS-PSO Deployments", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figure_path = Path(output_dir) / "Figure_5_5_representative_time_slices.png"
    selection_path = Path(output_dir) / "Figure_5_5_selected_time_slots.csv"
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    selected.to_csv(selection_path, index=False, encoding="utf-8-sig")
    return figure_path, selection_path


def save_constraint_figure(experiment: ExperimentResult, output_dir: Path) -> Path:
    summaries = _summary_map(experiment)
    order = ["MAS_PSO", "Standard_PSO", "Static_PSO", "KMeans", "Random_Deployment"]
    safety = [int(summaries[name]["safety_distance_violations"].sum()) for name in order]
    movement = [int(summaries[name]["movement_distance_violations"].sum()) for name in order]
    overlap = [float(summaries[name]["overlap_metric"].mean()) for name in order]
    x = np.arange(len(order))
    width = 0.24
    fig, left = plt.subplots(figsize=(11.0, 6.0), dpi=150)
    right = left.twinx()
    safety_bars = left.bar(x - width, safety, width, color="#2f6f9f", label="Safety-distance violations")
    movement_bars = left.bar(x, movement, width, color="#c43d3d", label="Movement-distance violations")
    overlap_bars = right.bar(x + width, overlap, width, color="#6a8f3f", label="Mean overlap metric")
    left.set_xticks(x, [LABELS[name] for name in order])
    left.set_ylabel("Violation count")
    right.set_ylabel("Mean overlap metric")
    right.set_ylim(0.0, max(0.1, max(overlap) * 1.25))
    left.set_title("Constraint Violations and Coverage Overlap across Algorithms")
    left.grid(axis="y", alpha=0.22)
    left.bar_label(safety_bars, padding=3, fontsize=8)
    left.bar_label(movement_bars, padding=3, fontsize=8)
    right.bar_label(overlap_bars, padding=3, fmt="%.3f", fontsize=8)
    left_handles, left_labels = left.get_legend_handles_labels()
    right_handles, right_labels = right.get_legend_handles_labels()
    left.legend(left_handles + right_handles, left_labels + right_labels, loc="upper left", frameon=False)
    fig.tight_layout()
    path = Path(output_dir) / "Figure_5_6_constraint_violations_and_overlap.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def save_all_figures(experiment: ExperimentResult, output_dir: Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    representative, selection = save_representative_figure(experiment, output)
    return {
        "coverage": save_coverage_figure(experiment, output),
        "movement": save_movement_figure(experiment, output),
        "representative": representative,
        "representative_selection": selection,
        "constraints": save_constraint_figure(experiment, output),
    }
