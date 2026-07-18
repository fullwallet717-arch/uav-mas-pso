r"""
Independent UAV simulation visualization.

Run in PyCharm:
1. Open project folder: D:\MSC Project Result\Project
2. Open this file: uav_simulation_visualization.py
3. Run this file.

Outputs are written to the repository's results/simulation directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "simulation"
RANDOM_SEED = 20260708
AREA_MIN = 0.0
AREA_MAX = 200.0
UAV_COUNT = 5
UAV_DRAW_RADIUS = 5.5
COVERAGE_RADIUS = 28.0


def make_people(
    rng: np.random.Generator,
    centers: np.ndarray,
    counts: list[int],
    stds: list[float],
    scattered_count: int = 50,
) -> np.ndarray:
    clouds = []
    for center, count, std in zip(centers, counts, stds):
        cloud = rng.normal(loc=center, scale=std, size=(count, 2))
        clouds.append(cloud)
    scattered = rng.uniform(AREA_MIN, AREA_MAX, size=(scattered_count, 2))
    points = np.vstack([*clouds, scattered])
    return np.clip(points, AREA_MIN, AREA_MAX)


def draw_base(ax: plt.Axes, title: str) -> None:
    ax.add_patch(
        Rectangle(
            (AREA_MIN, AREA_MIN),
            AREA_MAX - AREA_MIN,
            AREA_MAX - AREA_MIN,
            fill=False,
            edgecolor="#222222",
            linewidth=1.5,
        )
    )
    ax.set_xlim(AREA_MIN - 5, AREA_MAX + 5)
    ax.set_ylim(AREA_MIN - 5, AREA_MAX + 5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.18, linewidth=0.6)


def draw_people(ax: plt.Axes, people: np.ndarray, color: str = "#4b77be") -> None:
    ax.scatter(
        people[:, 0],
        people[:, 1],
        s=10,
        c=color,
        alpha=0.34,
        linewidths=0,
        label="population",
    )


def draw_density_centers(ax: plt.Axes, centers: np.ndarray, label: str) -> None:
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        s=140,
        marker="*",
        c="#ff9f1c",
        edgecolors="#7a3e00",
        linewidths=0.8,
        label=label,
        zorder=5,
    )


def draw_uavs(
    ax: plt.Axes,
    positions: np.ndarray,
    label_prefix: str = "UAV",
    coverage: bool = False,
) -> None:
    for idx, (x, y) in enumerate(positions, start=1):
        if coverage:
            ax.add_patch(
                Circle(
                    (x, y),
                    COVERAGE_RADIUS,
                    facecolor="#2ec4b6",
                    edgecolor="#128277",
                    alpha=0.16,
                    linewidth=1.0,
                    zorder=2,
                )
            )
        ax.add_patch(
            Circle(
                (x, y),
                UAV_DRAW_RADIUS,
                facecolor="#ffffff",
                edgecolor="#d62828",
                linewidth=2.2,
                zorder=6,
            )
        )
        ax.text(
            x,
            y,
            str(idx),
            ha="center",
            va="center",
            fontsize=8,
            color="#d62828",
            fontweight="bold",
            zorder=7,
        )
    ax.scatter([], [], s=90, facecolors="#ffffff", edgecolors="#d62828", label=label_prefix)


def draw_arrows(ax: plt.Axes, start_positions: np.ndarray, target_positions: np.ndarray) -> None:
    for start, target in zip(start_positions, target_positions):
        arrow = FancyArrowPatch(
            posA=(start[0], start[1]),
            posB=(target[0], target[1]),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color="#6a4c93",
            alpha=0.88,
            zorder=4,
        )
        ax.add_patch(arrow)


def save_stage(
    path: Path,
    title: str,
    people: np.ndarray,
    density_centers: np.ndarray,
    uav_positions: np.ndarray,
    target_positions: np.ndarray | None = None,
    coverage: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=150)
    draw_base(ax, title)
    draw_people(ax, people)
    draw_density_centers(ax, density_centers, "high density center")
    if target_positions is not None:
        draw_arrows(ax, uav_positions, target_positions)
    draw_uavs(ax, uav_positions if target_positions is None else uav_positions, coverage=coverage)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    initial_density_centers = np.array(
        [
            [42.0, 42.0],
            [158.0, 42.0],
            [42.0, 158.0],
            [158.0, 158.0],
            [100.0, 100.0],
        ]
    )
    moved_density_centers = np.array(
        [
            [54.0, 54.0],
            [146.0, 55.0],
            [55.0, 146.0],
            [146.0, 145.0],
            [108.0, 94.0],
        ]
    )

    initial_people = make_people(
        rng,
        initial_density_centers,
        counts=[95, 95, 95, 95, 110],
        stds=[10.0, 10.0, 10.0, 10.0, 12.0],
        scattered_count=55,
    )
    moved_people = make_people(
        rng,
        moved_density_centers,
        counts=[95, 95, 95, 95, 110],
        stds=[10.0, 10.0, 10.0, 10.0, 12.0],
        scattered_count=55,
    )

    initial_uavs = np.array(
        [
            [28.0, 176.0],
            [174.0, 48.0],
            [58.0, 28.0],
            [178.0, 174.0],
            [78.0, 122.0],
        ]
    )
    first_targets = np.array(
        [
            [42.0, 158.0],
            [158.0, 42.0],
            [42.0, 42.0],
            [158.0, 158.0],
            [100.0, 100.0],
        ]
    )
    second_targets = np.array(
        [
            [55.0, 146.0],
            [146.0, 55.0],
            [54.0, 54.0],
            [146.0, 145.0],
            [108.0, 94.0],
        ]
    )

    stage_specs = [
        (
            "Stage 1: random UAV deployment",
            initial_people,
            initial_density_centers,
            initial_uavs,
            None,
            True,
        ),
        (
            "Stage 2: UAVs choose movement directions",
            initial_people,
            initial_density_centers,
            initial_uavs,
            first_targets,
            True,
        ),
        (
            "Stage 3: UAVs cover high-density areas",
            initial_people,
            initial_density_centers,
            first_targets,
            None,
            True,
        ),
        (
            "Stage 4: population moves, UAVs re-plan",
            moved_people,
            moved_density_centers,
            first_targets,
            second_targets,
            True,
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 12.8), dpi=150)
    for ax, (title, people, centers, uavs, targets, coverage) in zip(axes.ravel(), stage_specs):
        draw_base(ax, title)
        draw_people(ax, people)
        draw_density_centers(ax, centers, "high density center")
        if targets is not None:
            draw_arrows(ax, uavs, targets)
        draw_uavs(ax, uavs, coverage=coverage)
        ax.text(
            5,
            194,
            f"coverage radius = {COVERAGE_RADIUS:g}",
            fontsize=9,
            color="#128277",
            ha="left",
            va="top",
        )
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("UAV Dynamic Deployment Simulation", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    four_stage_path = OUTPUT_DIR / "UAV_simulation_four_stage.png"
    fig.savefig(four_stage_path, dpi=300)
    plt.close(fig)

    individual_paths = []
    for idx, (title, people, centers, uavs, targets, coverage) in enumerate(stage_specs, start=1):
        path = OUTPUT_DIR / f"UAV_simulation_stage_{idx}.png"
        save_stage(path, title, people, centers, uavs, targets, coverage)
        individual_paths.append(path)

    rows = []
    for stage_name, positions in [
        ("initial_random_position", initial_uavs),
        ("first_high_density_target", first_targets),
        ("second_high_density_target", second_targets),
    ]:
        for uav_id, (x, y) in enumerate(positions, start=1):
            rows.append({"stage": stage_name, "uav_id": uav_id, "x": x, "y": y})
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "UAV_simulation_uav_positions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Simulation visualization completed.")
    print(f"Four-stage figure: {four_stage_path}")
    for path in individual_paths:
        print(f"Stage figure: {path}")
    print(f"UAV positions: {OUTPUT_DIR / 'UAV_simulation_uav_positions.csv'}")


if __name__ == "__main__":
    main()
