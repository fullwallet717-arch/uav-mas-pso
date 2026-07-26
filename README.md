# Dynamic UAV Deployment with MAS-PSO

This project studies dynamic UAV deployment for disaster response using mobile user trajectories.

## Version

Version `v0.8.0` updates the objective weights and PSO parameters used by MAS-PSO and the PSO-based baseline algorithms.

## Files

- `data_preprocessing.py`: filters the trajectory data, determines the disaster-area boundary, selects users, and fills missing positions.
- `mas_coordination.py`: shares UAV positions, builds communication links, and evaluates coverage, density, safety, movement, and overlap.
- `standard_pso.py`: searches for a joint UAV deployment with standard PSO updates.
- `dynamic_mas_pso.py`: runs MAS-PSO over consecutive time slices with warm-start.
- `deployment_results.py`: provides shared time-slice results, averages, validation, and UAV ID alignment.
- `random_deployment.py`: samples a new random UAV deployment for each time slice.
- `kmeans_deployment.py`: places UAVs at current population cluster centers.
- `static_pso.py`: optimizes one fixed deployment for the aggregate population.
- `experiment_runner.py`: prepares one shared dataset and runs all five algorithms with the same settings.
- `result_export.py`: exports UAV positions, time-slice summaries, and the algorithm comparison table.
- `result_visualization.py`: generates Figures 5.3-5.6 from the experiment results.
- `Picture/uav_simulation_visualization.py`: generates a four-stage illustration of the UAV deployment process.
- `Picture/`: contains reproducible scripts for the Chapter 3 methodology figures.
- `tests/test_mas_coordination.py`: checks the MAS calculations with small deterministic examples.
- `tests/test_standard_pso.py`: checks initialization, repeatability, boundaries, and convergence history.
- `tests/test_dynamic_mas_pso.py`: checks warm-start, time-slice tracking, averages, and repeatability.
- `tests/test_baseline_algorithms.py`: checks the four baseline workflows and UAV ID alignment.
- `tests/test_result_outputs.py`: checks runtime fields, CSV exports, representative time-slice selection, and figure generation.

## Environment

- Python 3.13.5
- NumPy 2.5.0
- Pandas 3.0.4
- Matplotlib 3.11.0

Install the dependencies with:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Data

Place `yjmob100k-dataset2.csv` in the `data` directory before running the preprocessing script.

```powershell
python .\data_preprocessing.py
```

## Outputs

The preprocessing module creates:

- `selected_stable_users.csv`
- `selected_user_observations.csv`
- `processed_user_trajectories.csv`
- `boundary_and_selected_users.png`

Run the simulation with:

```powershell
python .\Picture\uav_simulation_visualization.py
```

The generated simulation figures and Chapter 3 methodology figures are saved in `report_picture`.

## Complete Experiment

Run all five algorithms and save their results with:

```powershell
python .\experiment_runner.py --csv .\data\yjmob100k-dataset2.csv --output .\results
```

For each algorithm, the program exports:

- `<algorithm>_uav_positions.csv`: UAV coordinates and movement distance in each time slice.
- `<algorithm>_time_slot_summary.csv`: coverage, fitness, constraint, overlap, movement, communication, and runtime statistics.

The shared `algorithm_comparison.csv` compares average coverage, fitness, movement, constraint violations, overlap, and runtime across the five algorithms.

The following figures are also generated:

- `Figure_5_3_coverage_rate_by_time_slot.png`: coverage-rate curves for all five algorithms.
- `Figure_5_4_mean_uav_movement_by_time_slot.png`: movement curves for MAS-PSO, Standard PSO, K-means, and Random Deployment.
- `Figure_5_5_representative_time_slices.png`: MAS-PSO deployments at its lowest-coverage, largest-UAV-movement, and highest-coverage time slices.
- `Figure_5_6_constraint_violations_and_overlap.png`: safety violations, movement violations, and average overlap for all algorithms.

`Figure_5_5_selected_time_slots.csv` records the exact time slices selected for Figure 5.5 and the corresponding coverage and movement values.

## MAS Coordination

The MAS module uses a shared-state simulation. UAVs within the communication range are treated as neighbors and are included in overlap coordination. It does not implement a wireless network protocol.

Version `v0.2.0` used the initial objective weights. Version `v0.8.0` updates them to `0.03` for density, `0.18` for safety, `0.12` for movement, and `0.04` for overlap.

## Standard PSO

Each particle represents the coordinates of all five UAVs. The optimizer compares each particle's personal best deployment with the swarm's global best deployment, then updates velocity and position. Candidate coordinates are kept inside the disaster-area boundary.

Version `v0.3.0` used the initial PSO defaults. Version `v0.8.0` updates them to `32` particles, `25` iterations, `w = 0.72`, and `c1 = c2 = 1.45`.

## Dynamic MAS-PSO

The first time slice starts from random particles. From the second time slice onward, the preceding best UAV deployment is copied into the new swarm and used as the center of nearby perturbed particles. A small group of random restart particles keeps some global exploration ability when the population distribution changes.

The warm-start values remain a position-noise standard deviation of `5.0` grid units and a random-restart ratio of `0.20`.

## Algorithm Parameters in v0.8.0

- UAV count: `5`
- Coverage radius: `25.0`
- Communication range: `60.0`
- Minimum safe distance: `12.0`
- Maximum movement distance per time slice: `35.0`
- Density reward weight: `0.03`
- Safety penalty weight: `0.18`
- Movement penalty weight: `0.12`
- Overlap penalty weight: `0.04`
- PSO particles: `32`
- PSO iterations: `25`
- Inertia weight: `0.72`
- Cognitive coefficient `c1`: `1.45`
- Social coefficient `c2`: `1.45`
- Warm-start noise standard deviation: `5.0`
- Random restart ratio: `0.20`

## Baseline Algorithms

- Random Deployment independently samples UAV coordinates in every time slice.
- K-means independently fits population cluster centers in every time slice.
- Standard PSO independently initializes and optimizes a new swarm in every time slice.
- Static PSO optimizes once for all time slices combined and then keeps the UAV coordinates fixed.

The dynamic algorithm and all baselines now return the same time-slice fields and average coverage and fitness values. UAV coordinates from independently generated deployments are matched to previous UAV IDs by minimum total movement before movement metrics are calculated.

Run the tests with:

```powershell
python -m unittest discover -s tests -v
```
