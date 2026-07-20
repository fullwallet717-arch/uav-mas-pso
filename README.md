# Dynamic UAV Deployment with MAS-PSO

This project studies dynamic UAV deployment for disaster response using mobile user trajectories.

## Version

Version `v0.3.0` adds a standard PSO optimizer for one UAV deployment time slice. It builds on the data processing and MAS evaluation modules from the earlier versions.

## Files

- `data_preprocessing.py`: filters the trajectory data, determines the disaster-area boundary, selects users, and fills missing positions.
- `mas_coordination.py`: shares UAV positions, builds communication links, and evaluates coverage, density, safety, movement, and overlap.
- `standard_pso.py`: searches for a joint UAV deployment with standard PSO updates.
- `uav_simulation_visualization.py`: generates a four-stage illustration of the UAV deployment process.
- `tests/test_mas_coordination.py`: checks the MAS calculations with small deterministic examples.
- `tests/test_standard_pso.py`: checks initialization, repeatability, boundaries, and convergence history.

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
python .\uav_simulation_visualization.py
```

The generated simulation figures are saved in `results/simulation`.

## MAS Coordination

The MAS module uses a shared-state simulation. UAVs within the communication range are treated as neighbors and are included in overlap coordination. It does not implement a wireless network protocol.

Version `v0.2.0` uses preliminary objective weights: `0.05` for density, `0.20` for safety, `0.15` for movement, and `0.05` for overlap. These are initial round values rather than tuned parameters.

## Standard PSO

Each particle represents the coordinates of all five UAVs. The optimizer compares each particle's personal best deployment with the swarm's global best deployment, then updates velocity and position. Candidate coordinates are kept inside the disaster-area boundary.

The `v0.3.0` defaults are preliminary values: `20` particles, `20` iterations, `w = 0.70`, and `c1 = c2 = 1.50`. This version is limited to one time slice and does not use warm-start.

Run the tests with:

```powershell
python -m unittest discover -s tests -v
```
