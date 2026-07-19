# Dynamic UAV Deployment with MAS-PSO

This project studies dynamic UAV deployment for disaster response using mobile user trajectories.

## Version

The local `v0.2.0` candidate adds MAS shared-state coordination and deployment evaluation to the files introduced in `v0.1.0`.

## Files

- `data_preprocessing.py`: filters the trajectory data, determines the disaster-area boundary, selects users, and fills missing positions.
- `mas_coordination.py`: shares UAV positions, builds communication links, and evaluates coverage, density, safety, movement, and overlap.
- `uav_simulation_visualization.py`: generates a four-stage illustration of the UAV deployment process.
- `tests/test_mas_coordination.py`: checks the MAS calculations with small deterministic examples.

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

Run the tests with:

```powershell
python -m unittest discover -s tests -v
```
