# Dynamic UAV Deployment with MAS-PSO

This project studies dynamic UAV deployment for disaster response using mobile user trajectories.

## Version

Version `v0.1.0` contains the data preprocessing module and the UAV deployment simulation. Other modules will be added in later versions.

## Files

- `data_preprocessing.py`: filters the trajectory data, determines the disaster-area boundary, selects users, and fills missing positions.
- `uav_simulation_visualization.py`: generates a four-stage illustration of the UAV deployment process.

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
