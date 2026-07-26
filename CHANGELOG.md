# Changelog

## [0.8.0] - Unreleased

- Updated the MAS density, safety, movement, and overlap weights.
- Updated the default PSO swarm size, iteration count, inertia, and learning
  coefficients.
- Kept the existing coverage radius, communication range, safety distance,
  movement limit, and warm-start settings unchanged.

## [0.7.0] - 2026-07-24

- Added per-time-slice and aggregate runtime measurements for all algorithms.
- Added UAV position, time-slice summary, and cross-algorithm comparison CSV exports.
- Added one coverage comparison figure for MAS-PSO and all four baselines.
- Added one UAV movement comparison figure for all non-static algorithms.
- Added representative MAS-PSO time-slice selection by minimum coverage, maximum
  UAV movement, and maximum coverage.
- Added representative deployment and constraint comparison figures.
- Added automated tests for runtimes, exported tables, and generated figures.

## [0.6.0] - 2026-07-23

- Added one shared experiment configuration for data and algorithm settings.
- Added a single data preparation path for all deployment algorithms.
- Added fixed-order execution of all five algorithms with the same inputs and seed.
- Added an in-memory experiment result with algorithm lookup and summary metrics.
- Added deterministic tests for the complete experiment runner.

## [0.5.0] - 2026-07-22

- Added Random Deployment, K-means, Standard PSO, and Static PSO baselines.
- Added shared result structures and average coverage and fitness calculations.
- Added minimum-movement UAV ID alignment between time slices.
- Added deterministic tests for all baseline workflows.

## [0.4.0] - 2026-07-21

- Added consecutive time-slice MAS-PSO optimization.
- Added warm-start particles around the preceding best UAV deployment.
- Added random restart particles to retain global exploration.
- Added per-time-slice results and average coverage and fitness metrics.
- Added deterministic tests for the dynamic optimization workflow.

## [0.3.0] - 2026-07-21

- Added a standard PSO optimizer for one UAV deployment time slice.
- Represented each particle as a joint deployment of all UAVs.
- Added personal-best and global-best velocity updates with boundary handling.
- Added preliminary PSO defaults and deterministic unit tests.

## [0.2.0] - 2026-07-20

- Added shared UAV state and communication adjacency calculations.
- Added coverage, density, safety, movement, boundary, and overlap metrics.
- Added preliminary round weights for the reward and penalty terms.
- Added deterministic tests for the MAS coordination module.

## [0.1.0] - 2026-07-19

- Added the data preprocessing module.
- Added the UAV deployment simulation.
- Added the project README, dependencies, and Git ignore rules.
