# data/ — committed data artifacts

Small generated data files that the analysis needs but **cannot be regenerated
without the raw Waymo feature cache**, so they ship with the repo and travel via
`git pull` instead of being copied between accounts by hand.

## trajectory_clusters_stopfrac.csv

The frozen stop_frac trajectory K-means clustering:

    scenario_id, vehicle_id, cluster [, margin, cluster2, is_edge]

`cluster`: 0 = mid-speed, 1 = fast, 2 = stop&go, 3 = turning (verify against the
K=4 centroid plot, never by label alone).

Every trajectory-cluster analysis is keyed to this file. `slurm/nodiv_analysis.sbatch`
and `slurm/render_grid_traj.sbatch` use it directly; they look for a per-account
copy at `$SCRATCH_ROOT/traj_atlas/trajectory_clusters_stopfrac.csv` first and fall
back to this committed copy, so an account that never ran the clustering pipeline
still finds it after a pull.

Produced originally by the trajectory-clustering pipeline
(`trajectory_features.py` → cache → `trajectory_atlas.py`) on the analysis
account; it is placed here verbatim.
