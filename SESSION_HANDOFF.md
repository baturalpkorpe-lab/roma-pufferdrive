# ROMA-PufferDrive — Session Handoff (2026-07-07)

Handoff for a fresh Claude session. Read this top-to-bottom before acting.
Everything below reflects the state at the end of a long working session.

---

## 0. What this project is

A **role-conditioned multi-agent driving policy** (ROMA idea) trained with
**PPO/RL** on PufferDrive (Waymo Open Motion data). One **parameter-shared**
policy runs all agents (decentralized execution); a per-agent **role vector**
`z_i` is the symmetry-breaker that lets the shared network express diverse
driving styles. **The role variable is the novel contribution.**

Evaluation = **WOSAC realism** (matching the human behavior distribution) +
custom **role analysis** (does the role encode meaningful, controllable
heterogeneity?).

Two isolated cluster stacks (they need different C builds):
- **baseline** stack: conda env `roma`, `/scratch/e452103/PufferDrive` (branch 2.0),
  code `~/roma_pufferdrive`. PYTHONPATH `~/roma_pufferdrive:/scratch/e452103/PufferDrive`.
- **main/guidance** stack: conda env `/scratch/e452103/envs/roma_guid`,
  `/scratch/e452103/PufferDrive_guid` (branch kj/guidance_reward), code `~/roma_main`.
  PYTHONPATH `~/roma_main:/scratch/e452103/PufferDrive_guid`.

Conda on cluster: `module load miniconda3; source /apps/generic/miniforge3/25.11.0/etc/profile.d/conda.sh; conda activate <env>`.
SLURM: `--partition=gpu-a100 --ntasks=1 --cpus-per-task=16 --mem-per-cpu=4G --gpus-per-task=1 --account=research-ceg-tp`.
wandb entity = `s-rahmani-tu-delft` (train_roma has `--wandb_entity`; eval_roma reads `$WANDB_ENTITY`). API key is in `~/.bashrc`.

---

## 1. Headline results so far (the story)

- **Roles do NOT improve WOSAC realism** (flat across dim-8/4/3), BUT they add
  **interpretable, controllable heterogeneity at no realism cost**. That is the
  safe, supported thesis (**"Option A"**).
- **Role ablation (natural / shuffled / collapsed roles, eval-time, both 3B ckpts):**
  collapsing role heterogeneity drops realism_meta ~0.69 -> ~0.60 and worsens
  collision/offroad likelihoods -> roles causally produce human-calibrated
  heterogeneity. (Note: the shuffled column + spread rows had a slot-vs-vehicle-id
  bug; WOSAC natural/collapsed columns are valid.)
- **Realism bottleneck is kinematic**: `likelihood_linear_speed ~0.15`,
  `linear_accel ~0.27`; policy **over-speeds humans by +4.4..+5.9 m/s** in every
  regime. This is the classic "RL destroys realism" problem.
- **Role space is intrinsically ~2-D** regardless of nominal dim (dim-8: 2 PCs
  ~83%; dim-4: 2 PCs ~94%; strong inter-dim correlations). Named axes via Phase C:
  a conformity<->runaway axis + a smoothness/turning axis.
- **Phase C (map-regime + GT-referenced deltas)**: roles predict how each agent
  deviates from its OWN human counterpart; variance decomposition ~30-40% of
  Δspeed/ADE variance is role-attributable within regime; associations replicate
  across regimes (no sign flips) and survive within-scene z-score controls.

---

## 2. Current runs / branch state (git @ 2026-07-07)

| branch | purpose | tip commit | run status |
|---|---|---|---|
| `baseline` | dim-8 baseline + all analysis tooling | 3bc0416 | dim-8 3B ckpt exists |
| `baseline_role_3_dim` | dim-3 ablation (role_dim default 3) | 376ae79 | **TRAINING** — check squeue |
| `baseline_role_0_dim` | NO-ROLE ablation (option A, true removal) | 7c2b483 | **TRAINING** — was ~1.5B, converged; user considered stopping but leave to 3B for budget parity |
| `baseline_role_4_dim` | dim-4 ablation | (pushed) | dim-4 3B ckpt exists |
| `main` | guidance stack code + shared tooling | f45df4f | guidance run FINISHED but FAILED (see §3) |

**Checkpoints**: `/scratch/e452103/checkpoints/roma_baseline{,_dim4,_dim3,_dim0}`,
`/scratch/e452103/checkpoints/roma_main_guidance`. dim-8 & dim-4 have
`roma_dim{8,4}_step3000238080.pt`.

**Training sbatches**: `slurm/train_dim3.sbatch` (branch baseline_role_3_dim),
`slurm/train_dim0.sbatch` (branch baseline_role_0_dim), `slurm/train_main_guidance.sbatch`
(main). All: 3B steps, mid-run WOSAC every 500M @ 10 batches, final WOSAC @ 100
batches, `--role_episodes 0`. Each runs from its OWN clone (`~/roma_dim3`,
`~/roma_dim0`, `~/roma_main`) so they don't collide.

**NOTE**: dim-0 gets **no role analysis** (no role exists) — render + WOSAC only.

---

## 3. The guidance run FAILED — root cause confirmed

The main-branch guidance run (use_guided_autonomy=1, guidance_speed_weight=1.0,
guidance_heading_weight=1.0 — **my placeholder values**) produced a **degenerate
near-stationary policy** (mean speed 0.15-0.76 m/s, mean_return -20 negative,
agents bunch up / don't respect roads).

Root cause (confirmed from kj C source + the config):
- Guidance reward ADDS to base reward; each term is `-weight*(1-exp(-error²))`
  in [-weight,0] PER STEP -> up to -2/step at weight 1.0. Waypoint progress
  reward capped at +0.1/step. **~20x imbalance** -> policy minimizes movement.
- The **heading penalty specifically punishes motion** (moving = steering error =
  penalty; freezing avoids it) -> "freeze to avoid penalty" trap.
- **The authors' own `[sweep.env.*]` block sets guidance weights to 0.05-0.3
  (mean 0.1)** — my 1.0 was ~10x their center, outside their range. That is the
  definitive cause. use_guided_autonomy default = 0 (opt-in, experimental).
- The role SURVIVED the collapse (0/8 dead dims, action-KL 0.51) — so heavy
  anchoring does not kill the role.

**Do NOT WOSAC/render this checkpoint further — it's dead.**

---

## 4. The decided direction (discussed, NOT yet implemented)

Fix realism via **imitation**, not the position-penalty guidance:
- **Guidance penalty (position deviation) = FAILED approach** (freeze trap).
- **BC / action log-likelihood** = the right form. The kj env already provides
  it: `infer_human_actions()` (inverse bicycle model -> per-step human accel/steer),
  `_prep_human_data()` (caches expert (obs,action) to disk; needs
  `prep_human_data=True`), `sample_expert_data(n)` (samples expert batches).
  **No BC loss exists in `train_roma.py` yet — it must be added.**

Framing decided with the user:
- **Main loss stays PPO. This is an RL project. BC is a small optional `λ·CE`
  direction, NOT the centerpiece. The role variable stays the novelty.**
- **Option A (core, already supported):** role-conditioned RL gives controllable
  heterogeneity at NO realism cost (dim-0 vs dim-8 WOSAC parity).
- **Option B (bonus, hypothesis):** a small BC anchor lets the role resolve
  imitation **multimodality** (humans in the same state do different things; a
  role-conditioned policy can imitate the modes where a plain policy averages) ->
  role-conditioned imitation could improve realism. Test dim-0+BC vs dim-8+BC.
- BC only applies to real Waymo logs (have demos); any authored/synthetic hard
  scenarios would be RL-only. (Scenario-based / PEGASUS hard-scenario idea =
  useful as a controlled ROLE-demonstration testbed, NOT for WOSAC realism.)

**If proceeding with BC: use the MAIN branch + guid stack** (only that C build
exposes expert actions). Need `prep_human_data=True` + the new `--bc_weight` term.

---

## 5. Key tooling built this session (all pushed)

On `baseline` (synced to `main` where noted):
- `render_topdown.py` — headless matplotlib MP4/GIF renderer (no raylib/GLX).
  Modes: free / sweep / forced. **KNOWN LIMITATION**: uses num_maps=1 which cannot
  change map -> the "n maps" may be the SAME scene repeated; also lacks the
  hide-padding-agents + GT-background fixes that render_role_conditions has
  (phantom "splitting" cars, agents ignoring roads). Port those if used seriously.
- `render_role_conditions.py` — controlled role videos: per regime, N core maps,
  focal agent forced to natural/conformist/middle/runaway cluster-mean roles;
  same scene+seed, only focal role differs. Has: goal star (= last valid GT point,
  CONFIRMED to equal the env's goalPosition), respawn logic understanding,
  GT-background for context cars, hide-no-GT padding agents, all-or-nothing scene
  writes, per-frame goal-distance CSV. sbatch: `slurm/render_role_conditions.sbatch`
  (3072 agents / 128 maps).
- `map_features.py` + `map_atlas.py` (+ `slurm/map_atlas.sbatch`) — Phase A/B:
  per-scenario features -> K=5 map regimes (C0 quiet_local, C1 fast_corridors,
  C2 parking_low_speed, C3=JUNK/exclude, C4 congested_urban). Output
  `/scratch/e452103/map_atlas/map_clusters.csv`.
- `role_regime_analysis.py` (+ `slurm/role_regime.sbatch`) — Phase C: GT-referenced
  deltas per regime, correlation/replication/within-scene/variance/spider figures,
  role-axis dose-response, `--replot` mode (rebuild figures from CSV, no GPU).
- `role_ablation_eval.py` (+ `slurm/role_ablation.sbatch`) — natural/shuffled/
  collapsed role ablation with WOSAC + spread ratio + wandb logging.
- `bench_env_sps.py` (+ sbatch) — raw env SPS benchmark (diagnoses build regressions).

**Env facts learned (important, non-obvious):**
- `Drive(num_maps=1)` loads ONE fixed scenario; neither the seed nor
  `resample_maps()` can change it. To vary scenes, load a POOL (num_maps>1) and
  filter by scenario_id in the live allocation.
- `env.reset()` RESHUFFLES the slot<->scenario allocation; identify a focal by
  scenario_id + GT vehicle `id`, not slot index.
- Goal (goal_behavior=0) = each object's `goalPosition` in the JSON = its last
  valid GT point (CONFIRMED equal). Respawn condition (from C source):
  `distance_to_goal < goal_radius (2m) AND current_speed <= goal_speed`. The env
  default `goal_speed` gate (~20 m/s) blocks FAST agents from ever respawning even
  at 0 m — set `goal_speed=100` in render envs to disable it (already done in
  render_role_conditions). goal_radius=2m is small; fast/deviating agents step over it.
- SPS regression (55->35k) was the Jun-29 `-O2` rebuild; fixed with
  `-O3 -mavx2 -mfma -fopenmp` (build_binding.sh updated all branches). `-mavx2`
  not `-march=native` (same .so must run on Intel compute + AMD gpu-a100).
- The guid stack needed its OWN map binaries generated (different binary format;
  segfaults on baseline binaries). Generated via `process_all_maps` run from the
  PufferDrive_guid ROOT dir (NOT the pufferlib subdir — module shadowing).
- eval WOSAC on guid stack needs `pip install seaborn` in roma_guid.

---

## 6. STEP-BY-STEP WORKPLAN for the next session

Do these roughly in order. Confirm each command's paths before running.

### Step 1 — Finish & harvest the running ablations (dim-3, dim-0)
1. `squeue -u e452103` — confirm dim-3 and dim-0 are still training (or done).
2. **Let dim-0 finish to 3B** (budget parity with dim-8/4/3 for the ablation table).
   It converged by ~1B but equal steps kills the "trained longer" confound.
3. When each finishes, its `--run_eval` logs final WOSAC to wandb. Collect
   `eval/wosac_realism_score`, `likelihood_linear_speed`, `likelihood_collision`
   for dim-0/3/4/8 into ONE table. **This is the core Option-A result.**
4. dim-3/4: also run Phase C role analysis on their 3B checkpoints (map_atlas
   already exists; `role_regime_analysis.py` + `slurm/role_regime.sbatch`, point
   `--checkpoint` at each). dim-0: **skip role analysis** (no role).

### Step 2 — Assemble the Option-A paper result (no new training)
1. Four-way WOSAC table (dim-0/3/4/8) + param counts (dim-0=144,988; dim-4=180,260;
   compute dim-3/8) to preempt the "capacity not roles" objection.
2. Role-dimensionality figure: PCA %-variance + inter-dim correlation for dim-3/4/8
   (do 2 PCs still explain ~85-94%? does dim-3 reduce correlation?).
3. Re-run `role_ablation_eval.py` if you want the corrected shuffled/spread numbers
   (the reshuffle fix is in render_role_conditions but VERIFY it's in the ablation
   too; the WOSAC natural/collapsed columns were already valid).
4. Fix/rerun `render_role_conditions.sbatch` for clean role-condition videos of the
   dim-4 (or dim-8) checkpoint (goal star + GT background + natural condition all in).

### Step 3 — (Optional, Option-B bonus) BC / imitation experiment on MAIN branch
Only if Option A is solid and you want the realism-via-roles bonus claim.
1. Read how kj computes expert actions: `infer_human_actions`, `sample_expert_data`,
   `_prep_human_data` in `/scratch/e452103/PufferDrive_guid/pufferlib/ocean/drive/drive.py`.
2. Turn ON `prep_human_data=True` and run the one-time prep (caches
   `expert_actions_discrete_h32.pt` etc.). Confirm the `.pt` files appear.
3. Implement `--bc_weight` in `~/roma_main/roma_pufferdrive/train_roma.py`:
   each PPO update, `sample_expert_data(n)`, forward expert obs through the policy
   (role-conditioned, role inferred from expert obs), add
   `bc_weight * CrossEntropy(logits, expert_discrete_action)` to the loss.
   Keep bc_weight MODEST (start ~0.1-0.5; it's a regularizer, not the objective).
4. Run TWO short (500M-1B) sanity runs: **dim-8 + BC** and **dim-0 + BC**.
   Watch `mean_return` stays POSITIVE and mean speed becomes human-like (several m/s).
5. Compare WOSAC: does dim-8+BC beat dim-0+BC on realism (esp. interactive/
   multimodal metrics)? That tests "roles resolve multimodality."
6. **Verify the role SURVIVES BC** (dead dims, action-KL, Phase C) — same checks
   used on the guidance checkpoint.

### DO-NOT / gotchas for the next session
- Do NOT set guidance_speed/heading_weight to 1.0 — authors' range is 0.05-0.3.
  (We're pivoting to BC anyway; only relevant if someone revisits guidance.)
- Do NOT WOSAC/render the failed guidance checkpoint.
- Do NOT use `num_maps=1` expecting different maps.
- BC needs the MAIN branch + guid stack (only build with expert actions).
- Always confirm the guid binding is the `-O3` build (SPS ~55-76k, not ~35k).
- When rendering, remember goal_speed=100 to see respawns; goal=last valid GT point.

---

## 7. Open threads / decisions pending
- **A-core vs B-bonus**: keep PPO+roles as the thesis core (A); BC is optional (B).
  User leaning A-core, B as bonus if time.
- Whether to train a dim-2 (2 PCs explain most variance) — parked, only if asked.
- Guid stack `PufferDrive_guid` binding: confirm it's the `-O3` rebuild.
- The `human_ll_coef` exact wiring (how the discrete expert action indexes the
  policy's 91-way action head) — read before implementing BC.
