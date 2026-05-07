# ROMA on PufferDrive

This repository adapts **ROMA (Multi-Agent Reinforcement Learning with Emergent Roles)** (Wang et al., ICML 2020) to the **PufferDrive** autonomous driving simulator, built on 10,000 real-world driving scenarios from the Waymo Open Motion Dataset.

The core research question is: **can multidimensional role representations improve realism and diversity in multi-agent autonomous driving?**

---

## Motivation

In real traffic, drivers have distinct behavioural styles — some are aggressive, some are cautious, some are cooperative. Standard multi-agent RL treats all agents identically, producing a population of clones that fails to capture this natural heterogeneity.

We confirmed this empirically: the Waymo dataset has a speed coefficient of variation of **1.06** and a steering CV of **1.19**, meaning variation between drivers exceeds the average behaviour itself. This directly justifies the ROMA diversity loss.

ROMA gives each agent a latent role vector that emerges from training without supervision, conditioning every decision on its learned identity. We extend this from a scalar role (dim=1, original ROMA) to a multidimensional role vector where each dimension captures a separate behavioural factor.

---

## Repository Structure

```
roma_pufferdrive/
├── roma/
│   ├── __init__.py          — Package init
│   ├── role_encoder.py      — GRU-based role encoder
│   ├── aux_losses.py        — MI loss + cosine diversity loss
│   └── policy.py            — Structured encoder policy (ego/partner/road)
├── train_roma.py            — Full ROMA training (GPU/CPU, wandb, CSV logging)
├── train_baseline.py        — Vanilla PPO baseline (no roles)
├── eval_roma.py             — Evaluation: env metrics + WOSAC realism
├── show_diversity.py        — Waymo dataset diversity analysis
├── visualize_policy.py      — Role vector vs behaviour visualization
├── check_env.py             — Environment diagnostic (run first)
├── test_modules.py          — Unit tests (13 tests, run before training)
└── README.md
```

---

## Architecture

The observation (1121-dimensional flat vector) is split into three structured slices and processed by separate encoders:

```
obs (1121)
├── ego      [0:7]      → EgoEncoder     → 32 dims
├── partners [7:224]    → PartnerEncoder → 32 dims  (max-pool over 31 vehicles)
└── roads    [224:1120] → RoadEncoder    → 64 dims  (max-pool over 128 points)
                               ↓
                       env_embedding (128)
                               ↓
              ┌────────────────┤
              ↓                ↓
        Role Encoder      env_embedding
        GRU(64→64)
        mu_head → role_mean
        logvar_head → role_var
        reparameterize → role_z
              ↓                ↓
              └──── concat ────┘
                       ↓
              [env_embedding || role_z]
                (128 + role_dim dims)
                       ↓
               Policy GRUCell (128)
                       ↓
              Actor → 91 action logits
              Critic → 1 value estimate
```

**Training loss:**
```
Total = PPO loss + 1.0 × MI loss + 0.1 × Diversity loss
```

**Diversity loss** uses pairwise cosine similarity between role means — naturally bounded in [-1, +1]:
- `+1.0` = all agents have identical roles (collapse)
- ` 0.0` = roles uncorrelated
- `-1.0` = agents maximally diverse

This replaces the original KL-based diversity which diverged to -infinity without clipping.

---

## Installation

### 1. Clone and install PufferDrive

```bash
git clone https://github.com/Emerge-Lab/PufferDrive.git
cd PufferDrive
uv venv .venv && source .venv/bin/activate
uv pip install -e .
python setup.py build_ext --inplace --force
```

### 2. Download the Waymo dataset (~4.8 GB)

```bash
huggingface-cli download daphne-cornelisse/pufferdrive_womd_train \
  --repo-type dataset \
  --local-dir data/processed/training
unzip data/processed/training/training.zip -d data/processed/training/
```

### 3. Convert maps to binary format

```bash
python pufferlib/ocean/drive/drive.py
mkdir -p resources/drive/binaries/training
mv resources/drive/binaries/*.bin resources/drive/binaries/training/
```

### 4. Clone this repo into PufferDrive

```bash
git clone https://github.com/baturalpkorpe-lab/roma-pufferdrive.git roma_pufferdrive
```

### 5. Install dependencies

```bash
uv pip install matplotlib scikit-learn scipy wandb pandas
```

### 6. Verify environment

```bash
cd ~/PufferDrive
PYTHONPATH=/root/PufferDrive/roma_pufferdrive python3 roma_pufferdrive/check_env.py
```

Expected output: `obs_dim = 1121`

### 7. Run unit tests

```bash
PYTHONPATH=/root/PufferDrive/roma_pufferdrive python3 roma_pufferdrive/test_modules.py
```

All 13 tests must pass before training.

---

## Training

### CPU (quick test — ~2M steps in ~15 minutes)

```bash
cd ~/PufferDrive
PYTHONPATH=/root/PufferDrive/roma_pufferdrive python3 roma_pufferdrive/train_roma.py \
    --role_dim 1 \
    --num_maps 100 \
    --total_steps 2000000 \
    --num_agents 16 \
    --device cpu \
    --save_interval 1000000 \
    --save_dir roma_pufferdrive/checkpoints/roma_dim1_test
```

### GPU / Supercomputer (full training — 6B steps)

```bash
PYTHONPATH=/path/to/roma_pufferdrive python3 roma_pufferdrive/train_roma.py \
    --role_dim 1 \
    --num_maps 10000 \
    --total_steps 6000000000 \
    --save_interval 500000000 \
    --num_agents 64 \
    --device cuda \
    --run_eval \
    --wandb_project roma-pufferdrive \
    --wandb_entity YOUR_WANDB_USERNAME \
    --save_dir roma_pufferdrive/checkpoints/roma_dim1
```

### With wandb offline (no internet on supercomputer)

```bash
PYTHONPATH=/path/to/roma_pufferdrive python3 roma_pufferdrive/train_roma.py \
    --role_dim 1 \
    --num_maps 10000 \
    --total_steps 6000000000 \
    --num_agents 64 \
    --device cuda \
    --run_eval \
    --wandb_offline \
    --wandb_project roma-pufferdrive \
    --wandb_entity YOUR_WANDB_USERNAME \
    --save_dir roma_pufferdrive/checkpoints/roma_dim1
```

Sync later with `wandb sync`.

### Without wandb (CSV only)

Simply omit `--wandb_project`. Training log is always saved to `checkpoints/training_log.csv` regardless.

### Key training arguments

| Argument | Default | Description |
|---|---|---|
| `--role_dim` | 1 | Role vector dimension. 1=original ROMA, 8=proposed | (We want to see the results in dimension 1 initially)
| `--num_maps` | 10000 | Scenarios loaded. 10000=full, 100=CPU test |
| `--num_agents` | 64 | Agents per scene. 64 on GPU, 16 on CPU |
| `--total_steps` | 1B | Training steps. Use 6B for full run |
| `--device` | cuda | cuda or cpu. Auto-falls back to cpu |
| `--mi_weight` | 1.0 | Weight on MI loss | (Can be changed later according to the results)
| `--div_weight` | 0.1 | Weight on cosine diversity loss | (Can be changed later according to the results)
| `--run_eval` | False | Run WOSAC + env evaluation after training |
| `--wandb_project` | None | Wandb project name. Omit to disable wandb |
| `--wandb_entity` | None | Wandb username |
| `--wandb_offline` | False | Log locally, sync later |
| `--save_interval` | 100M | Save checkpoint every N steps |

---

## Evaluation

### Quick evaluation (CPU)

```bash
PYTHONPATH=/root/PufferDrive/roma_pufferdrive python3 roma_pufferdrive/eval_roma.py \
    --checkpoint roma_pufferdrive/checkpoints/roma_dim1/roma_dim1_final.pt \
    --role_dim 1 --obs_dim 1121 --n_episodes 10 \
    --wosac --wosac_rollouts 4 --wosac_num_maps 100 --wosac_max_batches 30
```

### Full evaluation (GPU / supercomputer — all 10,000 scenarios)

```bash
PYTHONPATH=/path/to/roma_pufferdrive python3 roma_pufferdrive/eval_roma.py \
    --checkpoint roma_pufferdrive/checkpoints/roma_dim1/roma_dim1_final.pt \
    --role_dim 1 --obs_dim 1121 --n_episodes 30 \
    --wosac --wosac_rollouts 32 --wosac_num_maps 10000 --wosac_max_batches 500
```

### WOSAC metrics explained

| Metric | Description |
|---|---|
| Realism meta-score | Weighted combination of all metrics. Higher = more realistic |
| Kinematic metrics | Speed, acceleration, steering match real human distributions |
| Interactive metrics | Distance to other vehicles, time-to-collision match real drivers |
| Map-based metrics | Road adherence matches real drivers |
| minADE | Minimum average displacement error vs ground truth (metres) |

---

## Baseline

```bash
PYTHONPATH=/root/PufferDrive/roma_pufferdrive python3 roma_pufferdrive/train_baseline.py \
    --num_maps 10000 --num_agents 64 --total_steps 6000000000 --device cuda \
    --save_dir roma_pufferdrive/checkpoints/baseline
```

---

## Dataset Diversity Analysis

To verify that the Waymo dataset contains genuine behavioural diversity justifying the diversity loss:

```bash
PYTHONPATH=/root/PufferDrive/roma_pufferdrive python3 roma_pufferdrive/show_diversity.py \
    --num_maps 1000 --num_resets 100
```

Results: speed CV=1.06, steering CV=1.19 — variation between drivers exceeds the mean. Full analysis saved to `diversity_plots/diversity_overview.png`.

---

## Visualize Policy

After training, visualize agent behaviour and role vectors:

```bash
PYTHONPATH=/root/PufferDrive/roma_pufferdrive python3 roma_pufferdrive/visualize_policy.py \
    --checkpoint roma_pufferdrive/checkpoints/roma_dim1/roma_dim1_final.pt \
    --role_dim 1 --obs_dim 1121
```

Produces:
- `behaviour_over_time.png` — speed and steering for all agents over 91 timesteps
- `role_vs_behaviour.png` — each agent's role vector alongside their behavioural statistics

---

## Logged Metrics

Training automatically saves to `checkpoints/training_log.csv`:

| Column | Description |
|---|---|
| step | Global training step |
| sps | Steps per second |
| policy_loss | PPO loss (~0, fluctuates around 0 normally) |
| value_loss | Critic loss (should decrease over time) |
| mi_loss | Mutual information loss (~0.007, stable) |
| div_loss | Diversity loss (starts ~1.0, decreases toward 0 as roles diverge) |
| score | Rolling average episode score (0-1, higher is better) |
| mean_return | Rolling average episode return |

If `--run_eval` is set, after training also saves:
- `eval_env_metrics.csv` — per-episode environment metrics
- `eval_wosac_metrics.csv` — per-scenario WOSAC realism metrics

---

## Notes

- PufferDrive requires Linux — native Windows not supported. Use WSL2.
- obs_dim is auto-detected at startup — no need to hardcode
- Run `check_env.py` first when setting up on a new machine
- Run `test_modules.py` before every training run
- div_loss naturally bounded in [-1, +1] — no clipping needed
- wandb is fully optional — CSV logging always runs

---

## Citation

```bibtex
@inproceedings{wang2020roma,
  title={ROMA: Multi-Agent Reinforcement Learning with Emergent Roles},
  author={Wang, Tonghan and Dong, Heng and Lesser, Victor and Zhang, Chongjie},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2020}
}
```

Built on:
- [PufferDrive](https://github.com/Emerge-Lab/PufferDrive) — driving simulator
- [Waymo Open Motion Dataset](https://waymo.com/open/data/motion/)
