# ROMA on PufferDrive

Implementation of ROMA (Multi-Agent Reinforcement Learning with Emergent Roles) adapted for the PufferDrive autonomous driving simulator.

## Overview

This project bridges two paradigms:
- **ROMA** (Wang et al., ICML 2020) — role-based multi-agent RL with emergent roles
- **PufferDrive** — high-performance driving simulator built on 10,000 real Waymo scenarios

Each agent learns a latent role vector that conditions its driving policy, shaped by:
- **Mutual Information loss** — forces role to encode meaningful behaviour
- **Diversity loss** — prevents all agents from collapsing to the same role

## Repository Structure
## Installation

### 1. Clone PufferDrive
```bash
git clone https://github.com/Emerge-Lab/PufferDrive.git
cd PufferDrive
uv venv .venv && source .venv/bin/activate
uv pip install -e .
python setup.py build_ext --inplace --force
```

### 2. Download Waymo dataset
```bash
huggingface-cli download daphne-cornelisse/pufferdrive_womd_train \
  --repo-type dataset \
  --local-dir data/processed/training
```

### 3. Convert maps to binary
```bash
python pufferlib/ocean/drive/drive.py
```

### 4. Clone this repo into PufferDrive
```bash
git clone https://github.com/YourUsername/roma-pufferdrive.git roma_pufferdrive
```

### 5. Install extra dependencies
```bash
pip install matplotlib scikit-learn scipy wandb
```

## Usage

### Run unit tests first
```bash
cd roma_pufferdrive
python3 test_modules.py
```

### Train vanilla PPO baseline
```bash
cd ~/PufferDrive
python3 roma_pufferdrive/train_baseline.py
```

### Train ROMA (original scalar role)
```bash
python3 roma_pufferdrive/train_roma.py --role_dim 1
```

### Train ROMA (proposed multidimensional role)
```bash
python3 roma_pufferdrive/train_roma.py --role_dim 8
```

## Results (50M steps, CPU)

| Metric | Baseline | ROMA dim=1 | ROMA dim=8 |
|---|---|---|---|
| Score | 0.0229 | 0.0021 | 0.0063 |
| Completion rate | 0.0521 | 0.1708 | - |
| Mean return | -2.3553 | -2.1681 | - |

Note: These are preliminary results at 50M steps on CPU.
Full GPU training (500M+ steps) is expected to show stronger improvements.

## Architecture
Training loss:
Total = PPO loss + 1.0 × MI loss + 0.05 × Diversity loss
rm ~/PufferDrive/roma_pufferdrive/README.md
ls ~/PufferDrive/roma_pufferdrive/
bashcat > ~/PufferDrive/roma_pufferdrive/README.md << 'EOF'
# ROMA on PufferDrive

This repository adapts **ROMA (Multi-Agent Reinforcement Learning with Emergent Roles)** (Wang et al., ICML 2020) to the **PufferDrive** autonomous driving simulator, built on 10,000 real-world driving scenarios from the Waymo Open Motion Dataset. The goal is to prove that role-based multi-agent reinforcement learning can be successfully applied to heterogeneous autonomous driving environments, serving as a foundation for extending ROMA with multidimensional disentangled role representations.

---

## Motivation

In real traffic, drivers have distinct behavioural styles — some are aggressive, some are cautious, some are cooperative. Standard multi-agent RL treats all agents identically, producing a population of clones that fails to capture this natural heterogeneity. ROMA addresses this by giving each agent a latent role vector that emerges from training without any manual supervision, conditioning every decision the agent makes on its learned identity.

---

## Repository Structure
roma_pufferdrive/
├── roma/
│   ├── init.py        # Package init
│   ├── role_encoder.py    # GRU-based role encoder (ROMA core)
│   ├── aux_losses.py      # MI loss + diversity loss
│   └── policy.py          # Full ROMA policy network
├── train_roma.py          # Train ROMA (--role_dim 1 or --role_dim 8)
├── train_baseline.py      # Train vanilla PPO baseline (no roles)
├── eval_roma.py           # Evaluate trained checkpoints
├── test_modules.py        # Unit tests (run before training)
└── README.md

---

## Architecture

Each agent receives a 1121-dimensional observation vector from PufferDrive at every timestep, encoding its own state, up to 31 surrounding vehicles, and nearby road geometry. This observation is processed through two parallel branches. The first is the ROMA role encoder — a GRU-based recurrent network that produces a latent role distribution parameterized by a mean and variance vector. The role vector is sampled from this distribution during training using the reparameterization trick, and set deterministically to the mean during evaluation. The second branch is a feedforward observation encoder that produces a 128-dimensional environment embedding. The role vector and environment embedding are concatenated and passed into the policy GRU, which outputs action logits over 91 discrete actions and a scalar value estimate.
observation (1121)
↓                      ↓
Role Encoder             Obs Encoder
Linear(1121→64)          Linear(1121→256) ReLU
ReLU                     Linear(256→128)  ReLU
GRUCell(64→64)
mu_head → role_mean          env_embedding (128)
logvar_head → role_var
reparameterize → role_z
↓                      ↓
└──────── concat ───────┘
↓
[env_embedding || role_z]
128 + role_dim numbers
↓
Policy GRUCell
↓
Actor  → 91 action logits
Critic → 1 value estimate

Training objective:
Total Loss = PPO Loss
+ 1.0  × MI Loss
+ 0.05 × Diversity Loss

Role dimensions:
role_dim=1 → [0.73]                                          (original ROMA)
role_dim=8 → [0.73, 0.12, 0.45, 0.89, 0.23, 0.67, 0.34, 0.91]  (proposed)

---

## Installation

### 1. Install WSL2 (Windows only)
```powershell
wsl --install
```

### 2. Clone and install PufferDrive
```bash
git clone https://github.com/Emerge-Lab/PufferDrive.git
cd PufferDrive
uv venv .venv && source .venv/bin/activate
uv pip install -e .
python setup.py build_ext --inplace --force
```

### 3. Download the Waymo dataset (~4.8 GB)
```bash
huggingface-cli download daphne-cornelisse/pufferdrive_womd_train \
  --repo-type dataset \
  --local-dir data/processed/training
unzip data/processed/training/training.zip -d data/processed/training/
```

### 4. Convert JSON maps to binary format
```bash
python pufferlib/ocean/drive/drive.py
mkdir -p resources/drive/binaries/training
mv resources/drive/binaries/*.bin resources/drive/binaries/training/
```

### 5. Clone this repo into PufferDrive
```bash
git clone https://github.com/YourUsername/roma-pufferdrive.git roma_pufferdrive
cd roma_pufferdrive
```

### 6. Install extra dependencies
```bash
pip install matplotlib scikit-learn scipy wandb
```

---

## Usage

### Run unit tests first (always do this before training)
```bash
cd ~/PufferDrive/roma_pufferdrive
python3 test_modules.py
```

### Train vanilla PPO baseline
```bash
cd ~/PufferDrive
python3 roma_pufferdrive/train_baseline.py
```

### Train ROMA with scalar role (original paper setting)
```bash
python3 roma_pufferdrive/train_roma.py --role_dim 1
```

### Train ROMA with multidimensional role (proposed extension)
```bash
python3 roma_pufferdrive/train_roma.py --role_dim 8
```

### Key training arguments
--role_dim       int    Role vector dimension (default 8)
--mi_weight      float  Weight on MI loss (default 1.0)
--div_weight     float  Weight on diversity loss (default 0.05)
--num_agents     int    Agents per scene (default 16)
--total_steps    int    Total training steps (default 50000000)
--num_maps       int    Number of maps to use (default 100)
--seed           int    Random seed (default 0)

---

## Results

Preliminary results after 50M steps on CPU (no GPU):

| Metric | Baseline | ROMA dim=1 | ROMA dim=8 |
|---|---|---|---|
| Score | 0.0229 | 0.0021 | 0.0063 |
| Collision rate | 0.3875 | 0.4292 | - |
| Off-road rate | 0.8583 | 1.0000 | - |
| Completion rate | 0.0521 | 0.1708 | - |
| Mean return | -2.3553 | -2.1681 | - |

Note: These are proof-of-concept results at 50M steps on CPU. ROMA requires more training steps than the baseline because it simultaneously learns to drive, what its role means, and how to differ from other agents. Full GPU training at 500M+ steps is expected to show the complete benefit of role representations. The completion rate improvement from 5.2% to 17.1% with ROMA dim=1 is a promising early signal.

---

## What This Code Is and Is Not

This repository is a **proof of concept** demonstrating that the ROMA architecture is compatible with and trainable on PufferDrive. It is not the final research implementation. The following extensions are planned for the full research phase:

- **Disentangled role dimensions** with beta-VAE regularization so each dimension captures a separate interpretable behavioural factor
- **Semantic alignment losses** that explicitly correlate role dimensions with observable behaviours such as speed, lane-change frequency, and time headway
- **PufferDrive official encoders** replacing the simple MLP with proper ego, partner, and road sub-encoders for permutation-invariant observation processing
- **Full GPU training** at 500M-1B steps for complete learning curves
- **Interpretability analysis** including t-SNE visualizations of the role space and Pearson correlation analysis between role dimensions and driving behaviours
- **Full ablation study** systematically removing components to quantify each contribution

---

## Citation

If you use this code please cite the original ROMA paper:

```bibtex
@inproceedings{wang2020roma,
  title={ROMA: Multi-Agent Reinforcement Learning with Emergent Roles},
  author={Wang, Tonghan and Dong, Heng and Lesser, Victor and Zhang, Chongjie},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2020}
}
```

This code is built on top of:
- [ROMA](https://github.com/TonghanWang/ROMA) — original ROMA implementation
- [PufferDrive](https://github.com/Emerge-Lab/PufferDrive) — driving simulator

---

## Notes

- All experiments were run on WSL2 Ubuntu with CPU only due to hardware constraints
- PufferDrive requires Linux or WSL2 — native Windows is not supported
- The observation dimension (1121) may differ from PufferDrive documentation (1848) depending on simulator configuration
- Run test_modules.py before any training to verify all components are working correctly
