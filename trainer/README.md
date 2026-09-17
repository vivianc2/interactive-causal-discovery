# trainer/ — SkyRL GRPO + LoRA training

An out-of-tree [SkyRL](https://github.com/NovaSky-AI/SkyRL) integration that trains a policy
on the id-space RL environment in `../rl_env/`. No fork of SkyRL / skyrl-gym is needed — this
package is symlinked into `SkyRL/examples/train/rpg` (see `../SETUP.md`).

## Files

| File | Role |
|---|---|
| `run_rpg.sh` | Launcher. Sets `RPG_SRC` / `RPG_PROTO`, model, GPUs, then runs `main_rpg` with all GRPO + LoRA + vLLM flags. |
| `main_rpg.py` | SkyRL entry point: builds the config, registers the `rpg` env, and runs the PPO/GRPO experiment. |
| `env.py` | `RPGSkyEnv` — a thin `BaseTextEnv` wrapper over `rl_env/RPGEnv`; rebuilds each world deterministically from its `(seed, skin, archetype)`. |
| `rpg_dataset.py` | Builds the SkyRL `train.parquet` / `validation.parquet` (one audited world per row). |

## Environment variables

- `RPG_SRC` — repository root (holds `benchmark/` and `rl_env/`). Used to put both on
  `sys.path`, including in Ray workers that run from a copied working dir. Default assumes
  the container layout in `SETUP.md`.
- `RPG_PROTO` — the world-generator package name (default `benchmark`); must match the
  generator the dataset was built with.
- `DATA_DIR`, `MODEL`, `NUM_GPUS`, `LOGGER` — see the header of `run_rpg.sh`.

## Running

Training runs **inside the SkyRL container** on a multi-GPU box. Follow `../SETUP.md` end to
end (clone SkyRL at the pinned commit, symlink this package, install the venv, point
`DATA_DIR` at the committed `benchmark/data`, launch `run_rpg.sh`).
