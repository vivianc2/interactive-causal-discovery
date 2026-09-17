# Interactive Causal Discovery with a Small LLM

This repository contains an **interactive causal-discovery benchmark** and the
**reinforcement-learning training/evaluation stack** used to study whether a small
(9B) language model can act like a *scientist*: choose what to measure, intervene on a
system, and identify a hidden cause under confounding — where the most obvious signal is
a trap that correlates with the outcome but does nothing when acted upon.

Most "causal" LLM benchmarks are *observational* — the model reads a described graph and
answers, which skips the hard part. Here every task is **interactive**: the model must
*discover* structure by acting (measure → intervene → answer), because observation alone
cannot separate correlation from causation.

## What's in each world

Each world is a **structural causal model** with:

- a **hidden cause** that actually drives the outcome,
- **decoys** — variables that correlate with the outcome but have *zero* interventional effect,
- a measurable **proxy/surrogate** metric (sometimes gameable),
- **intervenable knobs** (`do(·)`).

Every entity is shown as an **opaque identifier** (`m0`, `a3`, …), so the model cannot
pattern-match on names — it must run experiments. A **generate → audit → reject** loop runs
five independent audits on every candidate world and discards any that a shortcut could
solve, so each accepted world is "hard for the right reason." Scoring is a **deterministic,
rule-based reward** with no model in the evaluation path.

The primary metric is **utility recovered | completed** — the fraction of the achievable
causal fix that a chosen intervention recovers, scored on episodes that submitted an answer.
Comparisons are paired against the same base model on the same held-out worlds, with
bootstrap confidence intervals, and the headline metric is decomposed into
`completion-rate × quality` so that "the model learned to submit an answer" cannot
masquerade as skill.

## Repository layout

```
interactive-causal-discovery/
├── benchmark/    World generator, oracle, 5 audits, the SCM engine, and the
│                 free-text scientist runner. Ships the de-leaked dataset in data/.
├── rl_env/       Id-space RL environment (opaque-id catalog the policy sees),
│                 the deterministic reward, and the RL-comparable evaluation.
├── trainer/      SkyRL GRPO + LoRA launcher that trains a policy on the env.
├── requirements.txt   Runtime dependencies for benchmark/ and rl_env/.
└── SETUP.md      Step-by-step to reproduce the GPU training box (SkyRL container).
```

Each directory has its own `README.md` with the details.

### A note on file naming
Some modules carry `v6`/`v7` suffixes (`oracle_v6.py`, `sim_v6.py`, `generate_v7.py`,
`run_agent_v6.py`). Those numbers name the **design generation of that component**, carried
forward unchanged — they are **not** a dataset version. These are the current, live files.

## The pipeline

```
generate_v7.py            rpg_dataset.py                 run_rpg.sh                 id_space_eval.py
 (audited worlds)  ─────▶  (build train/val parquet) ──▶ (SkyRL GRPO + LoRA) ──▶   (RL-comparable eval)
 benchmark/                trainer/                       trainer/                  rl_env/
        │                                                                          run_agent_v6.py
        └────────────── dump_worlds.py ─────────────────────────────────────────▶ (free-text eval)
                        (parquet → the exact world JSONs)                          benchmark/
```

1. **Generate** audited worlds with `benchmark/generate_v7.py` (writes `world_*.json` +
   `manifest.json`). The canonical **de-leaked dataset is already committed** at
   `benchmark/data/{train,validation}.parquet` (**1,536 train / 128 held-out worlds**), so
   this step is only needed to build *more* worlds.
2. **Build the RL dataset** — `trainer/rpg_dataset.py` renders worlds into SkyRL prompt rows.
   (Optional; the committed parquet is ready to train on.)
3. **Train** — SkyRL GRPO + LoRA via `trainer/run_rpg.sh`. See `SETUP.md`.
4. **Evaluate**
   - **RL-comparable, id-space:** `rl_env/id_space_eval.py` (the opaque-id catalog the
     policy sees; GPU-free, API model over saved worlds).
   - **Free-text / API:** `benchmark/run_agent_v6.py` (natural-language scientist with a
     resolver mapping free text to the hidden variables).
   - `benchmark/dump_worlds.py` reconstructs the *exact* world JSONs behind a parquet split,
     so both evaluators can run on precisely the worlds the RL run trains/evals against.

## Quickstart (offline, no GPU or API key)

```bash
pip install -r requirements.txt   # numpy, pandas, scipy

# 1) Generate a few audited worlds
cd benchmark
python generate_v7.py --outdir /tmp/worlds --n 8 --seed 100000

# 2) Reconstruct the committed held-out worlds from the shipped parquet
python dump_worlds.py data/validation.parquet /tmp/heldout_worlds

# 3) Run the reward-integrity and env/parser regression checks
python test_reward_integrity.py
cd ../rl_env && PYTHONPATH=../benchmark:. python test_env_reward.py
PYTHONPATH=../benchmark:. python test_parse_regression.py
```

Running an agent against a world (`benchmark/run_agent_v6.py --world-file …`) or the
id-space eval (`rl_env/id_space_eval.py`) requires model-backend credentials — see
**Credentials** below. Use `--backend mock` with `run_agent_v6.py` to exercise the loop
with no API access.

## The recipe study

The study compares three ways to give a 9B model this skill:

- **Reinforcement learning (GRPO)** — train the policy on its own successful rollouts.
  *This repository contains the full GRPO stack* (`rl_env/` + `trainer/`).
- **Inference-time voting** — sample the model several times through the evaluation harness
  and take the plurality answer. This is an aggregation *over* the evaluators here, not
  separate code.
- **Distillation** — supervised fine-tuning of the 9B on correct traces from a stronger
  teacher (traces are collected by running `benchmark/run_agent_v6.py` with a strong model).
  The fine-tuning itself uses standard external tooling and is not included here.

The guiding theory: RL-with-verifiable-rewards is bounded by the base model's own sample
support — if the correct behavior never appears in its samples, training cannot manufacture
it, whereas distillation can inject it, and voting only aggregates capability the model
already has.

## Credentials

All model backends read credentials from **environment variables** — nothing is committed:

- `AWS_BEARER_TOKEN_BEDROCK`, `AWS_DEFAULT_REGION` — AWS Bedrock backend.
- `OPENAI_API_KEY` — OpenAI / OpenAI-compatible backends.
- For an OpenAI-compatible gateway, override the base URL with `NAUTILUS_BASE_URL` and pass
  its key via `NAUTILUS_API_KEY`.

The RL training run (`trainer/`) does not need any API key; model weights are public
Hugging Face checkpoints.

## Requirements

- `benchmark/` and `rl_env/` need only **numpy, pandas, scipy** (`requirements.txt`); add
  `boto3` (Bedrock) or `openai` (OpenAI-compatible) only if you run the API evaluators.
- `trainer/` runs **inside the SkyRL container** with its own pinned environment — see
  `SETUP.md`. It is not driven by `requirements.txt`.
