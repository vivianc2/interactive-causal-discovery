# benchmark/ — world generator, oracle, audits, and free-text runner

Procedurally generates audited causal-discovery worlds and provides the deterministic
oracle/grader and a free-text scientist runner. A world is a **structural causal model**
(hidden cause + confounders + decoys + a proxy metric + intervenable knobs); a world is a
pure, deterministic function of `(seed, skin, archetype)`.

## Files

| File | Role |
|---|---|
| `engine.py` | The SCM simulation engine (`WorldSCM`, mechanism library, actuators). Ground truth every other module reads. |
| `sampler.py` | Samples a world: `sample_world(seed, skin, archetype)`; defines `FEATURES`, `ARCHETYPES`. |
| `skins.py` | Domain "skins" — non-leaking scenario templates and names that dress the abstract graph. |
| `oracle_v6.py` | The oracle + audit suite: gold answer, counterfactual battery, the **five audits**, and `grade()`. |
| `generate_v7.py` | **Generation entry point** (CLI): samples N worlds, runs the audit gate, writes `world_*.json` + `manifest.json`. |
| `sim_v6.py` | Interactive assay/intervention sim used to run and score a world. |
| `resolver.py` | Maps free-text agent answers/requests to SCM variables and actuators. |
| `run_agent_v6.py` | **Free-text scientist runner** (CLI): an LLM plays the world in natural language (measure → intervene → answer). |
| `run_batch_v6.py` | Runs `run_agent_v6` over a directory of worlds and prints an aggregate report. |
| `sandbox.py` | Sandboxed subprocess for the agent's optional Python "code" actions. |
| `bedrock_llm.py`, `openai_llm.py` | Model backends (AWS Bedrock; OpenAI-compatible). |
| `dump_worlds.py` | Reconstructs the exact `world_*.json` behind a dataset parquet split. |
| `test_reward_integrity.py` | Reward-integrity smoke test over real audited worlds. |
| `data/` | The committed de-leaked dataset: `train.parquet` (1,536) and `validation.parquet` (128 held-out). |

The `v6`/`v7` suffixes name the **design generation** of a component (carried forward
unchanged), not a dataset version. These are the live files.

## Usage

```bash
# Generate audited worlds
python generate_v7.py --outdir /tmp/worlds --n 40 --seed 100000
# Optionally require a feature or fix an archetype:
python generate_v7.py --outdir /tmp/worlds_sf --n 20 --require-feature sign_flip

# Reconstruct the exact worlds behind a committed split (deterministic)
python dump_worlds.py data/validation.parquet /tmp/heldout_worlds

# Play a world in free text (use --backend mock for no API access)
python run_agent_v6.py --world-file /tmp/heldout_worlds/world_<id>.json \
    --backend mock --out /tmp/run.json -v

# Reward-integrity check (offline)
python test_reward_integrity.py
```

Scoring is **deterministic and rule-based** — no model sits in the evaluation path. The
primary metric is *utility recovered | completed* (see the top-level `README.md`).
