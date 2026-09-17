# AGENTS.md — Project handoff

## 1. Project overview

**What it is.** An *interactive* causal-discovery benchmark for language models, plus the
reinforcement-learning stack used to study how to install that skill in a small (9B) model.
Each task is a **structural causal model** ("world") with a hidden cause, decoys (correlate
with the outcome but have zero interventional effect), a measurable proxy, and intervenable
knobs. Every entity is an **opaque identifier** (`m0`, `a3`, …), so the model must *discover*
structure by acting — measure → intervene (`do(·)`) → submit an answer — not by reading names.

**Problem it solves.** Most "causal" LLM benchmarks are *observational* (read a described
graph, answer). That skips the hard part: deciding what to measure and intervening to
separate correlation from causation. This benchmark measures, and the study tries to improve,
whether a small deployable model can identify a **confounded cause by acting**.

**Current status.** *Research prototype, benchmark complete and stable; RL result is a
settled negative; the distillation lever is the productizable path.* The benchmark generator,
audits, reward, RL environment, and the SkyRL GRPO trainer are working and reproducible from
this repo. The distillation and voting recipes were evaluated in the study but their training
code / checkpoints are **not** in this repository (see §6, §9).

---

## 2. How to set up & run

### Offline (no GPU, no API key) — benchmark + reward + smoke tests
```bash
pip install -r requirements.txt          # numpy, pandas, pyarrow, scipy
cd benchmark
python generate_v7.py --outdir /tmp/worlds --n 8 --seed 100000     # generate audited worlds
python dump_worlds.py data/validation.parquet /tmp/heldout          # rebuild the exact held-out worlds
python test_reward_integrity.py                                     # reward-integrity checks
cd ../rl_env
PYTHONPATH=../benchmark:. python test_env_reward.py                 # env reward path
PYTHONPATH=../benchmark:. python test_parse_regression.py           # parser / sign canon
```
All three tests pass on a clean checkout (verified).

### Free-text / API evaluation
`benchmark/run_agent_v6.py --world-file <world.json> --backend {bedrock|openai|vllm|mock} --out run.json`.
Use `--backend mock` to exercise the loop with no API. `rl_env/id_space_eval.py` runs the
RL-comparable id-space evaluation over saved worlds.

### GPU training (SkyRL GRPO + LoRA)
Runs **inside the SkyRL container** on a multi-GPU box. Follow **`SETUP.md`** end to end.
Summary: clone SkyRL at the pinned commit `bce9ee9a`, symlink `trainer/` into
`SkyRL/examples/train/rpg`, `uv sync --extra fsdp`, then `bash examples/train/rpg/run_rpg.sh`
with `RPG_SRC`/`RPG_PROTO`/`DATA_DIR`/`MODEL` set. The committed dataset in `benchmark/data`
is used by default.

### Credentials & what commonly breaks
- Credentials are read from **environment variables only** — nothing is committed. Bedrock:
  `AWS_BEARER_TOKEN_BEDROCK`, `AWS_DEFAULT_REGION`. OpenAI-compatible: `OPENAI_API_KEY`, or
  `NAUTILUS_BASE_URL` + `NAUTILUS_API_KEY`. The GPU training run needs **no** key.
- **Truncation silently zeros the reward.** If rollouts hit the token cap before the model can
  submit an `answer`, the reward is 0 and can be misread as "the model can't learn." Always run
  at the model's true max context and a generous per-turn cap, and confirm `stop_reason=stop`
  (not `length`) at step-0. This has produced false negatives before. (See §9.)
- Bare-metal `uv` fails on Amazon Linux 2023 (glibc 2.34); SkyRL needs glibc ≥ 2.35 — hence the
  Docker image. See `SETUP.md §0`.
- `rl_env/` scripts import the generator by module name, so put `benchmark/` on `PYTHONPATH`
  (the trainer does this automatically via `RPG_SRC`/`RPG_PROTO`).

---

## 3. Codebase map

```
benchmark/     World generator, oracle, audits, SCM engine, free-text runner, dataset.
  engine.py            SCM simulation engine (WorldSCM, mechanism library, actuators) — ground truth.
  sampler.py           sample_world(seed, skin, archetype); defines FEATURES, ARCHETYPES.
  skins.py             Domain skins: non-leaking scenario templates / names.
  oracle_v6.py         Oracle + audit suite: gold answer, counterfactual battery, the 5 audits, grade().
  generate_v7.py       ENTRY POINT (CLI): sample N worlds, run the audit gate, write world_*.json + manifest.
  sim_v6.py            Interactive assay/intervention sim used to run & score a world.
  resolver.py          Maps free-text requests/answers to SCM variables & actuators.
  run_agent_v6.py      ENTRY POINT (CLI): free-text LLM scientist loop (measure/intervene/answer).
  run_batch_v6.py      Runs run_agent_v6 over a directory of worlds; aggregate report.
  sandbox.py           Sandboxed subprocess for the agent's optional Python "code" actions.
  bedrock_llm.py       AWS Bedrock backend.  openai_llm.py  OpenAI-compatible backend.
  dump_worlds.py       Rebuild the exact world_*.json behind a dataset parquet split (deterministic).
  data/                Committed de-leaked dataset: train.parquet (1,536), validation.parquet (128 held-out).

rl_env/        Id-space RL environment, reward, RL-comparable evaluation.
  catalog.py           Opaque-id catalog (canonical ↔ m*/a*, seeded shuffle, neutral descriptions).
  env.py               RPGEnv: gym-style reset()/step(text), hardened parsing, SYSTEM_PROMPT.
  reward.py            Deterministic reward (compute_reward, RewardConfig): part-A + part-B, invalid penalty, evidence gate.
  splits.py            Train / held-out split by reserved skins + archetypes (the transfer set).
  world_stream.py      Infinite seeded stream of audited worlds with split routing + leakage guard.
  rollout.py           Multi-turn episode + GRPO-group rollout helpers.
  id_space_eval.py     ENTRY POINT (CLI): evaluate an API model in id space (RL-comparable ceiling).

trainer/       SkyRL GRPO + LoRA integration (out-of-tree; symlinked into SkyRL/examples/train/rpg).
  run_rpg.sh           ENTRY POINT: launcher; sets env + all GRPO/LoRA/vLLM flags.
  main_rpg.py          SkyRL entry point: builds config, registers the `rpg` env, runs the experiment.
  env.py               RPGSkyEnv: thin BaseTextEnv wrapper over rl_env/RPGEnv.
  rpg_dataset.py       Builds the SkyRL train/validation parquet (one audited world per row).
```

**Naming note.** `v6`/`v7` suffixes name the design generation of a *component*, carried
forward unchanged. They are the current, live files — not a dataset version.

---

## 4. Data & resources

- **Dataset (committed):** `benchmark/data/{train,validation}.parquet` — **1,536 train / 128
  held-out** worlds, SkyRL/verl row format (`data_source, prompt, env_class, reward_spec,
  extra_info`). Worlds are deterministic in `(seed, skin, archetype)`; `extra_info` carries
  the identity so any row can be rebuilt exactly. Train seeds from `10_000_000`, validation
  from `20_000_000`.
- **Splits:** train uses 6 archetypes (`collider_selection`, `confounded_chain`,
  `confounded_reversal`, `dose_window`, `instrument_only`, `synergy_pair`); the held-out set
  reserves 3 additional transfer archetypes (`competing_causes`, `hidden_subtype`,
  `surrogate_trap`) plus reserved skins.
- **Models:** policy = `Qwen/Qwen3.5-9B` (public HF weights; no token needed). API backends
  (Bedrock / OpenAI-compatible) are used only for the API evaluators and for collecting
  teacher traces.
- **Outputs:** generated worlds → the `--outdir` you pass; free-text runs → the `--out` JSON;
  RL checkpoints/logs → the paths set in `run_rpg.sh` (`ckpt_path`, logger). W&B logging is
  opt-in (`LOGGER=wandb`), console by default.
- **Upstream:** SkyRL — https://github.com/NovaSky-AI/SkyRL, pinned commit `bce9ee9a`.

---

## 5. Key decisions & rationale

- **Interactive, not observational.** Observation cannot separate correlation from causation;
  only intervention can. Worlds therefore require acting, and observation-only answers earn no
  credit (the reward's *evidence gate*, `reward.py: require_evidence=True`).
- **Opaque identifiers for every entity.** Prevents the model from pattern-matching a helpful
  name to the answer; it must run experiments. A hardening pass removed two shortcuts (leaky
  names, all-binary interventions).
- **Generate → audit → reject loop with five independent audits** (`oracle_v6.py`: decoy,
  proxy-signal, distractor-inertness, gold self-consistency, counterintuitiveness). Any world a
  shortcut could solve is discarded, so each accepted world is "hard for the right reason."
- **Deterministic, rule-based reward — no model in the evaluation path.** Reproducible,
  bias-free scoring. Headline metric is **utility recovered | completed** (part-A conditioned on
  submitting an answer), decomposed as `completion-rate × quality` so "learned to submit an
  answer" cannot masquerade as skill. The combined training reward is
  `0.5·part_a + 0.5·part_b − 0.25·invalid` (weights env-gated via `RPG_W_A`/`RPG_W_B`).
- **Paired, budget-matched, confidence-interval protocol.** Every lever is compared against the
  same base model on the same held-out worlds at the deployment sampling budget, with bootstrap
  95% intervals; a result counts only when its interval excludes zero.
- **Held-out transfer split** (reserved skins + archetypes) rather than a random split — tests
  generalization to unseen structure, not memorization.
- **GRPO + LoRA via SkyRL** for the RL lever: multi-turn, terminal-reward-compatible, and LoRA
  keeps the run affordable. Config in `run_rpg.sh` (advantage_estimator=grpo, LoRA rank 16 /
  alpha 32, `n_samples_per_prompt=8`, lr 1e-6, `use_kl_loss=false`, `train_batch_size=64`).

---

## 6. What was tried — outcomes (including what did not work)

*Only settled, defensible findings are recorded here.*

- **Reinforcement learning from base — did not add the skill (settled negative).** GRPO from
  the base 9B lands within noise of the base; a longer run drifted down, not up. Rejection
  fine-tuning and preference-style variants were likewise null. **Why:** a hint probe showed
  that when *told* the true cause the model executes the fix well, but training on those same
  demonstrations with the hint removed transfers ≈ +0.00 — the wall is on *finding* the cause,
  not executing it ("being shown ≠ knowing"). This is model **capacity** at 9B, not reward
  design. Consistent with RL-with-verifiable-rewards being bounded by the base model's own
  sample support: if the behavior never appears in the base's samples, RL cannot manufacture it.
- **Measurement rigor was itself a finding.** An early, promising RL "**+0.10**" shrank to
  **+0.03** then **−0.01** once an output-truncation artifact, single-sample noise, and a shared
  parser bug were each removed. Lesson: decompose completion-rate × quality, use paired
  intervals, and read raw rollouts — aggregate dashboards hid pipeline bugs.
- **A larger, off-the-shelf reasoning model underperformed the base** on this budgeted,
  interactive task — it spent its budget on reasoning instead of acting. Scale alone did not
  supply the skill.
- **Distillation — worked (the productizable lever).** Fine-tuning the 9B on correct traces
  from a stronger teacher installs the skill (see §7). *Training code/checkpoints for this are
  not in this repo.*
- **Inference-time voting — worked, teacher-free**, as an aggregation over the evaluator. *No
  separate code in this repo; it is sampling the evaluator K times and taking the plurality.*
- **Structured residual difficulty.** Two archetypes — latent-subtype identification
  (`hidden_subtype`) and synergistic multi-cause (`synergy_pair`) — stay near the floor for
  **every** model tested, including the teacher: a genuine teacher-and-capacity ceiling, not a
  method bug.
- **External validity is limited.** On an independent published benchmark the distilled skill
  did **not** transfer, so the gain is benchmark-specific (motivates the grounding work in §8).

---

## 7. Current state & results

Working now: benchmark generation + audits, deterministic reward, id-space RL environment,
free-text and id-space evaluators, and the SkyRL GRPO trainer. All offline tests pass.

**Verified numbers** (metric = *utility recovered | completed* on the 128 held-out worlds;
direct measurement):

| Model | utility \| completed | n | Note |
|---|---|---|---|
| Base 9B | **0.29** | 8 | 0.27–0.33 across configs |
| Distilled 9B | **0.50** | 8 | statistically level with the 27B reference |
| 27B reference | ~0.47 | 1 | single-sample → noisier |
| RL (GRPO) from base | ≈ base (Δ ≈ −0.01) | 8 | interval spans zero (null) |

Best-of-4 on the base (~0.50) far exceeds single-try (~0.29): the correct answer is
*reachable but not committed* by the base — which is why distillation and voting help while
RL does not. Voting and combined distillation+voting produced further gains in the study; those
were measured on an earlier evaluation pipeline and **should be re-measured on the current
harness before being quoted** (see §9).

**How measured:** paired against the same base on the same 128 held-out worlds, temperature 1.0,
bootstrap 95% CIs; utility scored only on episodes that submitted an answer.

---

## 8. Open next steps (prioritized)

1. **[Promising] Ground the worlds in a real domain.** The central limitation is that worlds
   are domain-agnostic (hence the failed external transfer). Sample task-worlds from a domain
   **knowledge graph** and re-run the same audit gate. Treat KG relationships as *uncertain
   priors the model must test against real data*, not as ground truth — the skill to install is
   "verify, don't trust." Path to distilled per-domain expert models.
2. **[Promising] Distillation as the deployment recipe.** It is the one lever that adds the
   skill on a 9B and matches a much larger model. Next: land the SFT training code/checkpoints
   into a repo (currently external), and quantify the voting add-on's cost/benefit
   (voting adds ~4–8× inference for a shrinking gain on an already-strong distilled model).
3. **[Speculative] Retry RL with KL-regularization.** A prior unregularized long run drifted /
   collapsed. `run_rpg.sh` currently sets `use_kl_loss=false`; a KL-regularized configuration is
   designed but **untested** — worth a controlled run to see if it stays stable (do not expect it
   to beat distillation, given the capacity finding).
4. **[Promising] Close the two residual archetypes** (`hidden_subtype`, `synergy_pair`),
   unsolved even by the teacher — likely needs a stronger teacher and/or targeted curriculum.
5. **[Infra] Harden long training runs** — persistent checkpoint storage + auto-resume, so
   multi-day jobs survive host reboots.
6. **[Available hook, unvalidated] Belief-graph reward shaping** exists in `rl_env/env.py`
   (opt-in via `RPG_BELIEF_SHAPING`, default OFF). It is wired but not validated as a win; treat
   as an experiment, not a recommendation.

---

## 9. Known issues, gotchas & tech debt

- **Truncation → silent reward 0.** The single most dangerous failure mode. Always run at the
  model's true max context with a generous per-turn cap and verify `stop_reason=stop`. For
  SkyRL, raise `generator.max_input_length`, `trainer.max_prompt_length`,
  `generator.sampling_params.max_generate_length`, and the engine's `max_num_batched_tokens`
  together — the defaults are far too small for the verbose, catalog-repeating observations.
- **Metric vs. reward.** Evaluation summaries report a *combined reward*
  (`0.5·part_a + 0.5·part_b − 0.25·invalid`), which is **not** the headline *utility | completed*
  metric. Do not quote the combined reward as the utility score.
- **Voting / distillation numbers need re-measurement.** The distillation-alone and base
  numbers in §7 are directly verified; the voting and distillation+voting gains were on an
  earlier pipeline and are not re-verifiable from this repo as-is. Re-measure before quoting.
- **`RPG_PROTO` must match the dataset build.** It selects the generator package; a mismatch
  against the parquet's provenance produces subtly wrong worlds. Default `benchmark`.
- **Determinism assumption.** Worlds are pure functions of `(seed, skin, archetype)`.
  Reproducing a specific split requires the same seeds; `dump_worlds.py` is the bridge from a
  parquet back to the exact world JSONs.
- **Evidence gate can surprise.** An answer produced with zero applied interventions earns
  zero credit by design (`require_evidence=True`). A gold *answer* is not enough without a gold
  *rollout* that actually intervenes.
- **`trainer/` is not driven by `requirements.txt`** — it runs inside the SkyRL container with
  its own pinned environment (`SETUP.md`). SkyRL is pinned to commit `bce9ee9a` for parity.
- **Third-party evaluators require optional deps** not in `requirements.txt` by default:
  `boto3` (Bedrock) or `openai` (OpenAI-compatible).

---

## 10. Key references & contacts

- **Primary knowledge source:** the accompanying final report and slide deck (project
  deliverables) — the definitive narrative for background, method, and results.
- **This repo:** top-level `README.md` (structure + pipeline), `SETUP.md` (GPU reproduction),
  and the per-directory `README.md` files.
- **Upstream training framework:** SkyRL — https://github.com/NovaSky-AI/SkyRL (commit `bce9ee9a`).
- **Policy model:** `Qwen/Qwen3.5-9B` (public Hugging Face weights).
