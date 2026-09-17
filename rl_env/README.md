# rl_env/ — id-space RL environment, reward, and RL-comparable evaluation

The reinforcement-learning view of the benchmark. The policy interacts with a world in
**id space**: every variable and actuator is an opaque identifier (`m0`, `a3`, …) drawn from
a seeded catalog, with no free-text resolver in the loop. The reward is **deterministic and
terminal** (0 on measure/intervene/code turns; the graded scalar at answer / give-up /
turn-cap).

## Files

| File | Role |
|---|---|
| `catalog.py` | Opaque-id catalog: canonical variable/actuator ↔ `m*`/`a*`, seeded shuffle, neutral descriptions. |
| `env.py` | The RL environment (`RPGEnv`): gym-style `reset()` / `step(text)`, hardened action parsing, `SYSTEM_PROMPT`. |
| `reward.py` | The deterministic reward (`compute_reward`, `RewardConfig`): part-A (found the fix) + part-B (mechanism), invalid-id penalty, and the faithfulness gate (no credit without ≥1 applied intervention). |
| `splits.py` | Train / held-out split by reserved skins and archetypes (the transfer set). |
| `world_stream.py` | Infinite seeded stream of audited worlds with split routing and a leakage guard. |
| `rollout.py` | Multi-turn episode + GRPO-group rollout helpers. |
| `id_space_eval.py` | Evaluate an API model in id space over saved worlds — the RL-comparable ceiling. |
| `test_env_reward.py` | Gold id-answer → reward ≈ 1; evidence gate and part-B behavior. |
| `test_parse_regression.py` | Locks the action-parser and sign-canonicalization rules. |

## Import path

These modules import the world generator by module name, so put `benchmark/` on the path:

```bash
# from inside rl_env/
PYTHONPATH=../benchmark:. python test_env_reward.py
PYTHONPATH=../benchmark:. python test_parse_regression.py

# id-space evaluation of an API model over reconstructed worlds
PYTHONPATH=../benchmark:. python id_space_eval.py \
    --worlds-dir /tmp/heldout_worlds --backend bedrock --outdir /tmp/idspace_eval
```

(The SkyRL trainer sets this path automatically via `RPG_SRC` / `RPG_PROTO` — see
`../trainer/` and `../SETUP.md`.)
