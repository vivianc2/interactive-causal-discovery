# Reproducing the training/eval box (SkyRL + GPU)

Goal: reproduce GRPO training and evaluation on a fresh multi-GPU server. Almost everything
is public or regenerable — the only thing you supply out-of-band is **credentials** (only
needed for the API-based *evaluation*, not for the GPU training run). The launcher and the
de-leaked dataset are already in this repository.

Assumes ~8× GPUs (L40S / A100 / H100), NVIDIA driver + Docker + nvidia-container-runtime,
and a large data volume (the working dir grows to ~60 GB: SkyRL `.venv` ~17 GB, HF cache
~34 GB, plus data + checkpoints).

## 0. Why Docker (not optional)
Bare-metal `uv` fails on Amazon Linux 2023 (glibc 2.34): SkyRL pins a `vllm-router` wheel
that needs glibc ≥ 2.35. The published image provides it, so everything runs **inside the
container**.

## 1. Lay out a working dir (this becomes `/work` in the container)
Pick a path on the big volume, e.g. `~/rpgrl`. Under it:
```
rpgrl/
  interactive-causal-discovery/   # this repository (cloned)
  SkyRL/                          # step 2
  hf_cache/                       # step 5 (weights)
  data/                           # optional: fresh dataset builds (step 6)
  logs/  rl_ckpt/                 # created at runtime
  <credential files>             # step 4 (never commit)
```

## 2. Clone SkyRL (public) at the pinned commit
```bash
cd ~/rpgrl
git clone https://github.com/NovaSky-AI/SkyRL.git
cd SkyRL && git checkout bce9ee9a        # the commit used here, for dependency/behavior parity
```

## 3. Symlink the training package into SkyRL's examples
The SkyRL launcher expects this repo's `trainer/` package under `examples/train/rpg`:
```bash
ln -s ~/rpgrl/interactive-causal-discovery/trainer ~/rpgrl/SkyRL/examples/train/rpg
```

## 4. Credentials (only for the API evaluation; never commit)
The GPU RL training run needs **no** credentials. The API evaluators read from environment
variables:
- AWS Bedrock: `AWS_BEARER_TOKEN_BEDROCK`, `AWS_DEFAULT_REGION`.
- OpenAI-compatible endpoint: `OPENAI_API_KEY`, or `NAUTILUS_BASE_URL` + `NAUTILUS_API_KEY`.
- W&B logging is optional; the launcher defaults to console logging. Use your own W&B key if
  you enable it (`LOGGER=wandb`), and set the project in the launcher.

Hugging Face weights are public (no token needed).

## 5. Pull the image + start the container
```bash
docker pull novaskyai/skyrl-train-ray-2.56.0-py3.12-cu12.8
docker run -d --name skyrl --runtime=nvidia --gpus all --shm-size=16g \
  -v ~/rpgrl:/work  novaskyai/skyrl-train-ray-2.56.0-py3.12-cu12.8  sleep infinity
```
Everything below runs inside: `docker exec -it skyrl bash`.

## 6. Install SkyRL into an isolated uv venv (inside the container)
```bash
cd /work/SkyRL
uv sync --extra fsdp
uv pip install scipy pandas          # the oracle needs scipy/pandas
uv pip install boto3                 # only if you'll run the Bedrock API eval
```
Sanity: `/work/SkyRL/.venv/bin/python -c "import vllm, torch; print(torch.cuda.device_count())"`
should print your GPU count.

## 7. Weights (public — download, don't copy 34 GB)
```bash
docker exec skyrl bash -lc 'HF_HOME=/work/hf_cache huggingface-cli download Qwen/Qwen3.5-9B'
```

## 8. Dataset (already in this repo; or regenerate — deterministic)
The canonical de-leaked set is committed at
`interactive-causal-discovery/benchmark/data/{train,validation}.parquet`
(1,536 train / 128 held-out). To (re)build a fresh set (≈20–25 min, CPU):
```bash
docker exec skyrl bash -lc '
  cd /work/SkyRL
  export RPG_SRC=/work/interactive-causal-discovery RPG_PROTO=benchmark PYTHONUNBUFFERED=1
  uv run --isolated python -m examples.train.rpg.rpg_dataset \
    --output_dir /work/data/rpg --train_size 1536 --val_size 128 \
    --train_seed0 10000000 --val_seed0 20000000'
```

## 9. Launch a run
The launcher is `trainer/run_rpg.sh` (already symlinked into `SkyRL/examples/train/rpg/` by
step 3). It runs SkyRL GRPO + LoRA. Before launching, stop any stale Ray and confirm all GPUs
are idle: `docker exec skyrl bash -lc 'ray stop --force'`.
```bash
docker exec skyrl bash -lc '
  cd /work/SkyRL
  export RPG_SRC=/work/interactive-causal-discovery RPG_PROTO=benchmark
  export DATA_DIR=/work/interactive-causal-discovery/benchmark/data   # the committed set
  export MODEL=Qwen/Qwen3.5-9B NUM_GPUS=4        # use GPUs 1-4 if GPU0 hosts an eval server
  bash examples/train/rpg/run_rpg.sh'
```
`run_rpg.sh` sets the LoRA + vLLM flags; append any SkyRL overrides as extra args, e.g.
`... run_rpg.sh generator.inference_engine.gpu_memory_utilization=0.8`. For a larger model,
raise `NUM_GPUS` and lower `gpu_memory_utilization`.

**Verify after launch:** the step-0 eval should show `stop_reason=stop` (no truncation) and
no OOM. Truncated rollouts (`stop_reason=length`) silently zero the reward — always confirm
the model reaches an `answer` turn.
