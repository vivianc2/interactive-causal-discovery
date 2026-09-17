#!/usr/bin/env bash
# GRPO + LoRA training on the RPG causal-discovery env (Qwen3.5-9B; larger model = change MODEL).
# Run INSIDE the SkyRL container, from the SkyRL repo root, after (see SETUP.md):
#   1) this package symlinked in:  ln -s /work/interactive-causal-discovery/trainer examples/train/rpg
#   2) science dep present:        uv pip install scipy        # oracle uses scipy.stats
#   3) dataset: the committed set at benchmark/data is used by default (DATA_DIR below);
#      to (re)build a fresh set:   uv run --isolated python -m examples.train.rpg.rpg_dataset \
#                                     --output_dir /work/data/rpg --train_size 1536 --val_size 128
#   4) GPUs: if GPU0 hosts an eval server, e.g. export CUDA_VISIBLE_DEVICES=1,2,3,4
# Then: bash examples/train/rpg/run_rpg.sh
set -x

export RPG_SRC="${RPG_SRC:-/work/interactive-causal-discovery}"  # repo root (holds benchmark/ and rl_env/)
export RPG_PROTO="${RPG_PROTO:-benchmark}"   # world sampler/oracle package; must match the one the dataset was built with
DATA_DIR="${DATA_DIR:-$RPG_SRC/benchmark/data}"   # committed de-leaked set (1536 train / 128 held-out)
NUM_GPUS="${NUM_GPUS:-4}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"    # larger model (e.g. 27B) may need more GPUs / lower mem-util
LOGGER="${LOGGER:-console}"

uv run --no-sync --extra fsdp -m examples.train.rpg.main_rpg \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL" \
  trainer.placement.colocate_all=true \
  trainer.policy.model.lora.rank=16 \
  trainer.policy.model.lora.alpha=32 \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  generator.batched=false \
  generator.n_samples_per_prompt=8 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.algorithm.use_kl_loss=false \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=8 \
  trainer.max_prompt_length=4096 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.eval_before_train=true \
  trainer.eval_interval=10 \
  trainer.ckpt_interval=20 \
  environment.env_class=rpg \
  trainer.logger="$LOGGER" \
  trainer.project_name="rpg-causal-discovery" \
  trainer.run_name="rpg_qwen3.5_9b_grpo_lora" \
  trainer.ckpt_path="/work/rl_ckpt/rpg_skyrl" \
  "$@"

# --- optional refinements to try once the base run works (confirm exact flag names in
#     this SkyRL build):
#   Dr.GRPO (mean-subtract, no std-normalization) — disable GRPO std-norm
#   DAPO dynamic sampling — drop zero-variance groups (dynamic_sampling=filter)
#   These are config flags; leaving defaults (std-norm on, no dynamic sampling) is a valid
#   plain-GRPO first run.
