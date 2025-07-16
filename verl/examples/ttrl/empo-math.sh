#!/bin/bash
#export VLLM_ATTENTION_BACKEND=XFORMERS
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1

# ------------------------------------------------------------

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

TASK="dapo-math-17k-dedup"
BACKBONE="Qwen3-4B"
ADVANTAGE="grpo"

MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=$((1024 * 4))
N=1 # 8

EPISODE=1
DATA_TRAIN_BATCH_SIZE=8 # Rollout Prompt Num
N_VOTES_PER_PROMPT=32
N_SAMPLES_PER_PROMPT=32
MINI_BATCH_SIZE=2 # Actual mini batch size is MINI_BATCH_SIZE * N_SAMPLES_PER_PROMPT
MICRO_BATCH_SIZE=2

temperature=0.6
top_p=0.95
top_k=20 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

DATA_LOCAL_DIR="/home/zelin/verl/data"
BACKBONE_PATH="Qwen/${BACKBONE}"

ENTROPY_THRES=1.0
TARGET='semantic_entropy' # "gt" corresponds to GRPO, "semantic_entropy" to EMPO, and "voting" to TTRL

MODEL="${TASK}-${BACKBONE}"
EXPERIMENT="${TARGET}-Len@${K}k-bz-${DATA_TRAIN_BATCH_SIZE}-thres-${ENTROPY_THRES}"

WANDB_PROJECT="TTRL-verl"
LOG_NAME="${DATE}-${EXPERIMENT}-${MODEL}-${ADVANTAGE}"
OUTPUT_DIR="/home/zelin/verl/ckpts/${WANDB_PROJECT}/${MODEL}/${DATE}/${EXPERIMENT}-${ADVANTAGE}-${TIME_TAG}"

# ------------------------------------------------------------
python -m verl.trainer.main_ppo \
  reward_model.reward_manager=ttrl \
  reward_model.reward_kwargs.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  reward_model.reward_kwargs.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
  reward_model.reward_kwargs.mode="train" \
  reward_model.reward_kwargs.reward_type=$TARGET \
  reward_model.reward_kwargs.entropy_thres=$ENTROPY_THRES \
  data.train_files=["$DATA_LOCAL_DIR/$TASK.parquet"] \
  data.val_files=["$DATA_LOCAL_DIR/aime-2024.parquet"] \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=${temperature} \
  actor_rollout_ref.rollout.top_p=${top_p} \
  actor_rollout_ref.rollout.top_k="${top_k}" \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.do_vote=True \
  actor_rollout_ref.rollout.n_vote=$N_VOTES_PER_PROMPT \
  actor_rollout_ref.rollout.n=$N_SAMPLES_PER_PROMPT \
  actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
  actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
  actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=$N \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  critic.optim.lr=9e-6 \
  critic.model.use_remove_padding=True \
  critic.model.path=$BACKBONE_PATH \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  algorithm.kl_ctrl.kl_coef=0.001 \
  algorithm.adv_estimator=$ADVANTAGE \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$WANDB_PROJECT \
  trainer.experiment_name=$LOG_NAME \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=2 \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.max_critic_ckpt_to_keep=2 \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_epochs=$EPISODE "$@"

echo "Output directory: $OUTPUT_DIR"
