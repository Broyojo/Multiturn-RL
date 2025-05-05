set -x

export VLLM_USE_V1=1
export WANDB_API_KEY="121d8f0d656e06f7ebd1c51e71e4dbcdb654af8e"

# TODO: set this up for hyparameter optimization

# Hyperparamaters
MODEL="Qwen/Qwen3-8B"
PROJECT="multiturn-rl"
EXPERIMENT="$MODEL-swe-terminal"

export DOCKER_CLIENT_TIMEOUT=120
export COMPOSE_HTTP_TIMEOUT=120

# to enable wandb resuming:
# RUN_ID=$(echo -n "$EXPERIMENT" | md5sum | cut -c1-8)
# export WANDB_RESUME=allow
# export WANDB_RUN_ID="$RUN_ID"

python3 -m main \
    algorithm.adv_estimator=grpo \
    data.train_files=data/swebench/train.parquet \
    data.val_files=data/swebench/test.parquet \
    data.return_raw_chat=True \
    data.train_batch_size=63 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=262144 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.chat_scheduler=chat_scheduler.TerminalChatCompletionScheduler \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=1310720 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.rollout_data_dir=./rollouts/train \
    trainer.validation_data_dir=./rollouts/validation \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT \
    trainer.experiment_name=$EXPERIMENT \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 $@