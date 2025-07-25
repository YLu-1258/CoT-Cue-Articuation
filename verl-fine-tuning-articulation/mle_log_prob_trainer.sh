set -x
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export HYDRA_FULL_ERROR=1
export FLASH_ATTENTION_SKIP_CUDA_CHECK=TRUE
export DISABLE_FLASH_ATTN=1
export PYTHONUNBUFFERED=1
export VLLM_TARGET_DEVICE=cuda
export CUDA_VISIBLE_DEVICES=1,2

ray stop && ray start --head --num-gpus=2

python3 -m verl.trainer.main_mle \
    data.train_files=/data/alexl/gsm8k/train.parquet \
    data.val_files=/data/alexl/gsm8k/test.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    train.learning_rate=5e-5 \
    train.train_batch_size=8 \
    train.max_steps=5000 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.model.trust_remote_code=true \
    +actor_rollout_ref.ref.trust_remote_code=true \
    +actor_rollout_ref.rollout.trust_remote_code=true \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_mle_trainer' \
    trainer.experiment_name='qwen3_1_7b_log_prob_award' \
    trainer.n_gpus_per_node=2 \
    +ray_init.num_gpus=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 2>&1 | tee token_length_fine_tuning.log $@