PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files="data/biased_prompts.parquet" \
  data.val_files="data/biased_prompts.parquet" \
  data.train_batch_size=8 \
  data.max_prompt_length=256 \
  data.max_response_length=128 \
  actor_rollout_ref.model.path="/path/to/qwen-1.4b" \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  algorithm.adv_estimator=grpo \
  custom_reward_function.path="custom_reward" \
  custom_reward_function.name="compute_score" \
  trainer.total_epochs=10 \
  trainer.logger="['console','wandb']" \
  2>&1 | tee outputs/verl_grpo.log
