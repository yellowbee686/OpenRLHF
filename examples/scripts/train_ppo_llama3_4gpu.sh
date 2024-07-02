set -x 

export CUDA_VISIBLE_DEVICES='2,3,4,5'

read -r -d '' training_commands <<EOF
../train_ppo.py \
    --pretrain RLHFlow/LLaMA3-SFT \
    --reward_pretrain RLHFlow/ArmoRM-Llama3-8B-v0.1 \
    --save_path ./ckpt/llama3_8b_armo \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --micro_train_batch_size 2 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size 1024 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data OpenLLMAI/prompt-collection-v0.1 \
    --prompt_data_probs 0.1 \
    --max_samples 80000 \
    --normalize_reward \
    --actor_init_on_gpu \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)

if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi
