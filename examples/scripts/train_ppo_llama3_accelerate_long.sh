set -x 

export CUDA_VISIBLE_DEVICES='2,3,4,5,6,7'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

read -r -d '' training_commands <<EOF
../train_ppo.py \
    --pretrain RLHFlow/LLaMA3-SFT \
    --reward_pretrain sfairXC/FsfairX-LLaMA3-RM-v0.1 \
    --save_path ./ckpt/llama3_8b_bt \
    --logging_steps 1 \
    --save_steps 2 \
    --max_ckpt_num 1 \
    --eval_steps -1 \
    --train_batch_size 96 \
    --micro_train_batch_size 1 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 768 \
    --max_epochs 1 \
    --prompt_max_len 6144 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data RLHFlow/iterative-prompt-v1-iter3-20K \
    --prompt_data_probs 1.0 \
    --max_samples 80000 \
    --normalize_reward \
    --actor_init_on_gpu \
    --adam_offload \
    --flash_attn \
    --input_key context_messages \
    --apply_chat_template \
    --save_value_network \
    --gradient_checkpointing
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)


accelerate launch --config_file ./zero2_test.yaml --main_process_port 29505 $training_commands

