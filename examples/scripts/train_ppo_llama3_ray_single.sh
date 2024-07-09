set -x 
export CUDA_VISIBLE_DEVICES='1,2,3,4'

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"working_dir": "/openrlhf", "pip": "/openrlhf/requirements.txt"}' \
    -- python3 examples/train_ppo_ray.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 1 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --colocate_critic_reward \
    --colocate_actor_ref \
    --ref_reward_offload \
    --pretrain /openrlhf/examples/test_scripts/ckpt/llama3_8b_bt \
    --reward_pretrain sfairXC/FsfairX-LLaMA3-RM-v0.1 \
    --save_path /openrlhf/examples/test_scripts/ckpt/llama_8b_ray \
    --micro_train_batch_size 4 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 8 \
    --rollout_batch_size 1024 \
    --max_samples 100000 \
    --max_epochs 1 \
    --prompt_max_len 8192 \
    --generate_max_len 2048 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data RLHFlow/iterative-prompt-v1-iter2-20K \
    --input_key context_messages \
    --apply_chat_template \
    --normalize_reward \
    --adam_offload \
    --gradient_checkpointing