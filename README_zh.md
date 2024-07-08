<div align="center">
<p align="center">
<img alt="" src="./docs/logo.png" style="display: inline-block; height: 160px" />
</p>
</div>

<div align="center">
<p align="center">
      <a href="https://github.com/openllmai/OpenRLHF/graphs/contributors">
        <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/openllmai/OpenRLHF" />
      </a>
      <a href="https://github.com/openllmai/OpenRLHF/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/openllmai/OpenRLHF?color=0088ff" />
      </a>
      <a href="https://github.com/openllmai/OpenRLHF/discussions">
        <img alt="Issues" src="https://img.shields.io/github/discussions/openllmai/OpenRLHF?color=0088ff" />
      </a>
      <a href="https://github.com/openllmai/OpenRLHF/pulls">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/openllmai/OpenRLHF?color=0088ff" />
      <a href="https://github.com/openllmai/OpenRLHF/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/openllmai/OpenRLHF?color=ccf" />
      </a>
      <br>
      <em>开源 / 全面 / 轻量级 / 易用</em>
    </p>
</p>
</div>

<hr>

<span>[ <a href="README.md">English</a> | 中文 ]</span>

OpenRLHF 是一个基于 Ray、DeepSpeed 和 HF Transformers 构建的高性能 RLHF 框架：

- **简单易用**: OpenRLHF 是目前可用的最简单的高性能 RLHF 库之一，兼容 Huggingface 模型和数据集。
- **高性能**: RLHF 训练中 80% 的时间用于样本生成阶段。得益于使用 Ray 和 Adam Offload（固定内存）以及 vLLM 生成加速的能力，OpenRLHF 的性能是极致优化的 DeepSpeedChat with Hybrid Engine 的两倍以上。
- **分布式 RLHF**:  OpenRLHF 使用 Ray 将 Actor、Reward、Reference 和 Critic 模型分布到不同的 GPU 上，同时将 Adam 优化器放在 CPU 上。这使得使用多个 A100 80G GPU 和 vLLM 可以全面微调超过 70B+ 的模型 以及在多个 24GB RTX 4090 GPU 上微调 7B 模型。
- **PPO 实现技巧**: 我们集成了 PPO 的实现技巧以提高训练稳定性，详情参考 [知乎](https://zhuanlan.zhihu.com/p/622134699) 和 [Notion blog](https://difficult-link-dd7.notion.site/eb7b2d1891f44b3a84e7396d19d39e6f?v=01bcb084210149488d730064cbabc99f).

更多细节请参考 [技术报告](https://arxiv.org/abs/2405.11143) | [使用文档](https://openrlhf.readthedocs.io/)


## 特性

- 基于 Ray 的分布式 [PPO based on Ray](./examples/scripts/train_ppo_llama_ray.sh). 
- 支持使用 [超过 70 亿参数的模型进行全面 RLHF 微调](./examples/scripts/train_ppo_llama_ray_70b.sh).
- 支持 vLLM 生成加速在 RLHF 中 (--vllm_num_engines).
- 支持 多个奖励模型 (--reward_pretrain model1,model2...).
- 支持 [DPO (直接偏好优化)/IPO/cDPO](./examples/scripts/train_dpo_llama.sh).
- 支持 [Kahneman-Tversky 优化 (KTO)](./examples/scripts/train_kto_llama.sh).
- 支持 [拒绝采样](./examples/scripts/train_rejection_sampling_llama.sh).
- 支持 [Iterative DPO](./examples/scripts/train_iterative_dpo_llama.sh) (https://github.com/RLHFlow/Online-RLHF).
- 支持 [条件 SFT](./examples/scripts/train_conditional_llama.sh) (https://arxiv.org/abs/2308.12050).
- 支持 [知识蒸馏](./examples/scripts/train_knowledge_distillation.sh) (https://github.com/microsoft/LMOps/tree/main/minillm).
- 支持 [MoE](./examples/test_scripts/train_sft_mixtral_lora.sh) (--aux_loss_coef)
- 支持 Wandb 日志 (--wandb).
- 支持 FlashAttention2 (--flash_attn).
- 支持 QLoRA (--load_in_4bit), [LoRA (--lora_rank, --target_modules)]((./examples/scripts/train_sft_mixtral_lora.sh)).
- 支持 HuggingFace `tokenizer.apply_chat_template` 用于数据集处理 (--apply_chat_template and --input_key).
- 多节点 [训练脚本](./examples/scripts/train_llama_slurm.sh) 适用于 Slurm.


### PPO 支持矩阵

| 特性 | OpenRLHF | DSChat | CAIChat | TRL |
| ------------- |:-------------:| :-------------:| :-------------:| :-------------:| 
| 使用 16 个 A100 完成 70B+ 全微调      | ✅ | ❌ | ❌ | ❌ ||
| 使用 4 个 RTX4090 完成 7B 全微调 | ✅      |    ❌ | ❌ | ❌ | 
| 使用 8 个 A100 完成 34B DPO 全微调 | ✅      |    ❌ | ❌ | ❌ |   
| 支持推理引擎加速 | ✅      |    ✅ | ❌ | ❌ |  
| PPO 实现技巧 | ✅      |    ❌ | ❌ | ✅ | 
| 支持 QLoRA | ✅      |    ❌ | ❌ | ✅ | 
| 支持 Mixtral 8*7b | ✅      |    ❌ | ❌ | ❌ | 
| 支持未合并的 Actor-Critic | ✅     |   ✅ | ✅ | ❌ | 
| 支持多个奖励模型 | ✅      |    ❌ | ❌ | ❌ |   
| 支持 Huggingface 模型 | ✅      |    ✅ | ✅ | ✅ | 
| 易于使用 | ✅      |   ❌ (HybridEngine bugs) | ✅ | ✅ | 

## 快速开始

### 安装

要使用 OpenRLHF，首先 `git clone` 并启动 Docker 容器（**推荐**）：

```bash
git clone https://github.com/openllmai/OpenRLHF.git
# 也可以 `git checkout` 最近的稳定 release 版本.

# 如果需要使用 vLLM，请构建 Docker 镜像以避免依赖问题（可选）
docker build -t nvcr.io/nvidia/pytorch:24.02-py3 ./OpenRLHF/dockerfile

# 启动 Docker 容器
docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN -v $PWD/OpenRLHF:/openrlhf nvcr.io/nvidia/pytorch:24.02-py3 bash
```

> [!NOTE]
>我们提供了 [One-Click Installation Script of Nvidia-Docker](./examples/scripts/nvidia_docker_install.sh)

然后在 Docker 容器内 `pip install` openrlhf

```bash
cd /openrlhf
pip install --user .

cd examples
```

### 准备数据集
OpenRLHF 在其数据集类中提供了多种数据处理方法。
例如在 [Prompt Dataset](https://github.com/OpenLLMAI/OpenRLHF/blob/895e8089dc0b1db230316207ca702d5133ae18fd/openrlhf/datasets/prompts_dataset.py#L6) 中：

```python
def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
    if apply_chat_template:
        prompt = apply_chat_template(data[input_key], tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
    return prompt
```

- 我们可以使用 `--input_key` 指定输入数据集的 `JSON key name` `--prompt_data {name or path}` (PPO) 或 `--dataset {name or path}`，并使用 `--apply_chat_template` 利用 [Huggingface Tokenizer](https://huggingface.co/docs/transformers/main/en/chat_templating) 中的 `chat_template`。
- 如果不想使用 `--apply_chat_template`，可以改用 `--input_template`，或预先离线处理数据集。
- OpenRLHF 还支持使用 `--prompt_data_probs 0.1,0.4,0.5` (PPO) 或 `--dataset_probs 0.1,0.4,0.5` 混合多个数据集。

Chat Templating 的工作原理如下:

```python
dataset = [{"input_key": [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]}]

tokenizer.apply_chat_template(dataset[0]["input_key"], tokenize=False)

"<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]"
```
> [!NOTE]
> `JSON key` 选项取决于具体的数据集。请参阅 [Reward Dataset](https://github.com/OpenLLMAI/OpenRLHF/blob/895e8089dc0b1db230316207ca702d5133ae18fd/openrlhf/datasets/reward_dataset.py#L10) 和 [SFT Dataset](https://github.com/OpenLLMAI/OpenRLHF/blob/895e8089dc0b1db230316207ca702d5133ae18fd/openrlhf/datasets/sft_dataset.py#L9)


### Supervised Fine-tuning

OpenRLHF 的模型检查点完全兼容 HuggingFace 模型。您可以使用 `--pretrain  {name or path}`、`--reward_pretrain  {name or path}` 和 `--critic_pretrain  {name or path}` 指定模型名称或路径。我们在 [HuggingFace OpenLLMAI](https://huggingface.co/OpenLLMAI) 上提供了一些预训练的检查点和数据集。

然后您可以使用我们在 [examples/scripts](./examples/scripts/) 目录中提供的启动脚本，或者使用以下命令启动训练：

```bash 
deepspeed ./train_sft.py \
   --max_len 4096 \
   --dataset Open-Orca/OpenOrca \
   --input_key question \
   --output_key response \
   --input_template 'User: {}\nAssistant: ' \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --pretrain meta-llama/Meta-Llama-3-8B \
   --save_path ./checkpoint/llama3-8b-sft \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --use_wandb {wandb_token}

# 支持 HF tokenizer.apply_chat_template.
# --apply_chat_template 
# --input_key {JSON Key}
# --tokenizer_chat_template {HF Chat Template}

# 支持 SFT 样本 packing
# --packing_samples

# 也可用于 continued pre-training
# --pretrain_mode
```

> [!NOTE]
> OpenRLHF SFT 支持的 `--packing_samples` [基于 `--flash_attn`](https://github.com/MeetKai/functionary/tree/main/functionary/train/packing)

### Reward Model Training
```bash
deepspeed ./train_rm.py \
   --save_path ./checkpoint/llama3-8b-rm \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain OpenLLMAI/Llama-3-8b-sft-mixture \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 9e-6 \
   --dataset OpenLLMAI/preference_dataset_mixture2_and_safe_pku \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb {wandb_token}
```

### 不使用 Ray 的 PPO

```bash
deepspeed ./train_ppo.py \
  --pretrain OpenLLMAI/Llama-3-8b-sft-mixture \
  --reward_pretrain OpenLLMAI/Llama-3-8b-rm-mixture \
  --save_path ./checkpoint/llama-3-8b-rlhf \
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
  --input_key context_messages \
  --apply_chat_template \
  --max_samples 100000 \
  --normalize_reward \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb {wandb_token}
```

### 使用 Ray 和 vLLM 的 PPO

为了提高 RLHF 训练速度或支持 70B 模型，我们可以使用 Ray 和 vLLM 加速的 PPO

```bash
# 在容器中启动 Ray 的主节点
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# 如果要在更多节点上启动 Ray，请使用
ray start --address {MASTER-NODE-ADDRESS}:6379 --num-gpus 8

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"working_dir": "/openrlhf", "pip": "/openrlhf/requirements.txt"}' \
  -- python3 examples/train_ppo_ray.py \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 2 \
  --reward_num_nodes 1 \
  --reward_num_gpus_per_node 2 \
  --critic_num_nodes 1 \
  --critic_num_gpus_per_node 2 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 2 \
  --vllm_num_engines 2 \
  --vllm_tensor_parallel_size 2 \
  --colocate_critic_reward \
  --colocate_actor_ref \
  --ref_reward_offload \
  --pretrain OpenLLMAI/Llama-3-8b-sft-mixture \
  --reward_pretrain OpenLLMAI/Llama-3-8b-rm-mixture \
  --save_path /openrlhf/examples/checkpoint/llama3-8b-rlhf \
  --micro_train_batch_size 8 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 16 \
  --rollout_batch_size 1024 \
  --max_samples 100000 \
  --max_epochs 1 \
  --prompt_max_len 1024 \
  --generate_max_len 1024 \
  --zero_stage 3 \
  --bf16 \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.01 \
  --prompt_data OpenLLMAI/prompt-collection-v0.1 \
  --input_key context_messages \
  --apply_chat_template \
  --normalize_reward \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb {wandb_token}

```

> [!NOTE]
> 我们推荐使用 vLLM 0.4.2，因为 0.4.3+ 版本目前仅支持通过 Gloo 进行权重同步（DeepSpeed => vLLM）（`--vllm_sync_backend gloo`）。
> 设置 `--vllm_num_engines 0` 则是不使用 vLLM engine

所有支持算法的启动脚本和文档在 [example/scripts](./examples/scripts/) 和 [Documents - Usage](https://openrlhf.readthedocs.io/en/latest/usage.html)


## 性能
我们通过启用Adam卸载、奖励模型(RM)和参考模型(Ref)卸载等技术,尽可能优化了DSChat的性能,从而在推理阶段增加小批量大小并避免内存不足问题。我们甚至修复了DSChat中的一些bug,以启用LLaMA2的混合引擎(HE)。使用优化后的DSChat和OpenRLHF训练1024个提示需要1个PPO轮次的平均时间(秒)如下:

| **Size** | **NVIDIA A800 GPUs** | **Optimized DSChat (with  Hybrid Engine)** | **OpenRLHF** | **Speedup** |
| :---: | :---: | :---: | :---: | :---: |
| 7B | 16 | 855.09 | 471.11 | 1.82x |
| 13B | 32 | 1528.93 | 608.93 | 2.5x |
| 34B | 32 | 3634.98 | 1526.4 | 2.4x |
| 70B | 32 | 10407.0 | 4488.53 | 2.3x |

## 调优指南
为了获得最佳的性能，我们建议您分配更多的节点给 vLLM Engine。例如，对于 70B 模型以及 32 张 A100，建议分配 16 张以上 A100 给 vLLM Engine，8 张给 Actor 模型，以及最后 8 张给 Critic 模型，同时开启 `--colocate_critic_reward`, `--colocate_actor_ref` 和 `--ref_reward_offload` 选项合并部分节点。最后您应该尽可能增大 `--rollout_micro_batch_size` ，以及减小 vLLM 的 TP 切分数量，同时避免 Reward/Reference 模型 forward 时内存不足。训练阶段的 `micro_train_batch_size` 也是越大越好。为 vLLM 开启 `enable_prefix_caching` 当 `n_samples_per_prompt > 1`.


## 加入我们

**如何加入？**

1. 通过开源组织邮箱 xianyuai@openllmai.top 和个人联系邮箱 janhu9527@gmail.com (PIC) 向我们发送邮件。请包含以下信息：
   - 您的姓名
   - 您的 GitHub 用户名
   - 您感兴趣的领域
   - 您在 NLP 和/或 AI 相关的技能和经验
2. 您也可以通过官方 GitHub [OpenRLHF ↗](https://github.com/openllmai/OpenRLHF) 项目页面加入我们。只需创建一个关于您想要贡献的兴趣的 issue，我们会与您联系。

**您能做什么？**

1. 加入团队，参与 OpenRLHF 项目的开发。
2. 通过提交 pull 请求来为项目做出贡献。
3. 帮助改进文档，修复 bugs 或创建新功能。
4. 分享项目并帮助我们发展社区。

## 赞助我们

您的赞助可以帮助我们维护和改进 OpenRLHF。如果您觉得这个项目有用，请考虑赞助我们。您可以在 [Open Collective ↗](https://opencollective.com/openllmai) 上赞助我们。

## 星图

[![Star History Chart](https://api.star-history.com/svg?repos=openllmai/OpenRLHF&type=Date)](https://star-history.com/#openllmai/OpenRLHF&Date)

## 贡献者

非常感谢所有的贡献者！如果您想贡献，请随时创建 pull 请求或创建 issue。

<a href="https://github.com/openllmai/OpenRLHF/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=openllmai/OpenRLHF" />
</a>

## 引用与致谢

我们想要对以下项目和组织在 AI 和 NLP 领域的贡献表示感谢：

- [Hugging Face Transformers ↗](https://github.com/huggingface/transformers)
- [OpenAI GPT ↗](https://github.com/openai/gpt-3)
- [LLaMA2 ↗](https://ai.meta.com/llama/)
- [DeepSpeed ↗](https://github.com/microsoft/DeepSpeed)
- [Ray ↗](https://github.com/ray-project/ray)

我们的项目还想要感谢 [ColossalChat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat) 和 [DeepSpeedChat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)。在项目的早期阶段，我们参考了他们的代码设计。

## 引用
```
@article{hu2024openrlhf,
  title={OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework},
  author={Jian Hu and Xibin Wu and Weixun Wang and Xianyu and Dehao Zhang and Yu Cao},
  journal={arXiv preprint arXiv:2405.11143},
  year={2024}
}
```


______________________________________________________________________

*OpenRLHF © 2024 OpenLLMAI. 版权所有。*
