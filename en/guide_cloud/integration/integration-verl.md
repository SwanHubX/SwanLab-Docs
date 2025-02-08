# verl

## Introduce of verl

verl is a flexible, efficient and production-ready RL training framework designed for large language models (LLMs) post-training. It is an open source implementation of the HybridFlow paper.

<div style="text-align: center;">
    <img src="./verl/verl_logo.svg" alt="verl_logo" style="width: 150%;">
</div>

verl is flexible and easy to use with:

* **Easy extension of diverse RL algorithms:** The Hybrid programming model combines the strengths of single-controller and multi-controller paradigms to enable flexible representation and efficient execution of complex Post-Training dataflows. Allowing users to build RL dataflows in a few lines of code.

* **Seamless integration of existing LLM infra with modular APIs:** Decouples computation and data dependencies, enabling seamless integration with existing LLM frameworks, such as PyTorch FSDP, Megatron-LM and vLLM. Moreover, users can easily extend to other LLM training and inference frameworks.

* **Flexible device mapping and parallelism:** Supports various placement of models onto different sets of GPUs for efficient resource utilization and scalability across different cluster sizes.

Readily integration with popular HuggingFace models

verl is fast with:

* **State-of-the-art throughput:** By seamlessly integrating existing SOTA LLM training and inference frameworks, verl achieves high generation and training throughput.

* **Efficient actor model resharding with 3D-HybridEngine:** Eliminates memory redundancy and significantly reduces communication overhead during transitions between training and generation phases.

You can found more information with follow reference:

* verl GitHub link: [https://github.com/volcengine/verl](https://github.com/volcengine/verl)

* Office documents: [https://verl.readthedocs.io/en/latest/index.html](https://verl.readthedocs.io/en/latest/index.html)

* HybridFlow Paper Link: [https://arxiv.org/pdf/2409.19256v2](https://arxiv.org/pdf/2409.19256v2)

## Environment Setup

Required Environment:

* Python: Version >= 3.9

* CUDA: Version >= 12.1

Refer to the official verl documentation for installation instructions: [https://verl.readthedocs.io/en/latest/start/install.html](https://verl.readthedocs.io/en/latest/start/install.html)

Additionally, SwanLab needs to be installed:

```bash
pip install -U swanlab
```

## Use verl with SwanLab

### **Step 1: Set Up Online Tracking (Optional)**

To use SwanLab's online tracking, log in to the [SwanLab website](https://swanlab.cn) and obtain your API key from the [Settings page](https://swanlab.cn/space/~/settings). Then, authenticate using the following command:

```bash
swanlab login
```

If you prefer offline mode, skip this step.

### **Step 2: Configure SwanLab as the Logger**

To enable SwanLab as the experiment tracker, add `trainer.logger=['swanlab']` to your training command. For example, using the [Post-train a LLM using PPO with GSM8K dataset](https://verl.readthedocs.io/en/latest/start/quickstart.html) workflow:

```bash
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$HOME/data/gsm8k/train.parquet \
 data.val_files=$HOME/data/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.val_batch_size=1312 \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=['console','swanlab'] \
 +trainer.val_before_train=False \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 2>&1 | tee verl_demo.log
```

If you are not logged in, you will be prompted to choose a tracking mode:

1. **Cloud Mode**: Upload logs to SwanLab's cloud platform.
2. **Cloud-Only Mode**: Upload logs to the cloud but do not save them locally.
3. **Local Mode**: Save logs locally for offline tracking.

![select](./verl/select.png)

Alternatively, you can configure SwanLab using environment variables:

```bash
export SWANLAB_API_KEY=<your_api_key>          # Set API key for online tracking
export SWANLAB_LOG_DIR=<local_log_path>        # Set local log directory
export SWANLAB_MODE=<mode>                    # Set tracking mode: cloud (default), cloud-only, local, or disabled
```

### **Step 3: View Training Logs**

After logging in, you will see a confirmation message:
![track](./verl/track.png)

* **Online Tracking**: View logs on the [SwanLab website](https://swanlab.cn).
  ![remote](./verl/remote.png)
  For more details, refer to the [SwanLab Cloud Documentation](https://docs.swanlab.cn/guide_cloud/experiment_track/view-result.html).

* **Offline Tracking**: Use the local dashboard to visualize logs:

  ```bash
  swanlab watch
  ```

  For advanced configurations, such as setting a custom port, refer to the [Offline Dashboard Documentation](https://docs.swanlab.cn/guide_cloud/self_host/offline-board.html) and [CLI Documentation](https://docs.swanlab.cn/api/cli-swanlab-watch.html#%E8%AE%BE%E7%BD%AEip%E5%92%8C%E7%AB%AF%E5%8F%A3%E5%8F%B7).
