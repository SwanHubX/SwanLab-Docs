# 记录分布式训练指标

SwanLab 支持记录分布式训练实验的指标，帮助你追踪多个 GPU 或机器上的训练过程。

## 分布式训练场景

在分布式训练中，训练任务通常会在多个进程、多个 GPU 甚至多台机器上同时实验。SwanLab 提供了以下方式來记录这些训练指标：

1. **仅记录主进程**：只记录 rank 0（主进程/协调进程）的指标
2. **多进程分别记录**：每个进程创建独立的实验，使用 `group` 参数将它们关联到同一个实验

> **注意**：SwanLab 将在未来支持将所有进程记录到单个实验中。所有进程需要分别记录为独立的实验。

## 方法一：仅记录主进程

在 PyTorch DDP 等分布式训练框架中，通常只需要记录主进程（rank 0）的指标，因为损失值、梯度和参数等信息在主进程上可用。

```python
import os
import torch
import torch.distributed as dist
import swanlab

def main():
    # 初始化分布式训练
    dist.init_process_group(backend='nccl')

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    # 只在主进程初始化 SwanLab
    if local_rank == 0:
        swanlab.init(
            project="distributed_training",
            experiment_name="distributed_training",
        )

    # 训练循环
    for epoch in range(10):
        # ... 训练代码 ...

        # 只在主进程记录指标
        if local_rank == 0:
            swanlab.log({
                "loss": loss.item(),
                "accuracy": accuracy.item()
            })

    # 训练结束
    if local_rank == 0:
        swanlab.finish()

if __name__ == "__main__":
    main()
```

## 方法二：多进程分别记录

每个进程创建独立的实验，使用 `group` 参数将它们关联到同一个实验组，使用 `job_type` 参数区分不同类型的节点（如 "main" 和 "worker"）。

```python
import os
import torch
import torch.distributed as dist
import swanlab

def setup(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def train(rank, world_size, group_name):
    """每个进程的训练函数"""
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    # 每个进程初始化独立的实验
    swanlab.init(
        project="distributed_training",
        experiment_name="distributed_experiment",
        group=group_name,  # 使用相同的 group 名称关联到同一个实验
        job_type="train" if rank == 0 else "worker"  # 区分主进程和工作进程
    )

    # 训练循环
    for epoch in range(10):
        # ... 训练代码 ...

        # 每个进程记录自己的指标
        swanlab.log({
            "loss": loss.item(),
            "epoch": epoch,
            "rank": rank
        })

    swanlab.finish()
    cleanup()

def main():
    world_size = 2  # 使用 2 个进程
    group_name = f"exp_{swanlab.run.public.run_id[:8]}"

    # 使用 torchrun 或 mpirun 启动
    import torch.multiprocessing as mp
    mp.spawn(
        train,
        args=(world_size, group_name),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
```

## 使用环境变量设置

除了在代码中设置，你也可以使用环境变量来配置分布式训练的记录：

```python
import os

# 在主进程设置
os.environ["SWANLAB_GROUP"] = "distributed_exp_001"
os.environ["SWANLAB_JOB_TYPE"] = "train"

import swanlab
swanlab.init(experiment="distributed_experiment")
```

支持的分布式训练环境变量：

| 环境变量 | 描述 |
|---------|------|
| `SWANLAB_GROUP` | 实验分组名称，用于关联多个实验 |
| `SWANLAB_JOB_TYPE` | 任务类型，如 "train"、"eval"、"inference" |
| `SWANLAB_NAME` | 实验名称 |
| `SWANLAB_DESCRIPTION` | 实验描述 |

## 多节点训练

对于跨多台机器的训练，每台机器上的每个进程都应该：

1. 使用相同的 `group` 名称
2. 使用有意义的 `job_type` 区分不同角色（如 "node_0"、"node_1"）

```python
import os
import swanlab

# 获取当前节点标识
node_id = os.environ.get("NODE_RANK", "0")
rank = os.environ.get("RANK", "0")

swanlab.init(
    experiment="multi_node_training",
    group=f"distributed_exp_{os.environ.get('EXP_ID', '001')}",
    job_type=f"node_{node_id}_rank_{rank}"
)

swanlab.log({"node": node_id, "rank": rank})
swanlab.finish()
```

## 常见问题

### 1. 指标记录顺序

SwanLab 不保证多进程指标记录的有序性，建议在应用层处理同步逻辑。

### 2. 启动时卡住

确保在所有进程中都正确调用 `swanlab.finish()` 来结束记录。

### 3. 资源占用

在多进程场景下，每个进程都会创建独立的网络连接。确保系统有足够的文件描述符和网络资源。

## 推荐实践

1. **使用一致的 group 名称**：确保同一分布式训练任务中的所有进程使用相同的 group
2. **设置 job_type**：使用 job_type 区分不同类型的进程，便于后续筛选和分析
3. **记录 rank 信息**：在日志中包含 rank 信息，便于区分不同进程的指标
4. **适当结束实验**：在训练结束后调用 `swanlab.finish()` 确保数据正确上传
