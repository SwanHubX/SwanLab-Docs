# Logging Distributed Training Metrics

SwanLab supports logging distributed training experiments, helping you track training processes across multiple GPUs or machines.

## Distributed Training Scenarios

In distributed training, training tasks typically run concurrently across multiple processes, GPUs, or even multiple machines. SwanLab provides the following methods for logging these training metrics:

1. **Log main process only**: Only log metrics from rank 0 (main/coordinator process)
2. **Log each process separately**: Each process creates its own run, using the `group` parameter to associate them into the same experiment

> **Note**: SwanLab does not support logging all processes to a single run. Each process must be logged as a separate run.

## Method 1: Log Main Process Only

In distributed training frameworks like PyTorch DDP, you typically only need to log metrics from the main process (rank 0), as loss values, gradients, and parameters are available on rank 0.

```python
import os
import torch
import torch.distributed as dist
import swanlab

def main():
    # Initialize distributed training
    dist.init_process_group(backend='nccl')

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    # Only initialize SwanLab on the main process
    if local_rank == 0:
        swanlab.init(
            experiment="distributed_training",
            experiment_id="exp_001"
        )

    # Training loop
    for epoch in range(10):
        # ... training code ...

        # Only log metrics on the main process
        if local_rank == 0:
            swanlab.log({
                "loss": loss.item(),
                "accuracy": accuracy.item()
            })

    # Training finished
    if local_rank == 0:
        swanlab.finish()

if __name__ == "__main__":
    main()
```

## Method 2: Log Each Process Separately

Each process creates its own run, using the `group` parameter to associate them into the same experiment group, and the `job_type` parameter to distinguish different types of nodes (e.g., "main" and "worker").

```python
import os
import torch
import torch.distributed as dist
import swanlab

def setup(rank, world_size):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    """Cleanup distributed training environment"""
    dist.destroy_process_group()

def train(rank, world_size, group_name):
    """Training function for each process"""
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    # Each process initializes its own run
    swanlab.init(
        experiment="distributed_experiment",
        group=group_name,  # Use the same group name to associate with the same experiment
        job_type="train" if rank == 0 else "worker"  # Distinguish main process from worker processes
    )

    # Training loop
    for epoch in range(10):
        # ... training code ...

        # Each process logs its own metrics
        swanlab.log({
            "loss": loss.item(),
            "epoch": epoch,
            "rank": rank
        })

    swanlab.finish()
    cleanup()

def main():
    world_size = 2  # Use 2 processes
    group_name = f"exp_{swanlab.run.get_uuid()[:8]}"

    # Launch using torchrun or mpirun
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

## Using Environment Variables

In addition to setting in code, you can also use environment variables to configure distributed training logging:

```python
import os

# Set in the main process
os.environ["SWANLAB_GROUP"] = "distributed_exp_001"
os.environ["SWANLAB_JOB_TYPE"] = "train"

import swanlab
swanlab.init(experiment="distributed_experiment")
```

Supported environment variables for distributed training:

| Environment Variable | Description |
|---------------------|-------------|
| `SWANLAB_GROUP` | Experiment group name, used to associate multiple runs |
| `SWANLAB_JOB_TYPE` | Task type, such as "train", "eval", "inference" |
| `SWANLAB_NAME` | Run name |
| `SWANLAB_DESCRIPTION` | Run description |

## Multi-Node Training

For training across multiple machines, each process on each machine should:

1. Use the same `group` name
2. Use meaningful `job_type` to distinguish different roles (e.g., "node_0", "node_1")

```python
import os
import swanlab

# Get current node identifier
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

## FAQ

### 1. Metric Logging Order

SwanLab does not guarantee ordered metric logging across processes. It is recommended to handle synchronization logic at the application layer.

### 2. Hanging at Startup

Ensure that `swanlab.finish()` is correctly called in all processes to end logging.

### 3. Resource Usage

In multi-process scenarios, each process creates a separate network connection. Ensure the system has enough file descriptors and network resources.

## Best Practices

1. **Use consistent group names**: Ensure all processes in the same distributed training task use the same group
2. **Set job_type**: Use job_type to distinguish different types of processes for easier filtering and analysis
3. **Log rank information**: Include rank information in logs to distinguish metrics from different processes
4. **End runs properly**: Call `swanlab.finish()` after training to ensure data is uploaded correctly
