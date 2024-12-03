# Hello World

This is an introductory case, a minimal simulation of deep learning training.

## Environment Setup

```bash
pip install swanlab
```

## Complete Code

```python
import swanlab
import random

offset = random.random() / 5

# Initialize SwanLab
run = swanlab.init(
    project="my-project",
    config={
        "learning_rate": 0.01,
        "epochs": 10,
    },
)

# Simulate the training process
for epoch in range(2, run.config.epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")

    swanlab.log({"accuracy": acc, "loss": loss})  # Record metrics
```