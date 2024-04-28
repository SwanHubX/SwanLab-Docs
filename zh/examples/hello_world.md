# Hello World

这是一个入门案例，是一个最简的深度学习训练模拟。

## 环境准备

```bash
pip install swanlab
```

## 完整代码

```python
import swanlab
import random

offset = random.random() / 5

# 初始化SwanLab
run = swanlab.init(
    project="my-project",
    config={
        "learning_rate": 0.01,
        "epochs": 10,
    },
)

# 模拟训练过程
for epoch in range(2, run.config.epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")

    swanlab.log({"accuracy": acc, "loss": loss})  # 记录指标
```