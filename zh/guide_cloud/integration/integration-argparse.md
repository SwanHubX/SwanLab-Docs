# Argparse

`argparse` 是 Python 标准库中的一个模块，用于解析命令行参数和选项。通过 argparse，开发者可以轻松地编写用户友好的命令行接口，定义命令行参数的名称、类型、默认值、帮助信息等。

`argparse` 与swanlab的集成非常简单，直接将创建好的argparse对象传递给swanlab.config，即可记录为超参数：

```python
import argparse
import swanlab

# 初始化Argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=20)
parser.add_argument('--lr', default=0.001)
args = parser.parse_args()

swanlab.init(config=args)
```

运行案例：
```bash
python main.py --epochs 100 --lr 1e-4
```

![alt text](/assets/ig-argparse.png)
