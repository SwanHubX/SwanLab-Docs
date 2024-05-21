# Weights & Biases

[Weights & Biases](https://github.com/wandb/wandb) (Wandb) 是一个用于机器学习和深度学习项目的实验跟踪、模型优化和协作平台。W&B 提供了强大的工具来记录和可视化实验结果，帮助数据科学家和研究人员更好地管理和分享他们的工作。

![wandb](/assets/ig-wandb.png)

你可以使用swanlab convert将Wandb上已存在的项目转换成SwanLab项目。

::: info
在当前版本暂仅支持转换标量图表。
:::

## 找到你的projecy、entity和runid

projecy、entity和runid是转换所需要的（runid是可选的）。  
project和entity的位置：
![alt text](/assets/ig-wandb-2.png)

runid的位置：

![alt text](/assets/ig-wandb-3.png)

## 方式一：命令行转换

首先，需要确保当前环境下，你已登录了wandb，并有权限访问目标项目。

转换命令行：

```bash
swanlab convert -t wandb --wb-project [WANDB_PROJECT_NAME] --wb-entity [WANDB_ENTITY]
```

支持的参数如下：

- `-t`: 转换类型，可选wandb与tensorboard。
- `--wb-project`：待转换的wandb项目名。
- `--wb-entity`：wandb项目所在的空间名。
- `--wb-runid`: wandb Run（项目下的某一个实验）的id。

如果不填写`--wb-runid`，则会将指定项目下的全部Run进行转换；如果填写，则只转换指定的Run。

## 方式二：代码内转换

```python
from swanlab.converter import WandbConverter

wb_converter = WandbConverter()
# wb_runid可选
wb_converter.run(wb_project="WANDB_PROJECT_NAME", wb_entity="WANDB_USERNAME")
```
效果与命令行转换一致。