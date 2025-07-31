# 🤗HuggingFace Accelerate

HuggingFace 的 [accelerate](https://huggingface.co/docs/accelerate/index) 是一个简化和优化深度学习模型训练与推理的开源库。

> 🚀在几乎任何设备和分布式配置上启动、训练和使用PyTorch模型的简单方法，支持自动混合精度(包括fp8)，以及易于配置的FSDP和DeepSpeed

它提供了高效的分布式训练和推理的工具，使开发者能够更轻松地在不同硬件设备上部署和加速模型。通过简单的几行代码改动，就可以轻松将现有的训练代码集成进 `torch_xla` 和 `torch.distributed` 这类平台，而无需为复杂的分布式计算架构烦恼，从而提升工作效率和模型性能。

![hf-accelerate-image](./huggingface_accelerate/logo.png)

你可以使用`accelerate`快速进行模型训练，同时使用SwanLab进行实验跟踪与可视化。

> `accelerate`>=1.8.0 的版本，已官方集成了swanlab  
> 如果你的版本低于1.8.0，请使用 **SwanLabTracker集成**


## 1. 两行代码完成集成

```python {4,9}
from accelerate import Accelerator

# 告诉 Accelerator 对象使用 swanlab 进行日志记录
accelerator = Accelerator(log_with="swanlab")

# 初始化您的 swanlab 实验，传递 swanlab 参数和任何配置信息
accelerator.init_trackers(
    ...
    init_kwargs={"swanlab": {"experiment_name": "hello_world"}}
    )
```

::: warning 补充信息
1. swanlab项目名由`accelerator.init_trackers`的`project_name`参数指定
2. 向`init_kwargs`传递的`swanlab`字典，key-value和`swanlab.init`的参数完全一致（除了project）。
:::

最小能跑代码：

```python {4,10}
from accelerate import Accelerator

# Tell the Accelerator object to log with swanlab
accelerator = Accelerator(log_with="swanlab")

# Initialise your swanlab experiment, passing swanlab parameters and any config information
accelerator.init_trackers(
    project_name="accelerator",
    config={"dropout": 0.1, "learning_rate": 1e-2},
    init_kwargs={"swanlab": {"experiment_name": "hello_world"}}
    )

for i in range(100):
    # Log to swanlab by calling `accelerator.log`, `step` is optional
    accelerator.log({"train_loss": 1.12, "valid_loss": 0.8}, step=i+1)

# Make sure that the swanlab tracker finishes correctly
accelerator.end_training()
```

## 2. SwanLabTracker集成

如果你使用的是`accelerate<1.8.0`的版本，则可以使用SwanLabCallback集成。

### 2.1 引入

```bash
from swanlab.integration.accelerate import SwanLabTracke
```


### 2.2 在初始化accelerate时指定日志记录器

```python (1,7,9,12)
from swanlab.integration.accelerate import SwanLabTracker
from accelerate import Accelerator

...

# 创建SwanLab日志记录器
tracker = SwanLabTracker("YOUR_SMART_PROJECT_NAME")
# 传入Accelerator
accelerator = Accelerator(log_with=tracker)

# 初始化所有日志记录器
accelerator.init_trackers("YOUR_SMART_PROJECT_NAME", config=config)

# training code
...
```

- 虽然上面的代码两次设定了项目名，实际上只有第一个项目名设置才起了作用

- 显式调用`init_trackers`来初始化所有日志记录是`accelerate`的机制，第二次设置的项目名是当有多个日志记录器时,初始化内置的日志记录器的情况下才会用到。

### 2.3 完整案例代码

下面是一个使用accelerate进行cifar10分类，并使用SwanLab进行日志跟踪的案例：

```python (10,45,46,47,71,90)
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from accelerate import Accelerator
from accelerate.logging import get_logger
import swanlab
from swanlab.integration.accelerate import SwanLabTracker


def main():
    # hyperparameters
    config = {
        "num_epoch": 5,
        "batch_num": 16,
        "learning_rate": 1e-3,
    }

    # Download the raw CIFAR-10 data.
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    BATCH_SIZE = config["batch_num"]
    my_training_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    my_testing_dataloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Using resnet18 model, make simple changes to fit the data set
    my_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    my_model.conv1 = torch.nn.Conv2d(my_model.conv1.in_channels, my_model.conv1.out_channels, 3, 1, 1)
    my_model.maxpool = torch.nn.Identity()
    my_model.fc = torch.nn.Linear(my_model.fc.in_features, 10)

    # Criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    my_optimizer = torch.optim.SGD(my_model.parameters(), lr=config["learning_rate"], momentum=0.9)

    # Init accelerate with swanlab tracker
    tracker = SwanLabTracker("CIFAR10_TRAING")
    accelerator = Accelerator(log_with=tracker)
    accelerator.init_trackers("CIFAR10_TRAING", config=config)
    my_model, my_optimizer, my_training_dataloader, my_testing_dataloader = accelerator.prepare(
        my_model, my_optimizer, my_training_dataloader, my_testing_dataloader
    )
    device = accelerator.device
    my_model.to(device)

    # Get logger
    logger = get_logger(__name__)

    # Begin training

    for ep in range(config["num_epoch"]):
        # train model
        if accelerator.is_local_main_process:
            print(f"begin epoch {ep} training...")
        step = 0
        for stp, data in enumerate(my_training_dataloader):
            my_optimizer.zero_grad()
            inputs, targets = data
            outputs = my_model(inputs)
            loss = criterion(outputs, targets)
            accelerator.backward(loss)
            my_optimizer.step()
            accelerator.log({"training_loss": loss, "epoch_num": ep})
            if accelerator.is_local_main_process:
                print(f"train epoch {ep} [{stp}/{len(my_training_dataloader)}] | train loss {loss}")

        # eval model
        if accelerator.is_local_main_process:
            print(f"begin epoch {ep} evaluating...")
        with torch.no_grad():
            total_acc_num = 0
            for stp, (inputs, targets) in enumerate(my_testing_dataloader):
                predictions = my_model(inputs)
                predictions = torch.argmax(predictions, dim=-1)
                # Gather all predictions and targets
                all_predictions, all_targets = accelerator.gather_for_metrics((predictions, targets))
                acc_num = (all_predictions.long() == all_targets.long()).sum()
                total_acc_num += acc_num
                if accelerator.is_local_main_process:
                    print(f"eval epoch {ep} [{stp}/{len(my_testing_dataloader)}] | val acc {acc_num/len(all_targets)}")

            accelerator.log({"eval acc": total_acc_num / len(my_testing_dataloader.dataset)})

    accelerator.wait_for_everyone()
    accelerator.save_model(my_model, "cifar_cls.pth")

    accelerator.end_training()


if __name__ == "__main__":
    main()
```
