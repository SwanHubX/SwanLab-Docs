# 🤗HuggingFace Accelerate

🤗HuggingFace 的 [Accelerate](https://huggingface.co/docs/accelerate/index) 是一个简化并优化深度学习模型训练与推理的开源库。它提供了高效的分布式训练和推理的工具，使开发者能够更轻松地在不同硬件设备上部署和加速模型。通过简单的几行代码改动，就可以轻松将现有的训练代码集成进torch_xla 和 torch.distributed这类平台，而无需为复杂的分布式计算架构烦恼，从而提升工作效率和模型性能。

![hf-transformers-image](/assets/ig-huggingface-transformers.png)

你可以使用Transformers快速进行模型训练，同时使用SwanLab进行实验跟踪与可视化。

## 1. 引入

```python
from swanlab.integration.accelerate import SwanLabTracker
```

**SwanLabTracker**是适配于 Accelerate 的日志记录类。

**SwanLabTracker**可以定义的参数包括project_name、logging_dir 等与 swanlab.init 效果一致的参数, 用于SwanLab项目的初始化。

## 在初始化accelerate时指定日志记录器

```python (1,7,12)
from swanlab.integration.accelerate import SwanLabTracker
from accelerate import Accelerator

...

# 在初始化accelerate时指定日志记录器
tracker = SwanLabTracker("YOUR_SMART_PROJECT_NAME")
accelerator = Accelerator(log_with=tracker)

# 初始化所有日志记录器
accelerator.init_trackers("YOUR_SMART_PROJECT_NAME", config=config)

# training code
...
```

虽然上面的代码两次次设定了项目名，实际上只有第一个项目名设置才起了作用

显式调用init_trackers来初始化所有日志记录其是accelerate的机制，第二次设置的项目名是当有多个日志记录器时初始化内置的日志记录器时才会用到。

## 3. 完整案例代码

下面提供一个使用accelerate进行cifar10分类并且使用SwanLab进行日志跟踪的案例

```python (1,7,12)
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from tutils import open_dev_mode
import swanlab
from swanlab.integration.accelerate import SwanLabTracker

swanlab.login(open_dev_mode())

import torch
from torchvision.models import resnet18, ResNet18_Weights
import torchvision

from accelerate import Accelerator
from accelerate.logging import get_logger


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
