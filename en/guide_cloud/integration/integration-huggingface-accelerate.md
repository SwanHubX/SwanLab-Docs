# 🤗HuggingFace Accelerate

HuggingFace's [accelerate](https://huggingface.co/docs/accelerate/index) is an open-source library designed to simplify and optimize the training and inference of deep learning models.

> 🚀 A straightforward way to launch, train, and use PyTorch models on almost any device and distributed configuration, with support for automatic mixed precision (including fp8), and easily configurable FSDP and DeepSpeed.

It provides tools for efficient distributed training and inference, enabling developers to deploy and accelerate models across different hardware devices with ease. With just a few lines of code changes, existing training scripts can be seamlessly integrated into platforms like `torch_xla` and `torch.distributed`, eliminating the need to grapple with complex distributed computing architectures. This boosts productivity and enhances model performance.

![hf-accelerate-image](./huggingface_accelerate/logo.png)

You can use `accelerate` for rapid model training while leveraging SwanLab for experiment tracking and visualization.

> Versions of `accelerate` >=1.8.0 officially support SwanLab integration.  
> If your version is below 1.8.0, please use the **SwanLabTracker integration**.

## 1. Integration in Two Lines of Code

```python {4,9}
from accelerate import Accelerator

# Instruct the Accelerator object to use SwanLab for logging
accelerator = Accelerator(log_with="swanlab")

# Initialize your SwanLab experiment, passing SwanLab parameters and any configuration
accelerator.init_trackers(
    ...
    init_kwargs={"swanlab": {"experiment_name": "hello_world"}}
    )
```

::: warning Additional Notes
1. The SwanLab project name is specified by the `project_name` parameter in `accelerator.init_trackers`.
2. The `swanlab` dictionary passed to `init_kwargs` accepts key-value pairs identical to the arguments of `swanlab.init` (except for `project`).
:::

Minimal working example:

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

## 2. SwanLabTracker Integration

For versions of `accelerate` <1.8.0, use the `SwanLabTracker` integration.

### 2.1 Import

```bash
from swanlab.integration.accelerate import SwanLabTracker
```

### 2.2 Specify the Logger During Accelerator Initialization

```python (1,7,9,12)
from swanlab.integration.accelerate import SwanLabTracker
from accelerate import Accelerator

...

# Create a SwanLab logger
tracker = SwanLabTracker("YOUR_SMART_PROJECT_NAME")
# Pass it to Accelerator
accelerator = Accelerator(log_with=tracker)

# Initialize all loggers
accelerator.init_trackers("YOUR_SMART_PROJECT_NAME", config=config)

# training code
...
```

- Although the project name is set twice in the above code, only the first setting takes effect.  
- Explicitly calling `init_trackers` to initialize all loggers is part of `accelerate`'s mechanism. The second project name setting is only relevant when multiple loggers are used, such as initializing built-in loggers.

### 2.3 Complete Example Code

Below is an example of using `accelerate` for CIFAR10 classification with SwanLab for logging:

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