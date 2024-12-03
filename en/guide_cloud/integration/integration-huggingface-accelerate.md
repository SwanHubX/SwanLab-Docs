# 🤗HuggingFace Accelerate

HuggingFace's [accelerate](https://huggingface.co/docs/accelerate/index) is an open-source library that simplifies and optimizes the training and inference of deep learning models.

> 🚀 A simple way to launch, train, and use PyTorch models on almost any device and distributed configuration, supporting automatic mixed precision (including fp8), as well as easy-to-configure FSDP and DeepSpeed.

It provides efficient tools for distributed training and inference, making it easier for developers to deploy and accelerate models on different hardware devices. With just a few lines of code changes, you can easily integrate existing training code into platforms like `torch_xla` and `torch.distributed` without worrying about complex distributed computing architectures, thereby improving work efficiency and model performance.

![hf-accelerate-image](/assets/ig-huggingface-accelerate.png)

You can use `accelerate` to quickly train models while using SwanLab for experiment tracking and visualization.

## 1. Import

```python
from swanlab.integration.accelerate import SwanLabTracker
```

## 2. Specify the Logger When Initializing Accelerate

```python (1,7,9,12)
from swanlab.integration.accelerate import SwanLabTracker
from accelerate import Accelerator

...

# Create SwanLab logger
tracker = SwanLabTracker("YOUR_SMART_PROJECT_NAME")
# Pass to Accelerator
accelerator = Accelerator(log_with=tracker)

# Initialize all loggers
accelerator.init_trackers("YOUR_SMART_PROJECT_NAME", config=config)

# training code
...
```

- Although the above code sets the project name twice, only the first project name setting takes effect.

- Explicitly calling `init_trackers` to initialize all loggers is a mechanism of `accelerate`. The second project name setting is used when there are multiple loggers and the built-in logger needs to be initialized.

## 3. Complete Example Code

Below is an example of using accelerate for CIFAR10 classification and using SwanLab for logging tracking:

```python (10,45,46,47,71,89)
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
    # Hyperparameters
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