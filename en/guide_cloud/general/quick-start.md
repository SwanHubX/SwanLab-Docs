
# 🚀Quick Start

Install SwanLab and start tracking your AI experiments in minutes

![quick-start-1](/assets/quick-start.png)


## 1. Install SwanLab

Install the swanlab library on a computer with a Python 3 environment using [pip](https://pip.pypa.io/en/stable/).

Open the command line and enter:

```bash
pip install swanlab
```
Press Enter and wait for the installation to complete.


## 2. Log in

> If you don't have a SwanLab account yet, please register for free on the [official website](https://swanlab.cn).

Open the command line and enter:

```bash
swanlab login
```

When you see the following prompt:

```bash
swanlab: Logging into swanlab cloud.
swanlab: You can find your API key at: https://swanlab.cn/settings
swanlab: Paste an API key from your profile and hit enter, or press 'CTRL-C' to quit:
```

Copy your API Key from the [user settings](https://swanlab.cn/settings) page, paste it and press Enter to complete the login. You don't need to log in again after that.


> If your computer does not support the `swanlab login` method, you can also log in using a Python script:
> import swanlab  
> swanlab.login(api_key="Your API Key")


## 3. Start an experiment and track hyperparameters

In the Python script, we use swanlab.init to create a SwanLab experiment and pass a dictionary containing hyperparameter key-value pairs to the config parameter:


```python
import swanlab

run = swanlab.init(
    # Set project
    project="my-project",
    # Track hyperparameters and experiment metadata
    config={
        "learning_rate": 0.01,
        "epochs": 10,
    },
)
```

`run` is the fundamental component of SwanLab, and you will often use it to record and track experiment metrics.


## 4. Record experiment metrics

In the Python script, use `swanlab.log` to record experiment metrics (such as accuracy and loss value).

The usage is to pass a dictionary containing the metrics to `swanlab.log`:

```python
swanlab.log({"accuracy": acc, "loss": loss})
```

## 5. Complete code, view the visualization dashboard online

We integrate the above steps into the complete code shown below:

```python (5,25)
import swanlab
import random

# Initialize SwanLab
run = swanlab.init(
    # Set project
    project="my-project",
    # Track hyperparameters and experiment metadata
    config={
        "learning_rate": 0.01,
        "epochs": 10,
    },
)

print(f"Learning rate is {run.config.learning_rate}")

offset = random.random() / 5

# Simulate training process
for epoch in range(2, run.config.epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    # Record metrics
    swanlab.log({"accuracy": acc, "loss": loss})
```

Run the code, visit [SwanLab](https://swanlab.cn), and see the improvement of the metrics (accuracy and loss value) you recorded using SwanLab in each training step.


![quick-start-1](/assets/quick-start-1.jpg)


## What's next

1. See how SwanLab records multimedia content (images, audio, text,...)
2. See how SwanLab records the [MNIST handwritten recognition](/zh/examples/mnist.md) case
3. See integration with other frameworks
4. See how to collaborate with your team through SwanLab

## FAQ

### 1. Where can I find my API Key?
After logging in to the SwanLab website, the API Key will be displayed on the [user settings](https://swanlab.cn/settings) page.

### 2. Can I use SwanLab offline?
Yes, please refer to the [self-hosting](/zh/guide_cloud/self_host/offline-board.md) section for the specific process.
