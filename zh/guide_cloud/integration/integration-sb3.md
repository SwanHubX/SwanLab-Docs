# Stable-Baseline3

[![](/assets/colab.svg)](https://colab.research.google.com/drive/1JfU4oCKCS7FQE_AXqZ3k9Bt1vmK-6pMO?usp=sharing)


Stable Baselines3 (SB3) 是一个强化学习的开源库，基于 PyTorch 框架构建。它是 Stable Baselines 项目的继任者，旨在提供一组可靠且经过良好测试的RL算法实现，便于研究和应用。StableBaseline3主要被应用于机器人控制、游戏AI、自动驾驶、金融交易等领域。

![sb3](/assets/ig-sb3.png)

你可以使用sb3快速进行模型训练，同时使用SwanLab进行实验跟踪与可视化。

## 1.引入SwanLabCallback

```python
from swanlab.integration.sb3 import SwanLabCallback
```

**SwanLabCallback**是适配于 Stable Baselines3 的日志记录类。

**SwanLabCallback**可以定义的参数有：

- project、experiment_name、description 等与 swanlab.init 效果一致的参数, 用于SwanLab项目的初始化。
- 你也可以在外部通过`swanlab.init`创建项目，集成会将实验记录到你在外部创建的项目中。

## 2.传入model.learn

```python (1,7)
from swanlab.integration.sb3 import SwanLabCallback

...

model.learn(
    ...
    callback=SwanLabCallback(),
)
```
在`model.learn`的`callback`参数传入`SwanLabCallback`实例，即可开始跟踪。


## 3.完整案例代码

下面是一个PPO模型的简单训练案例，使用SwanLab做训练可视化和监控：

```python (6,31)
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import swanlab
from swanlab.integration.sb3 import SwanLabCallback


config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 25000,
    "env_name": "CartPole-v1",
}


def make_env():
    env = gym.make(config["env_name"], render_mode="rgb_array")
    env = Monitor(env)
    return env


env = DummyVecEnv([make_env])
model = PPO(
    config["policy_type"],
    env,
    verbose=1,
)

model.learn(
    total_timesteps=config["total_timesteps"],
    callback=SwanLabCallback(
        project="PPO",
        experiment_name="MlpPolicy",
        verbose=2,
    ),
)

swanlab.finish()

```