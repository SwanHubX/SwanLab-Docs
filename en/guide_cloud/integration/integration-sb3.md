# Stable-Baseline3

Stable Baselines3 (SB3) is an open-source reinforcement learning library built on the PyTorch framework. It is the successor to the Stable Baselines project, aiming to provide a set of reliable and well-tested RL algorithm implementations for research and application. Stable Baselines3 is primarily used in fields such as robotics control, game AI, autonomous driving, and financial trading.

![sb3](/assets/ig-sb3.png)

You can use SB3 to quickly train models while using SwanLab for experiment tracking and visualization.

## 1. Import SwanLabCallback

```python
from swanlab.integration.sb3 import SwanLabCallback
```

**SwanLabCallback** is a logging class adapted for Stable Baselines3.

**SwanLabCallback** can define parameters such as:

- `project`, `experiment_name`, `description`, and other parameters consistent with `swanlab.init`, used for initializing the SwanLab project.
- You can also create the project externally via `swanlab.init`, and the integration will log the experiment to the project you created externally.

## 2. Pass to model.learn

```python (1,7)
from swanlab.integration.sb3 import SwanLabCallback

...

model.learn(
    ...
    callback=SwanLabCallback(),
)
```

Pass the `SwanLabCallback` instance to the `callback` parameter of `model.learn` to start tracking.

## 3. Complete Example Code

Below is a simple training example for a PPO model, using SwanLab for training visualization and monitoring:

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