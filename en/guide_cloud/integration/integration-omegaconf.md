# Omegaconf

OmegaConf is a Python library for handling configurations, especially useful in scenarios that require flexible configurations and configuration merging.
Integrating OmegaConf with swanlab is very simple; just pass the `omegaconf` object to `swanlab.config` to record it as hyperparameters:

```python
from omegaconf import OmegaConf
import swanlab

cfg = OmegaConf.load("config.yaml")
swanlab.init(config=cfg,)
```

If unexpected results occur when passing `cfg`, you can first convert `omegaconf.DictConfig` to the original type:

```python
from omegaconf import OmegaConf
import swanlab

cfg = OmegaConf.load("config.yaml")
swanlab.init(config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

```