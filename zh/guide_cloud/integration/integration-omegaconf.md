# Omegaconf

OmegaConf 是一个用于处理配置的 Python 库，尤其适用于需要灵活配置和配置合并的场景。
OmegaConf 与swanlab的集成非常简单，直接将`omegaconf`对象传递给`swanlab.config`，即可记录为超参数：

```python
from omegaconf import OmegaConf
import swanlab

cfg = OmegaConf.load("config.yaml")
swanlab.init(config=cfg,)
```

如果传递`cfg`时出现意外的结果，那么可以先转换`omegaconf.DictConfig`为原始类型：

```python
from omegaconf import OmegaConf
import swanlab

cfg = OmegaConf.load("config.yaml")
swanlab.init(config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

```