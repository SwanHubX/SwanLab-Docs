
# Hydra

[hydra](https://hydra.cc/)是一个由Facebook AI Research创建的开源框架，旨在简化Python应用程序中配置的创建、管理和使用过程。Hydra通过使配置内容动态化和可组合，大大简化了处理多个配置集合的复杂性，特别是对于那些具有大量参数和需要在多种环境下运行的应用程序而言。

![hydra-image](/assets/hydra-image.jpg)

你可以继续使用 Hydra 进行配置管理，同时使用SwanLab的强大功能。

## 跟踪指标

和常规一样，用·swanlab.init`和`swanlab.log`跟踪你的指标。在这里，`swanlab.organization`和`swanlab.project`被hydra配置文件定义：

```python
import swanlab
import hydra

@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
    run = swanlab.init(organization=cfg.swanlab.organization, project=cfg.swanlab.project)
    swanlab.log({"loss": loss})
```

## 跟踪超参数
Hydra使用[omegaconf](https://omegaconf.readthedocs.io/en/2.1_branch/)作为与配置字典交互的默认方式。

OmegaConf的字典不是原始字典的子类，因此直接传递Hydra的Config给`swanlab.config`会导致出现意外的结果。在传递之前有必要转换`omegaconf.DictConfig`为原始类型。

```python
@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
    run = swanlab.init(organization=cfg.swanlab.organization, project=cfg.swanlab.project)
    swanlab.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    swanlab.log({"loss": loss})
    model = Model(**swanlab.config.model.configs)
```