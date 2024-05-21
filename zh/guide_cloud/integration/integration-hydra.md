
# Hydra

[hydra](https://hydra.cc/)是一个由Facebook AI Research创建的开源框架，旨在简化Python应用程序中配置的创建、管理和使用过程。Hydra通过使配置内容动态化和可组合，大大简化了处理多个配置集合的复杂性，特别是对于那些具有大量参数和需要在多种环境下运行的应用程序而言。

![hydra-image](/assets/hydra-image.jpg)

你可以继续使用 Hydra 进行配置管理，同时使用SwanLab的强大功能。

## 跟踪指标

和常规一样，用`swanlab.init`和`swanlab.log`跟踪你的指标。  
假设你的hydra配置文件为`configs/defaults.yaml`，则添加几行：

```yaml
swanlab:
  project: "my-project"
```


在训练脚本中，将配置文件中的`project`传入：

```python
import swanlab
import hydra

@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
    run = swanlab.init(project=cfg.swanlab.project)
    ...
    swanlab.log({"loss": loss})
```

## 跟踪超参数
Hydra使用[omegaconf](https://omegaconf.readthedocs.io/en/2.1_branch/)作为与配置字典交互的默认方式。

可以直接将OmegaConf的字典传递给`swanlab.config`：

```python
@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
    run = swanlab.init(project=cfg.swanlab.project,
                       config=cfg,
    )
    ...
    swanlab.log({"loss": loss})
    model = Model(**swanlab.config.model.configs)
```

如果传递`cfg`时出现意外的结果，那么可以先转换`omegaconf.DictConfig`为原始类型：

```python
@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
    run = swanlab.init(project=cfg.swanlab.project,
                       config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    ...
    swanlab.log({"loss": loss})
    model = Model(**swanlab.config.model.configs)
```