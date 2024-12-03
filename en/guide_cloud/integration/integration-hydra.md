# Hydra

[Hydra](https://hydra.cc/) is an open-source framework created by Facebook AI Research to simplify the creation, management, and usage of configurations in Python applications. Hydra significantly simplifies the complexity of handling multiple configuration sets by making the configuration content dynamic and composable, especially for applications with a large number of parameters and that need to run in various environments.

![hydra-image](/assets/hydra-image.jpg)

You can continue to use Hydra for configuration management while leveraging the powerful features of SwanLab.

## Track Metrics

As usual, use `swanlab.init` and `swanlab.log` to track your metrics.  
Assume your Hydra configuration file is `configs/defaults.yaml`, then add a few lines:

```yaml
swanlab:
  project: "my-project"
```

In your training script, pass the `project` from the configuration file:

```python
import swanlab
import hydra

@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
    run = swanlab.init(project=cfg.swanlab.project)
    ...
    swanlab.log({"loss": loss})
```

## Track Hyperparameters
Hydra uses [omegaconf](https://omegaconf.readthedocs.io/en/2.1_branch/) as the default way to interact with configuration dictionaries.

You can directly pass the OmegaConf dictionary to `swanlab.config`:

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

If unexpected results occur when passing `cfg`, you can first convert `omegaconf.DictConfig` to the original type:

```python
@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg):
    run = swanlab.init(project=cfg.swanlab.project,
                       config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    ...
    swanlab.log({"loss": loss})
    model = Model(**swanlab.config.model.configs)
```