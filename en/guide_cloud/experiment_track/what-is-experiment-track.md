# Experiment Tracking

**Experiment Tracking** refers to the process of recording metadata (hyperparameters, configurations) and metrics (loss, accuracy...) for each experiment during the development of machine learning models, and organizing and presenting this information. It can be understood as recording various key information of experiments when conducting machine learning experiments.

The purpose of experiment tracking is to help researchers manage and analyze experimental results more effectively, in order to better understand changes in model performance and optimize the model development process. Its functions include:

1. **Details and Reproducibility**: Recording experimental details and facilitating the repetition of experiments.
2. **Comparison**: Comparing results of different experiments to analyze which changes led to performance improvements.
3. **Team Collaboration**: Facilitating the sharing and comparison of experimental results among team members, thereby improving collaboration efficiency.

![](/assets/swanlab-overview.png)

**SwanLab** helps you track machine learning experiments with just a few lines of code and view and compare results in an interactive dashboard. The figure above shows an example dashboard where you can view and compare metrics from multiple experiments and analyze the key elements that lead to performance changes.

## How Does SwanLab Perform Experiment Tracking?

With SwanLab, you can track machine learning experiments with a few lines of code. The tracking process is as follows:

1. Create a SwanLab experiment.
2. Store hyperparameter dictionaries (such as learning rate or model type) in your configuration (swanlab.config).
3. Record metrics (swanlab.log) over time in the training loop, such as accuracy and loss.

The following pseudocode demonstrates a common **SwanLab experiment tracking workflow**:

```python
# 1. Create a SwanLab experiment
swanlab.init(project="my-project-name")

# 2. Store model inputs or hyperparameters
swanlab.config.learning_rate = 0.01

# Write your model training code here
...

# 3. Record metrics over time to visualize performance
swanlab.log({"loss": loss})
```

## How to Get Started?

Explore the following resources to learn about SwanLab experiment tracking:

- Read the [Quick Start](/en/guide_cloud/general/quick-start)
- Explore this chapter to learn how to:
  - [Create an Experiment](/en/guide_cloud/experiment_track/create-experiment)
  - [Configure an Experiment](/en/guide_cloud/experiment_track/set-experiment-config.md)
  - [Log Metrics](/en/guide_cloud/experiment_track/log-experiment-metric.md)
  - [View Experiment Results](/en/guide_cloud/experiment_track/view-result.md)
  - Explore the SwanLab Python library in the [API Documentation](/en/api/api-index).
