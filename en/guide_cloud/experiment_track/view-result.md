# View Experiment Results

Use the SwanLab dashboard to manage and visualize AI model training results in one place.

## ☁️ Cloud Synchronization, Freeing Productivity

No matter where you train your models—your own computer, a server cluster in the lab, or an instance in the cloud—we can collect and aggregate your training data, allowing you to access your training progress anytime, anywhere.

You also don't need to spend time taking screenshots of terminal outputs or pasting them into Excel, nor do you need to manage Tensorboard files from different computers. SwanLab makes it easy.

## Table View

Compare each training experiment through the table view to see which hyperparameters have changed.  
The table view defaults to sorting data in the order of `[Experiment Name]-[System Data]-[Configuration]-[Metrics]`.

![view-result](/assets/view-result-1.jpg)

## Chart Comparison

The **Chart Comparison View** allows you to integrate charts from each experiment to generate a multi-experiment comparison chart view.  
In the multi-experiment chart, you can clearly compare the changes and performance differences of different experiments under the same metric.

![chart-comparison](/assets/chart-comparison.jpg)

## Logging

From the start to the end of the experiment, SwanLab records terminal outputs from `swanlab.init` to the end of the experiment and logs them in the experiment's "Logs" tab, where you can view, copy, and download them at any time. We also support searching for key information.

![logging](/assets/logging.jpg)

## Environment

After the experiment starts, SwanLab records the training-related environment parameters, including:

- **Basic Data**: Running time, hostname, operating system, Python version, Python interpreter, running directory, command line, Git repository URL, Git branch, Git commit, log file directory, SwanLab version
- **System Hardware**: Number of CPU cores, memory size, number of GPUs, GPU model, GPU memory
- **Python Libraries**: All Python libraries in the running environment

![environment](/assets/environment.jpg)