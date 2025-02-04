# ⚡️Changelog

::: warning Update Guide
Upgrade to latest version: `pip install -U swanlab`  
Github: https://github.com/SwanHubX/SwanLab
:::

## v0.4.6 - 2025.2.3

**🚀New Features**
- `swanlab.log` supports nested dictionary input

**Optimizations**
- Optimized distributed training optimization in PyTorch Lightning framework


## v0.4.5 - 2025.1.22

**🚀New Features**
- Added `swanlab.sync_tensorboardX()` and `swanlab.sync_tensorboard_torch()`: Supports synchronizing metrics to SwanLab when using TensorboardX or PyTorch.utils.tensorboard for experiment tracking. [Docs](/en/guide_cloud/integration/integration-tensorboard.md)

**Optimizations**
- Optimized the code compatibility of `sync_wandb()`


## v0.4.3 - 2025.1.17

**🚀 New Features**
- Added `swanlab.sync_wandb()`: Supports synchronizing metrics to SwanLab when using Weights&Biases for experiment tracking. [Docs](/en/guide_cloud/integration/integration-wandb.md)
- Added framework integration: Configuration items will now record the framework being used.

**Optimizations**
- Improved table view interactions, adding row and column dragging, filtering, and sorting interactions.
- Significantly optimized workspace loading performance.
- Significantly optimized log rendering performance.
- Improved the interaction when executing `swanlab.init()` on a non-logged-in computer.
- Fixed several known issues.


## New Year's Day Update

**🚀 New Features**
- Upgraded chart smoothing; the state will remain preserved after webpage refresh
- Updated chart resizing; now you can change the size by dragging the bottom right corner of the chart

**⚙️ Bug Fixes**
- Fixed a bug where the project settings did not display the delete option when there were no experiments

## v0.4.2 - 2024.12.24

**🚀New Features**
- Added password login
- Added project settings page

**Improvements**
- Fixed warning issues when running hardware monitoring on some devices


## v0.4.0 - 2024.12.15

🎉The long-awaited hardware monitoring feature (cloud version) is now available, supporting system-level monitoring of **CPU, NPU, and GPU**:

- **CPU**: Utilization, Thread Count
- **Memory**: Utilization, Process Utilization, Available Memory
- **Nvidia GPU**: Utilization, Memory Allocation, Temperature, Power Consumption
- **Ascend NPU**: Utilization, HBM Allocation, Temperature

More monitoring features are on the way!

by Cunyue

## v0.3.28 - 2024.12.6

> 🍥Announcement: Hardware monitoring feature coming soon!

**🚀New Features**
- Added integration with LightGBM
- Added integration with XGBoost

**Improvements**
- Increased line length limit for log recording
- Improved performance in preparation for version 0.4.0


## v0.3.27 - 2024.11.26

**🚀New Features**
- Added Huawei Ascend NPU GPU detection
- Added integration with Coreshub

## New UI Released!

![alt text](/assets/new-homepage.png)

**🚀What We Improved**
- Launched new website and UI interface focused on user experience
- Released personal/organization homepage
- Added "Dark Mode"
- Fully optimized "Quick Start Guide" with framework integrations and examples
- Improved experiment selection logic in "Chart Comparison View"

## v0.3.25 - 2024.11.11

**🚀New Features**
- 🎉[VSCode Extension](https://marketplace.visualstudio.com/items?itemName=SwanLab.swanlab&ssr=false#overview) is now available
- Added integration with Keras framework
- Added `run.public` method to access experiment project name, experiment name, links etc., [#732](https://github.com/SwanHubX/SwanLab/pull/732)

## v0.3.22 - 2024.10.18

**🚀New Features**
- 🎉Benchmark Community Beta version is now live: https://swanlab.cn/benchmarks
- Added integration with PaddleYolo, [Documentation](/en/guide_cloud/integration/integration-paddleyolo.md)

**Bug Fixes**
- Fixed sqlite parallel read/write errors during multiple parallel experiment submissions, [#715](https://github.com/SwanHubX/SwanLab/issues/715)
- Fixed compatibility issues with CPU brand recording

## v0.3.21 - 2024.9.26

**🚀New Features**
- [Organization creation](/en/guide_cloud/general/organization.md) is now fully open, with a limit of 15 people per organization
- Experiment names now support "duplicates" with a new experiment naming system

## v0.3.19 - 2024.9.2

**🚀New Features**
- (Beta) Added cloud storage functionality for task-based training `swanlab task`, [Documentation](/en/api/cli-swanlab-remote-gpu.html)

**Improvements**
- [Environment] Added CPU brand recording

**Bug Fixes**
- Fixed issues with `swanlab login` in Windows command line caused by misoperations

## v0.3.17 - 2024.8.18

1. Completed code refactoring of cloud chart library and frontend, improved many interactions
2. Fixed parameter display issue in experiment table sidebar for unloaded experiments
3. Fixed network connection errors caused by requests package for some users
4. [Environment] Added NVIDIA driver version recording
5. Local board now supports automatic port renewal for occupied ports

## v0.3.16 - 2024.7.31

**🚀New Features**
- (Beta) Added task-based training `swanlab task` functionality
- Added integration with `torchtune`, [Documentation](/en/guide_cloud/integration/integration-pytorch-torchtune)

**Improvements**
- Added `public` parameter to `swanlab.init` to set new project visibility, defaults to `False`
- Changed default visibility of projects created with `swanlab.init` to private
- Added `dataclass` type support for `swanlab.config`

**Bug Fixes**
- Fixed missing dependency issues when importing swanlab in conda-forge environment

## v0.3.14 - 2024.7.20

**Bug Fixes**
- Fixed environment dependency installation issues
- Fixed various compatibility issues on Windows systems

## v0.3.13 - 2024.6.27

**🚀New Features**
- Added support for changing experiment colors

**⚡️Improvements**
- Optimized issues in Google CoLab and Jupyter Notebook
- Improved error log collection and printing

**Bug Fixes**
- Fixed various issues when running on Windows systems
- Fixed terminal printing issues with frameworks like Hydra
- Fixed issue where save_dir couldn't be None in SwanlabVisBackend for mmengine integration

## v0.3.11 - 2024.6.14

**🚀New Features**
- Added PID and Python Verbose to environment recording
- Support for changing project visibility
- Offline board command changed to `swanlab watch [LOG PATH]`

**⚡️Improvements**
- Optimized Python environment search performance
- Optimized SwanLab library architecture

**Bug Fixes**
- Fixed offline board startup failure issues

## v0.3.10 - 2024.6.10

**Bug Fixes**
- Fixed encoding errors when uploading certain texts
- Fixed environment information not uploading correctly

## v0.3.9 - 2024.6.8

**🚀New Features**
- `swanlab logout`: Support logging out of SwanLab account in terminal

**👥Integration**
- Added integration with HuggingFace Accelerate, [Documentation](/en/guide_cloud/integration/integration-huggingface-accelerate.md)

**⚡️Improvements**
- Improved media file upload stability

**Bug Fixes**
- Fixed nvml library compatibility issues
- Resolved 409 errors when uploading large media files at experiment end
- Fixed OSError issues on some machines

## v0.3.8 - 2024.5.31

**⚡️Improvements**
- Improved integration with ultralytics in ddp scenarios
- Added latest version notification during swanlab.init

**Bug Fixes**
- Fixed thread crash when log value is `inf`
- Fixed image upload failures during long training sessions

## v0.3.6 - 2024.5.28

**Bug Fixes**
- Fixed some logging data upload issues
- Fixed `swanlab login` issues

## v0.3.4 - 2024.5.27

**🚀New Features**
- Added `mode` parameter to `swanlab.init`, supporting new `disabled` mode
- Support for batch experiment deletion

**⚡️Improvements**
- Optimized ultralytics integration code

**👥Integration**
- Integration with Stable Baseline3, [Guide](/en/guide_cloud/integration/integration-sb3.md)

## v0.3.3 - 2024.5.22

**👥Integration**
- Integration with Weights & Biases, supporting wandb project conversion to `SwanLab` projects, [Guide](/en/guide_cloud/integration/integration-wandb.md)
- Integration with Ultralytics, [Guide](/en/guide_cloud/integration/integration-ultralytics.md)
- Integration with fastai, [Guide](/en/guide_cloud/integration/integration-fastai.md)

## v0.3.2 - 2024.5.17

**👥Integration**
- Integration with Tensorboard, supporting conversion of `Tensorboard` log files to `SwanLab` experiments, [Guide](/en/guide_cloud/integration/integration-tensorboard.md)

**🚀New Features**
- Support for downloading line charts as PNG images
- SwanLab experiments can now be embedded in online documents (Feishu/Notion etc. that support webpage embedding)
- Table view supports CSV export
- Table view supports metrics-only view

**⚡️Improvements**
- Optimized value display in line charts and table views

**⚙️Bug Fixes**
- Fixed config table display bug when loading `hydra` config files with `swanlab.config` on Windows
- Resolved SwanLab login issues in Jupyter Notebook

## v0.3.1 - 2024.5.3

**⚡️Improvements**
- Added default `.gitignore` to `swanlog` log folder

**⚙️Bug Fixes**
- Fixed compatibility issues with Omegaconfig and similar types in `swanlab.init` config