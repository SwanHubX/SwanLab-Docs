# ⚡️Changelog

::: warning Update Guide
Upgrade to latest version: `pip install -U swanlab`  
Github: https://github.com/SwanHubX/SwanLab
:::



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
- Added integration with PaddleYolo, [Documentation](/zh/guide_cloud/integration/integration-paddleyolo.md)

**Bug Fixes**
- Fixed sqlite parallel read/write errors during multiple parallel experiment submissions, [#715](https://github.com/SwanHubX/SwanLab/issues/715)
- Fixed compatibility issues with CPU brand recording

## v0.3.21 - 2024.9.26

**🚀New Features**
- [Organization creation](/zh/guide_cloud/general/organization.md) is now fully open, with a limit of 15 people per organization
- Experiment names now support "duplicates" with a new experiment naming system

## v0.3.19 - 2024.9.2

**🚀New Features**
- (Beta) Added cloud storage functionality for task-based training `swanlab task`, [Documentation](/zh/api/cli-swanlab-remote-gpu.html)

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
- Added integration with `torchtune`, [Documentation](/zh/guide_cloud/integration/integration-pytorch-torchtune)

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
- Added integration with HuggingFace Accelerate, [Documentation](/zh/guide_cloud/integration/integration-huggingface-accelerate.md)

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
- Integration with Stable Baseline3, [Guide](/zh/guide_cloud/integration/integration-sb3.md)

## v0.3.3 - 2024.5.22

**👥Integration**
- Integration with Weights & Biases, supporting wandb project conversion to `SwanLab` projects, [Guide](/zh/guide_cloud/integration/integration-wandb.md)
- Integration with Ultralytics, [Guide](/zh/guide_cloud/integration/integration-ultralytics.md)
- Integration with fastai, [Guide](/zh/guide_cloud/integration/integration-fastai.md)

## v0.3.2 - 2024.5.17

**👥Integration**
- Integration with Tensorboard, supporting conversion of `Tensorboard` log files to `SwanLab` experiments, [Guide](/zh/guide_cloud/integration/integration-tensorboard.md)

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

## v0.3.0 Cloud Version Launch! - 2024.5.1

**🚀New Features**
- Released [Cloud Version](https://dev101.swanlab.cn)
- `swanlab.init` supports initialization with config files
- "Environment" now records Apple M series chips

**👥Integration**
- Integration with 🤗HuggingFace Transformers, [Guide](/zh/guide_cloud/integration/integration-huggingface-transformers.md)
- Integration with PyTorch Lightning, [Guide](/zh/guide_cloud/integration/integration-pytorch-lightning.md)
- Integration with Jupyter Notebook
- Integration with Hydra, [Guide](/zh/guide_cloud/integration/integration-hydra.md)

**⚡️Improvements**
- Optimized line chart rendering for large datasets
- Improved performance in Jupyter
- Fixed numerous issues from previous versions

## v0.2.4 - 2024.3.17

**⚡️Improvements**
- Improved image chart display
- `swanlab.Image` now supports PyTorch tensor input

**🔧Bug Fixes**
- Fixed multimedia chart caption parameter issues
- Fixed issues caused by interrupting `swanlab.init` initialization
- Fixed SwanLab initialization errors in Jupyter notebook

## v0.2.3 - 2024.3.12

**🚀New Features**
- Added line chart smoothing with 3 algorithms
- Added chart pinning
- Added chart hiding

**⚡️Improvements**
- Remembers chart group collapse states
- Multimedia charts default to showing last step

## v0.2.2 - 2024.3.4

**⚡️Improvements**
- Multimedia charts support arrow key navigation

**🔧Bug Fixes**
- Fixed various bugs

## v0.2.1 - 2024.3.1

**🚀New Features**
- Text charts: Support for log text type
- Multi-experiment charts support image and audio charts

**⚡️Improvements**
- Improved line chart component performance

## v0.2.0 - 2024.2.8

**New Features**
- Multi-experiment comparison charts: Compare log data from multiple experiments in one chart
- Image charts: Support for logging images (files, numpy arrays, PIL.Image, matplotlib)
- Audio charts: Support for logging audio (files, numpy arrays)
- Auto-view other charts at same position when viewing one chart

**Improvements**
- Added suffix parameter to swanlab.init for custom experiment suffixes
- Changed loggings parameter to logger in swanlab.log, supporting dictionary control of auto-printing
- Changed default experiment name format to: '%b%d-%h-%m-%s' (example:'Feb03_14-45-37')
- Changed Environment logdir to specific experiment log file path
- Added hardware data monitoring class
- Improved many UI details

**Bug Fixes**
- Fixed line chart display errors caused by swanlab.log step parameter
- Fixed logs loading failures in some cases

## v0.1.6 - 2024.1.25

**New Features**
- Added APIloggings to swanlab.init and swanlab.log, auto-prints metrics to terminal when enabled
- New Config/Summary table component with parameter search

**Improvements**
- Optimized web fonts
- Optimized web header
- Optimized handling of NaN in swanlab.log

**Fixes**
- Fixed swanlab watch errors in Python 3.8
- Fixed GridView and Summary component crashes with incompatible data types in swanlab.log

## v0.1.5 - 2024.1.23

**New Features**
- 🚨Replaced basic config info read/write with SQLite database and Peewee library (#114)
  - Major improvement for future development but incompatible with old versions (swanlab<=v0.1.4). Use [conversion script](https://github.com/SwanHubX/SwanLab/blob/main/script/transfer_logfile_0.1.4.py) for old log files
- Experiment list supports quick CSV export
- Experiment list supports "Summary Only" view
- Experiment list supports search
- Environment items have "Quick Copy" interaction
- Auto environment recording adds logdir, Run path
- New API swanlab.config

**Improvements**
- Improved various UI elements
- Added Y-axis lines to line charts
- Improved error messages for non-existent log files in swanlab watch

**Fixes**
- Fixed errors with hydra library parameter input
- Fixed errors with spaces in swanlab.log dictionary keys
- Fixed errors when running without git initialization

## v0.1.4 - 2024.1.13

**New Features**
- New UI & UX
- Responsive design: Optimized for phones, tablets, and various resolutions
- Additional auto environment recording:
  - Command
  - Git Branch
  - Git Commit
  - Memory
  - Requirements - Auto records current pip environment
- Logs and Requirements support search, copy, and download

**New APIs**
- swanlab.init adds logdir API: Set log file save location
- swanlab watch adds --logdir API: Specify log file location to read
- swanlab.init->config supports Argparse-style calls

**Improvements**
- Optimized Charts x-axis display logic
- Optimized Charts auto-refresh logic for better performance
- Improved Charts and Summary recording of very small/large numbers
- Added unit tests

**Bug Fixes**
- Fixed Requirements recording issues
- Fixed Charts display issues with rounded floating points

## v0.1.3 - 2024.1.7

**New Features**
- Projects and experiments can be deleted via web interface
- Projects and experiments support name and description editing via web
- Added experiment time suffix
- Default project name is training script folder name

**Improvements**
- Enhanced Table component interaction and performance
- Improved pypi display information
- Added message popup component and placeholders
- Improved message popup component styles
- Added git branch name and latest commit hash functions
- Renamed OverView to Project Dashboard

**Bug Fixes**
- Fixed terminal log format issues
- Fixed naming conflicts
- Fixed experiment board port occupation errors
- Improved error page navigation
- Fixed warning issues when running experiment board

## v0.1.2 - 2024.1.1

- Increased Charts data update frequency, added request status mapping
- Fixed project and experiment OverView status not updating after training
- Fixed errors with slashes in swanlab.log data parameter keys
- Fixed display issues with non-standard value types in swanlab.log data
- Optimized Chart value ranges based on recorded data max/min
- Improved experiment table component
- Added command and requirements recording functions
- Added git hooks for code formatting and commit testing
- Fixed duplicate column issues in experiment table with slashed keys

## v0.1.1 - 2023.12.26

**Feature Updates**
- Added Experiment Metrics Comparison Table to compare configurations and results
- Added more experiment parameter recording (Python interpreter directory, system hardware, Git repository path, Swanlab version)

**Improvements**
- New table component for better interaction
- Adjusted time display format

**API Updates**
- Added step parameter to swanlab.log

## v0.1.0 - 2023.12.22

**Feature Updates**
- Added Log functionality to experiment board, auto-records terminal output (including errors)

**Improvements**
- Changed default language to English
- Changed sidebar text from "Community Version" to "Version Number"
- Optimized chart data display format

## v0.0.2 - 2023.12.16

**Feature Updates**
- Added experiment status: Stop
- Added Configuration and Summary information display
- Added automatic terminal log collection, stored in swanlog/experiment_folder/console/yy-mm-dd.log
- Added error pages to experiment board
- Added new API for swanlab watch: host, details in Experiment Board

**Major Updates**
- Adjusted log file structure (incompatible with previous versions). From v0.0.2, default log folder is swanlog

**Improvements**
- Improved chart styles: bold titles, chart zoom support, x-axis display
- Improved terminal output during training
- Improved chart color selection mechanism
- Improved default chart group names and styles
- Improved default project names
- Improved experiment sorting: newest first by default
