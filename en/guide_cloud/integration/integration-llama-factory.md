# LLaMA Factory

[[toc]]

## 0. Preface

![](/zh/guide_cloud/integration/llama_factory/0.png)

We are thrilled to announce the partnership between **SwanLab** and [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory), dedicated to providing Chinese trainers with a high-quality and efficient large model training experience.

Now, when you start training with the new version of LLaMA Factory, you can check the "Use SwanLab" option in the "SwanLab configurations" card in the WebUI to track, record, and visualize this large model fine-tuning through SwanLab's powerful training dashboard.

![](/zh/guide_cloud/integration/llama_factory/1.png)

LLaMA Factory is an open-source toolkit for fine-tuning large language models (LLMs). It provides a unified and efficient framework, supporting fine-tuning for over 100 LLMs (including Qwen, LLaMA, ChatGLM, Mistral, etc.), covering various training methods, datasets, and advanced algorithms.

Fine-tuning large language models is a task with a high entry barrier. LLaMA Factory significantly lowers this barrier by offering a user-friendly Web UI and command-line interface, combined with its unified and efficient framework, making the process from fine-tuning to testing and evaluation much more accessible.

To provide users with a better experience in monitoring and logging the fine-tuning process of large models, we have collaborated with the LLaMA Factory team on two initiatives: enhancing LLaMA Factory's experiment monitoring capabilities using SwanLab, and recording LLaMA Factory-specific hyperparameters in SwanLab.

> LLaMA Factory: https://github.com/hiyouga/LLaMA-Factory  
> SwanLab: https://swanlab.cn  
> SwanLab Open Source Repository: https://github.com/SwanHubX/SwanLab  
> Experiment Process: https://swanlab.cn/@ZeyiLin/llamafactory/runs/y79f9ri9jr1mkoh24a7g8/chart

We will use the case of fine-tuning Qwen2.5 with LLaMA Factory + SwanLab visualization as an example.

## 1. Environment Setup

First, ensure you have a Python 3.8+ environment and Git tools, then clone the repository:

```shellscript
git clone https://github.com/hiyouga/LLaMA-Factory
```

Install the necessary environment:

```shellscript
cd LLaMA-Factory
pip install -e ".[torch,metrics,swanlab]"
```

> If you are an Ascend NPU user, you can visit: [Huawei NPU Adaptation](https://llamafactory.readthedocs.io/zh-cn/latest/advanced/npu.html) for the Ascend NPU version installation tutorial.

## 2. Start Training with LLaMA Board

LLaMA Board is a visual fine-tuning interface based on Gradio. You can start LLaMA Board with the following command:

```shellscript
llamafactory-cli webui
```

Tip: LLaMA Factory's default model/dataset download source is HuggingFace. If your network environment is not friendly to HuggingFace downloads, you can set the download source to ModelScope or OpenMind before starting LLaMA Board:

```shellscript
# Change download source to ModelScope
export USE_MODELSCOPE_HUB=1 # Windows use `set USE_MODELSCOPE_HUB=1`

# Change download source to OpenMind
export USE_OPENMIND_HUB=1 # Windows use `set USE_OPENMIND_HUB=1`
```

After executing `llamafactory-cli webui`, you will see the following UI interface in your browser. This case selects Qwen2-1.5B-instruct as the model and alpaca_zh_demo as the dataset:

![](/zh/guide_cloud/integration/llama_factory/2.png)

At the bottom of the page, you will see a "SwanLab Configurations" card. Expand it to configure SwanLab's project name, experiment name, workspace, API key, and mode.

> If you are using SwanLab for the first time, you need to register an account at swanlab.cn to get your exclusive API key.

Check **"Use SwanLab":**

![](/zh/guide_cloud/integration/llama_factory/3.png)

Now, click the **"Start" button** to begin fine-tuning:

![](/zh/guide_cloud/integration/llama_factory/4.png)

After loading the model and dataset and officially starting the fine-tuning, you can find the SwanLab section in the command-line interface:

![](/zh/guide_cloud/integration/llama_factory/5.png)

Click the experiment link indicated by the arrow to open the SwanLab experiment tracking dashboard in your **browser**:

![](/zh/guide_cloud/integration/llama_factory/6.png)

In the "Cards" section under the "Configuration" table, the first entry will be LLamaFactory, indicating the framework used for this training.

![](/zh/guide_cloud/integration/llama_factory/7.png)

## 3. Start Training via Command Line

LLaMA Factory also supports fine-tuning via YAML configuration files in the command line.

Edit the **examples/train_lora/qwen2vl_lora_sft.yaml** file in the LLaMA Factory project directory and add the following at the end:

```yaml
...

### swanlab
use_swanlab: true
swanlab_project: llamafactory
swanlab_run_name: Qwen2-VL-7B-Instruct
```

Then run:

```shellscript
llamafactory-cli train examples/train_lora/qwen2vl_lora_sft.yaml
```

After loading the model and dataset and officially starting the fine-tuning, as with LLaMA Board, you can find the SwanLab section in the command-line interface and access the SwanLab experiment dashboard via the experiment link.

![](/zh/guide_cloud/integration/llama_factory/8.png)

![](/zh/guide_cloud/integration/llama_factory/9.png)

***

We salute the LLaMA Factory team for providing such an excellent model training tool to the open-source community. As we continue to collaborate, stay tuned for more in-depth and powerful experiment tracking features from SwanLab for large model trainers.

## 4. Appendix: Supported Parameters

```yaml
# swanlab
use_swanlab: true
swanlab_project: your_project_name
swanlab_run_name: your_experiment_name
swanlab_workspace: your_workspace
swanlab_mode: your_mode
swanlab_api_key: your_api_key
```

> For more details, see the `SwanLabArguments` class in [LLaMA Factory - Github](https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md#%E5%AE%89%E8%A3%85-llama-factory).