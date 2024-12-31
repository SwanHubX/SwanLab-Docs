# LLaMA Factory

![](/zh/guide_cloud/integration/llama_factory/0.png)

We are thrilled to announce the partnership between **SwanLab** and **LLaMA Factory**, dedicated to providing Chinese trainers with a high-quality and efficient large model training experience.

Now, before starting training with the new version of LLaMA Factory, you can check the "Use SwanLab" option in the "SwanLab configurations" card on the WebUI. This allows you to track, record, and visualize this large model fine-tuning session through SwanLab's powerful training dashboard.

![](/zh/guide_cloud/integration/llama_factory/1.png)

LLaMA Factory is an open-source toolkit for fine-tuning large language models (LLMs). It provides a unified and efficient framework, supporting fine-tuning for over 100 LLMs (including Qwen, LLaMA, ChatGLM, Mistral, etc.), covering various training methods, datasets, and advanced algorithms.

Fine-tuning large language models is a task with a steep learning curve. LLaMA Factory significantly lowers the barrier to entry by offering a user-friendly Web UI and command-line interface, combined with its unified and efficient framework, making it easier to go from fine-tuning to testing and evaluation.

To provide users with a better experience in monitoring and logging the fine-tuning process of large models, we have collaborated with the LLaMA Factory team on two initiatives: enhancing LLaMA Factory's experiment monitoring capabilities with SwanLab, and recording LLaMA Factory-specific hyperparameters in SwanLab.

> LLaMA Factory: https://github.com/hiyouga/LLaMA-Factory  
> SwanLab: https://swanlab.cn  
> SwanLab Open Source Repository: https://github.com/SwanHubX/SwanLab  
> Experiment Process: https://swanlab.cn/@ZeyiLin/llamafactory/runs/y79f9ri9jr1mkoh24a7g8/chart

<br>

## Use Case

We will demonstrate the process of fine-tuning Qwen2.5 using LLaMA Factory + SwanLab visualization.

### 1. Environment Setup

First, ensure you have Python 3.8 or above and Git installed, then clone the repository:

```shellscript
git clone https://github.com/hiyouga/LLaMA-Factory
```

Install the necessary dependencies:

```shellscript
cd LLaMA-Factory
pip install -e ".[torch,metrics,swanlab]"
```

> If you are an Ascend NPU user, refer to: https://github.com/hiyouga/LLaMA-Factory/blob/main/README\_zh.md#%E5%AE%89%E8%A3%85-llama-factory for the installation guide.

### 2. Start Training with LLaMA Board

LLaMA Board is a Gradio-based visual fine-tuning interface. You can start LLaMA Board with the following command:

```shellscript
llamafactory-cli webui
```

Tip: By default, LLaMA Factory downloads models/datasets from HuggingFace. If your network environment is not favorable for HuggingFace downloads, you can switch the download source to ModelScope or OpenMind before starting LLaMA Board:

```shellscript
# Switch to ModelScope
export USE_MODELSCOPE_HUB=1 # Windows use `set USE_MODELSCOPE_HUB=1`

# Switch to OpenMind
export USE_OPENMIND_HUB=1 # Windows use `set USE_OPENMIND_HUB=1`
```

After executing `llamafactory-cli webui`, you will see the following UI interface in your browser. For this case, select Qwen2-1.5B-instruct as the model and alpaca\_zh\_demo as the dataset:

![](/zh/guide_cloud/integration/llama_factory/2.png)

At the bottom of the page, you will find a "SwanLab Configurations" card. Expand it to configure SwanLab's project name, experiment name, workspace, API key, and mode.

> If you are using SwanLab for the first time, you need to register an account on swanlab.cn to get your exclusive API key.

Check the **"Use SwanLab"** option:

![](/zh/guide_cloud/integration/llama_factory/3.png)

Now, click the **"Start" button** to begin fine-tuning:

![](/zh/guide_cloud/integration/llama_factory/4.png)

After loading the model and dataset, and officially starting the fine-tuning, you can find the SwanLab section in the command-line interface:

![](/zh/guide_cloud/integration/llama_factory/5.png)

Click the experiment link indicated by the arrow to open the SwanLab experiment tracking dashboard in your **browser**:

![](/zh/guide_cloud/integration/llama_factory/6.png)

In the "Cards" section under the "Configuration" table, the first entry will be LLamaFactory, indicating the framework used for this training.

![](/zh/guide_cloud/integration/llama_factory/7.png)

### 3. Start Training via Command Line

LLaMA Factory also supports fine-tuning via YAML configuration files in the command line.

Edit the **examples/train\_lora/qwen2vl\_lora\_sft.yaml** file in the LLaMA Factory project directory, and add the following at the end of the file:

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

After loading the model and dataset, and officially starting the fine-tuning, similar to LLaMA Board, you can find the SwanLab section in the command-line interface and access the SwanLab experiment dashboard via the experiment link.

![](/zh/guide_cloud/integration/llama_factory/8.png)

![](/zh/guide_cloud/integration/llama_factory/9.png)

***

We salute the LLaMA Factory team for providing such an excellent model training tool to the open-source community. As we continue our collaboration, stay tuned for SwanLab to offer more in-depth and powerful experiment tracking features for large model trainers.