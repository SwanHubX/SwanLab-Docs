# LLaMA Factory

[[toc]]

## 0. 前言

![](/zh/guide_cloud/integration/llama_factory/0.png)

我们非常高兴地宣布**SwanLab**与**LLaMA Factory**建立合作伙伴关系，致力于为中国训练者提供优质、高效的大模型训练体验。

现在你使用新版本的LLaMA Factory启动训练前，可以在WebUI的「SwanLab configurations」（中文：SwanLab参数设置）卡片中勾选「Use SwanLab」，就可以通过SwanLab强大的训练看板进行这一次大模型微调的跟踪、记录与可视化。

![](/zh/guide_cloud/integration/llama_factory/1.png)

LLaMA Factory 是一个用于微调大语言模型 (LLM) 的开源工具包，它提供了一个统一且高效的框架，支持 100 多个 LLM （包括Qwen、LLaMA、ChatGLM、Mistral等）的微调，涵盖了各种训练方法、数据集和先进算法。

大语言模型的微调是一个上手门槛颇高的工作，LLaMA Factory通过提供用户友好的 Web UI 和命令行界面，结合其统一且高效的框架，大幅降低了大模型从微调到测试评估的上手门槛。

为了提供用户更好的大模型微调过程监控与日志记录体验，我们与LLaMA Factory团队合作开展了两项举措：利用SwanLab增强LLaMA Factory的实验监控能力，以及在SwanLab中记录 LLaMA Factory的专属超参数。


> LLaMA Factory：https://github.com/hiyouga/LLaMA-Factory  
> SwanLab：https://swanlab.cn  
> SwanLab开源仓库：https://github.com/SwanHubX/SwanLab  
> 实验过程：https://swanlab.cn/@ZeyiLin/llamafactory/runs/y79f9ri9jr1mkoh24a7g8/chart

我们将以使用LLaMA Factory + SwanLab可视化微调Qwen2.5为案例。

## 1. 安装环境

首先，你需要确保你拥有Python3.8以上环境与Git工具，然后克隆仓库：

```shellscript
git clone https://github.com/hiyouga/LLaMA-Factory
```

安装相关环境：

```shellscript
cd LLaMA-Factory
pip install -e ".[torch,metrics,swanlab]"
```

> 如果你是昇腾NPU用户，可以访问：https://github.com/hiyouga/LLaMA-Factory/blob/main/README\_zh.md#%E5%AE%89%E8%A3%85-llama-factory 查看昇腾NPU版安装教程。

## 2. 使用LLaMA Board开启训练

LLaMA Board是基于Gradio的可视化微调界面，你可以通过下面的代码启动LLaMA Board：

```shellscript
llamafactory-cli webui
```

提示：LLaMA Factory默认的模型/数据集下载源是HuggingFace，如果你所在的网络环境对与HuggingFace下载并不友好，可以在启动LLaMA Board之前，将下载源设置为魔搭社区或魔乐社区：

```shellscript
# 下载源改为魔搭社区
export USE_MODELSCOPE_HUB=1 # Windows 使用 `set USE_MODELSCOPE_HUB=1`

# 下载源改为魔乐社区
export USE_OPENMIND_HUB=1 # Windows 使用 `set USE_OPENMIND_HUB=1`
```

执行 llamafactory-cli webui 之后，你可以在浏览器看到下面的UI界面。本案例选择Qwen2-1.5B-instruct作为模型，alpaca\_zh\_demo作为数据集：

![](/zh/guide_cloud/integration/llama_factory/2.png)

在页面的下方，你会看到一个「SwanLab参数设置」的卡片，展开后，你就可以配置SwanLab的项目名、实验名、工作区、API 密钥以及模式等参数。

> 如果你是第一次使用SwanLab，还需要在 swanlab.cn 注册一个账号获取专属的API密钥。

我们勾&#x9009;**「使用SwanLab」：**

![](/zh/guide_cloud/integration/llama_factory/3.png)

现在，点&#x51FB;**「开始」按钮**，就可以开启微调：

![](/zh/guide_cloud/integration/llama_factory/4.png)

在完成载入模型、载入数据集，正式开启微调后，我们可以在命令行界面找到SwanLab部分：

![](/zh/guide_cloud/integration/llama_factory/5.png)

点击箭头对应的实验链接，就可以在**浏览器**中打开SwanLab实验跟踪看板：

![](/zh/guide_cloud/integration/llama_factory/6.png)

在「卡片」栏下的「配置」表中，第一个就会是LLamaFactory，标识了这次训练的使用框架。

![](/zh/guide_cloud/integration/llama_factory/7.png)



## 3. 使用命令行开启训练

LLaMA Factory还支持通过yaml配置文件，在命令行中进行微调。

我们编辑LLaMA Factory项目目录下的 **examples/train\_lora/qwen2vl\_lora\_sft.yaml** 文件，在文件尾部增加：

```yaml
...

### swanlab
use_swanlab: true
swanlab_project: llamafactory
swanlab_run_name: Qwen2-VL-7B-Instruct
```

然后运行：

```shellscript
llamafactory-cli train examples/train_lora/qwen2vl_lora_sft.yaml
```

在完成载入模型、载入数据集，正式开启微调后，与LLaMA Board一样，可以在命令行界面找到SwanLab部分，通过实验链接访问SwanLab实验看板。

![](/zh/guide_cloud/integration/llama_factory/8.png)

![](/zh/guide_cloud/integration/llama_factory/9.png)

***

致敬 LLaMA Factory 团队，感谢他们为开源社区提供了这么一个优秀的模型训练工具。随着我们的继续合作，敬请期待SwanLab工具为大模型训练师提供更深入、强大的实验跟踪功能。

## 4.附录：支持的参数

```yaml
# swanlab
use_swanlab: true
swanlab_project: your_project_name
swanlab_run_name: your_experiment_name
swanlab_workspace: your_workspace
swanlab_mode: your_mode
swanlab_api_key: your_api_key
```

> 更多可见：[LLaMA Factory - Github](https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md#%E5%AE%89%E8%A3%85-llama-factory) 中的`SwanLabArguments`类。
