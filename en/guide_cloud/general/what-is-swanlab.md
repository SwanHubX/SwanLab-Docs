# Welcome to SwanLab

[SwanLab.cn](https://swanlab.cn) · [Integration](/guide_cloud/integration/integration-huggingface-transformers.html) · [Github](https://github.com/swanhubx/swanlab) · [VSCode Extension](https://marketplace.visualstudio.com/items?itemName=SwanLab.swanlab&ssr=false#overview) · [Quick Start](/en/guide_cloud/general/quick-start.md) · [Sync Wandb](/en/guide_cloud/integration/integration-wandb.md#_1-synchronized-tracking)


<!-- ![](/assets/swanlab-show.png) -->

![alt text](/assets/product-swanlab-1.png)

::: warning 👋 We are developing a self-hosted version
Expected to be released in March 2025, supporting Docker deployment, with the same features as the cloud version
:::


SwanLab is an open-source, lightweight AI experiment tracking tool that provides a platform for tracking, comparing, and collaborating on experiments, designed to accelerate the R&D efficiency of AI R&D teams by 100 times.

It provides a user-friendly API and a beautiful interface, combined with hyperparameter tracking, metric logging, online collaboration, experiment link sharing, real-time message notification and other functions, allowing you to quickly track ML experiments, visualize the process, and share with peers.

With SwanLab, researchers can accumulate their training experience, seamlessly communicate and collaborate with collaborators, and machine learning engineers can develop production-ready models faster.


## Why SwanLab?

Unlike software engineering, artificial intelligence is an experimental discipline. Generating inspiration, rapid experimentation, and validating ideas are the main themes of AI research. Recording the experimental process and inspiration, just like chemists recording experimental manuscripts, is the core of every AI researcher and research organization to form accumulation and increase acceleration.

Previous experimental recording methods involved staring at the output printed by the terminal in front of the computer, copying and pasting log files (or TFEvent files). Rough logs created obstacles to the emergence of inspiration, and offline log files made it difficult for researchers to form a synergy.

In contrast, SwanLab provides a cloud-based AI experiment tracking solution that focuses on the training process, providing functions such as training visualization, experiment tracking, hyperparameter recording, log recording, and multi-person collaboration. Researchers can easily find iterative inspiration through intuitive visual charts and break down communication barriers through the sharing of online links and organization-based multi-person collaborative training.

> Previous AI research and open-source efforts focused more on results, while we emphasize the process. <br>
> Many community users leverage SwanLab to achieve process transparency.<br>
> Lin Zeyi, SwanLab Co-founder

More importantly, SwanLab is open source, built by a group of machine learning engineers who love open source and the community. We provide a fully self-hosted version that can ensure your data security and privacy.

We hope that the above information and this guide can help you understand this product, and we believe that SwanLab can help you.


## What can SwanLab do?

**1. 📊Experiment metrics and hyperparameter tracking**: Minimal code embedded in your machine learning pipeline to track and record key training metrics
 - Free hyperparameter and experiment configuration recording
- **Supported metadata types**: scalar metrics, images, audio, text, etc.
- **Supported chart types**: line chart, media chart (image, audio, text), etc.
- **Automatic recording**: console logging, GPU hardware, Git information, Python interpreter, Python library list, code directory
2. **⚡️Comprehensive framework integration**: PyTorch, PyTorch Lightning, 🤗HuggingFace Transformers, MMEngine, Ultralytics and other mainstream frameworks
3. **📦Organize experiments**: Centralized dashboard, quickly manage multiple projects and experiments, and get a quick overview of the overall training through a holistic view
4. **🆚Compare results**: Compare hyperparameters and results of different experiments through online forms and comparison charts to explore iterative inspiration
5. **👥Online collaboration**: You can conduct collaborative training with your team, supporting real-time synchronization of experiments under one project. You can view your team's training records online and express your opinions and suggestions based on the results.
6. **✉️Share results**: Copy and send persistent URLs to share each experiment, easily send it to partners, or embed it in online notes.
7. **💻Support self-hosting**: Support for offline use, the self-hosted community edition can also view dashboards and manage experiments.


## Where to start

- [Quick Start](/en/guide_cloud/general/quick-start.md): SwanLab introductory tutorial, master experiment tracking in five minutes!
- [API documentation](/en/api/api-index.md): Complete API documentation
- [Online support](/en/guide_cloud/community/online-support.md): Join the community, provide feedback and contact us
- [Self-hosting](/en/guide_cloud/self_host/offline-board.md): Self-hosting (offline version) usage tutorial
- [Cases](/en/examples/mnist.md): View cases of SwanLab with various deep learning tasks

## Comparison with familiar products

### Tensorboard vs SwanLab

- **☁️Team Collaboration**：
  For cross-team machine learning projects, SwanLab streamlines collaboration by making it easy to manage group training projects, share experiment links, and facilitate communication and discussions. In contrast, TensorBoard is primarily designed for individual use, making multi-user collaboration and experiment sharing more challenging.

- **👥Team Collaboration**：
  For cross-team machine learning projects, SwanLab streamlines collaboration by making it easy to manage group training projects, share experiment links, and facilitate communication and discussions. In contrast, TensorBoard is primarily designed for individual use, making multi-user collaboration and experiment sharing more challenging.

- **💻Persistent, centralized dashboard**：
  No matter where you train your model, whether on your local computer, in a lab cluster, or in a public cloud GPU instance, your results will be logged to the same centralized dashboard. Using TensorBoard, on the other hand, requires time to copy and manage TFEvent files from different machines. 

- **💪Enhanced Table Functionality**: SwanLab provides powerful tables that enable you to view, search, and filter results from multiple experiments effortlessly. This allows easy access to thousands of model versions and helps you quickly identify the best performing models for various tasks. In comparison, TensorBoard is less suited for handling large projects.


### W&B vs SwanLab

- Weights and Biases is a closed-source MLOps platform that must be used online
- SwanLab supports not only online use, but also open-source, free, and self-hosted versions


## Training Framework Integration

Use your favorite framework with SwanLab!  
Below is a list of frameworks we have integrated, please submit [Issue](https://github.com/swanhubx/swanlab/issues) to feedback the framework you want to integrate.

**Basic Frameworks**
- [PyTorch](/en/guide_cloud/integration/integration-pytorch.html)
- [MindSpore](/en/guide_cloud/integration/integration-ascend.html)
- [Keras](/en/guide_cloud/integration/integration-keras.html)

**Specialized/Fine-tuned Frameworks**
- [PyTorch Lightning](/en/guide_cloud/integration/integration-pytorch-lightning.html)
- [HuggingFace Transformers](/en/guide_cloud/integration/integration-huggingface-transformers.html)
- [OpenMind](https://modelers.cn/docs/zh/openmind-library/1.0.0/basic_tutorial/finetune/finetune_pt.html#%E8%AE%AD%E7%BB%83%E7%9B%91%E6%8E%A7)
- [LLaMA Factory](/en/guide_cloud/integration/integration-llama-factory.html)
- [Modelscope Swift](/en/guide_cloud/integration/integration-swift.html)
- [Sentence Transformers](/en/guide_cloud/integration/integration-sentence-transformers.html)
- [Torchtune](/en/guide_cloud/integration/integration-pytorch-torchtune.html)
- [XTuner](/en/guide_cloud/integration/integration-xtuner.html)
- [MMEngine](/en/guide_cloud/integration/integration-mmengine.html)
- [FastAI](/en/guide_cloud/integration/integration-fastai.html)
- [LightGBM](/en/guide_cloud/integration/integration-lightgbm.html)
- [XGBoost](/en/guide_cloud/integration/integration-xgboost.html)


**Computer Vision**
- [Ultralytics](/en/guide_cloud/integration/integration-ultralytics.html)
- [MMDetection](/en/guide_cloud/integration/integration-mmdetection.html)
- [MMSegmentation](/en/guide_cloud/integration/integration-mmsegmentation.html)
- [PaddleDetection](/en/guide_cloud/integration/integration-paddledetection.html)
- [PaddleYOLO](/en/guide_cloud/integration/integration-paddleyolo.html)

**Reinforcement Learning**
- [Stable Baseline3](/en/guide_cloud/integration/integration-sb3.html)
- [veRL](/en/guide_cloud/integration/integration-verl.html)
- [HuggingFace trl](/en/guide_cloud/integration/integration-huggingface-trl.html)
- [EasyR1](/en/guide_cloud/integration/integration-easyr1.html)

**Others:**
- [Tensorboard](/en/guide_cloud/integration/integration-tensorboard.html)
- [Weights&Biases](/en/guide_cloud/integration/integration-wandb.html)
- [MLFlow](/en/guide_cloud/integration/integration-mlflow.html)
- [HuggingFace Accelerate](/en/guide_cloud/integration/integration-huggingface-accelerate.html)
- [Hydra](/en/guide_cloud/integration/integration-hydra.html)
- [Omegaconf](/en/guide_cloud/integration/integration-omegaconf.html)
- [OpenAI](/en/guide_cloud/integration/integration-openai.html)
- [ZhipuAI](/en/guide_cloud/integration/integration-zhipuai.html)

[更多集成](/en/guide_cloud/integration/integration-pytorch-lightning.html)

## Online Support

- **[GitHub Issues](https://github.com/SwanHubX/SwanLab/issues)**：Report errors and issues encountered while using SwanLab

- **Email Support**: Feedback regarding issues with SwanLab
  - Product: <contact@swanlab.cn>, <zeyi.lin@swanhub.co>(Product Manager Email)

- **WeChat Group and Lark Group**: See [Online Support](/en/guide_cloud/community/online-support.md)

- **WeChat Official Account**:

<div align="center">
<img src="/assets/wechat_public_account.jpg" width=300>
</div>
