# 欢迎使用SwanLab 

[官网](https://swanlab.cn) · [Github](https://github.com/swanhubx/swanlab) · [VSCode插件](https://marketplace.visualstudio.com/items?itemName=SwanLab.swanlab&ssr=false#overview) · [快速开始](/guide_cloud/general/quick-start.md) · [同步Wandb](/guide_cloud/integration/integration-wandb.md#_1-同步跟踪)

<!-- ![](/assets/swanlab-show.png) -->

<!-- ![alt text](/assets/product-swanlab-1.png) -->

::: warning 👋 我们正在开发私有化部署版
预计25年3月份与大家见面，支持Docker部署，功能与云端版一致
:::

![alt text](/assets/product-swanlab-1.png)



SwanLab 是一款**开源、轻量**的 AI 模型训练跟踪与可视化工具，提供了一个**跟踪、记录、比较、和协作实验**的平台。

SwanLab 面向人工智能研究者，设计了友好的Python API 和漂亮的UI界面，并提供**训练可视化、自动日志记录、超参数记录、实验对比、多人协同等功能**。在SwanLab上，研究者能基于直观的可视化图表发现训练问题，对比多个实验找到研究灵感，并通过**在线网页**的分享与基于组织的**多人协同训练**，打破团队沟通的壁垒，提高组织训练效率。

借助SwanLab，科研人员可以沉淀自己的每一次训练经验，与合作者无缝地交流和协作，机器学习工程师可以更快地开发可用于生产的模型。



## 为什么使用SwanLab？

与软件工程不同，人工智能是一个**实验性学科**，产生灵感、快速试验、验证想法 是AI研究的主旋律。而记录下实验过程和灵感，就像化学家记录实验手稿一样，是每一个AI研究者、研究组织**形成积累、提升加速度**的核心。

先前的实验记录方法，是在计算机前盯着终端打印的输出，复制粘贴日志文件（或TFEvent文件），**粗糙的日志对灵感的涌现造成了障碍，离线的日志文件让研究者之间难以形成合力**。

与之相比，SwanLab提供了一套云端AI实验跟踪方案，面向训练过程，提供了训练可视化、实验跟踪、超参数记录、日志记录、多人协同等功能，研究者能轻松**通过直观的可视化图表找到迭代灵感，并且通过在线链接的分享与基于组织的多人协同训练**，打破团队沟通的壁垒。

> 以往的AI研究的分享和开源更关注结果，而我们更关注过程。<br>
> 社区用户对SwanLab的产品评价可以归结为**简洁易用、提升效率与迭代迅速**<br>
> ——泽毅，SwanLab 联合创始人

更重要的是，SwanLab是开源的，由一帮热爱开源的机器学习工程师与社区共同构建，我们提供了完全自托管的版本，可以保证你的数据安全与隐私性。

希望以上信息和这份指南可以帮助你了解这款产品，我们相信 SwanLab 能够帮助到你。

## SwanLab能做什么？

**1. 📊 实验指标与超参数跟踪**: 极简的代码嵌入您的机器学习 pipeline，跟踪记录训练关键指标

- 支持**云端**使用（类似Weights & Biases），随时随地查看训练进展。[手机看实验的方法](https://docs.swanlab.cn/guide_cloud/general/app.html)
- 支持**超参数记录**与表格展示
- **支持的元数据类型**：标量指标、图像、音频、文本、...
- **支持的图表类型**：折线图、媒体图（图像、音频、文本）、...
- **后台自动记录**：日志logging、硬件环境、Git 仓库、Python 环境、Python 库列表、项目运行目录

**2. ⚡️ 全面的框架集成**: PyTorch、🤗HuggingFace Transformers、PyTorch Lightning、🦙LLaMA Factory、MMDetection、Ultralytics、PaddleDetetion、LightGBM、XGBoost、Keras、Tensorboard、Weights&Biases、OpenAI、Swift、XTuner、Stable Baseline3、Hydra 在内的 **30+** 框架

![](/assets/integrations.png)

**3. 💻 硬件监控**: 支持实时记录与监控CPU、NPU（昇腾Ascend）、GPU（英伟达Nvidia）、内存的系统级硬件指标

**4. 📦 实验管理**: 通过专为训练场景设计的集中式仪表板，通过整体视图速览全局，快速管理多个项目与实验

**4. 🆚 比较结果**: 通过在线表格与对比图表比较不同实验的超参数和结果，挖掘迭代灵感

**5. 👥 在线协作**: 您可以与团队进行协作式训练，支持将实验实时同步在一个项目下，您可以在线查看团队的训练记录，基于结果发表看法与建议

**6. ✉️ 分享结果**: 复制和发送持久的 URL 来共享每个实验，方便地发送给伙伴，或嵌入到在线笔记中

**7. 💻 支持自托管**: 支持离线环境使用，自托管的社区版同样可以查看仪表盘与管理实验


## 从哪里开始

- [快速开始](/guide_cloud/general/quick-start.md): SwanLab入门教程，五分钟玩转实验跟踪！
- [API文档](/api/api-index.md): 完整的API文档
- [在线支持](/guide_cloud/community/online-support.md): 加入社区、反馈问题和联系我们
- [自托管](/guide_cloud/self_host/offline-board.md): 自托管（离线版本）使用方式教程
- [案例](/examples/mnist.md): 查看SwanLab与各个深度学习任务的案例

## 与熟悉产品的对比

### Tensorboard vs SwanLab

- **☁️支持在线使用**：
  通过SwanLab可以方便地将训练实验在云端在线同步与保存，便于远程查看训练进展、管理历史项目、分享实验链接、发送实时消息通知、多端看实验等。而Tensorboard是一个离线的实验跟踪工具。

- **👥多人协作**：
  在进行多人、跨团队的机器学习协作时，通过SwanLab可以轻松管理多人的训练项目、分享实验链接、跨空间交流讨论。而Tensorboard主要为个人设计，难以进行多人协作和分享实验。

- **💻持久、集中的仪表板**：
  无论你在何处训练模型，无论是在本地计算机上、在实验室集群还是在公有云的GPU实例中，你的结果都会记录到同一个集中式仪表板中。而使用TensorBoard需要花费时间从不同的机器复制和管理 TFEvent文件。
  
- **💪更强大的表格**：
  通过SwanLab表格可以查看、搜索、过滤来自不同实验的结果，可以轻松查看数千个模型版本并找到适合不同任务的最佳性能模型。 TensorBoard 不适用于大型项目。  


### W&B vs SwanLab

- Weights and Biases 是一个必须联网使用的闭源MLOps平台

- SwanLab 不仅支持联网使用，也支持开源、免费、自托管的版本

## 在线支持

- **[GitHub Issues](https://github.com/SwanHubX/SwanLab/issues)**：反馈使用SwanLab时遇到的错误和问题

- **电子邮件支持**：反馈关于使用SwanLab的问题
  - 产品: <contact@swanlab.cn>, <zeyi.lin@swanhub.co>(产品经理邮箱)

- **微信群与飞书群**: 见[在线支持](/guide_cloud/community/online-support.md)

- **微信公众号**:

<div align="center">
<img src="/assets/wechat_public_account.jpg" width=300>
</div>
