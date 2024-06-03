# 欢迎使用SwanLab

[官网](https://dev101.swanlab.cn) · [Github](https://github.com/swanhubx/swanlab) · [贡献指南](https://github.com/SwanHubX/SwanLab/blob/main/CONTRIBUTING.md) · [SwanHub开源社区](https://swanhub.co)

<!-- ![](/assets/swanlab-show.png) -->

![alt text](/assets/product-swanlab-1.png)

SwanLab是一款开源、轻量级的AI实验跟踪工具，提供了一个**跟踪、比较、和协作**实验的平台，旨在加速AI研发团队100倍的研发效率。

其提供了友好的API和漂亮的界面，结合了超参数跟踪、指标记录、在线协作、实验链接分享、实时消息通知等功能，让您可以快速跟踪ML实验、可视化过程、分享给同伴。

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

![alt text](/assets/how-to-do-swanlab.png)

**1. 📊实验指标与超参数跟踪**: 极简的代码嵌入您的机器学习pipeline，跟踪记录训练关键指标
  - 自由的超参数与实验配置记录
  - **支持的元数据类型**：标量指标、图像、音频、文本、...
  - **支持的图表类型**：折线图、媒体图（图像、音频、文本）、...
  - **自动记录**：控制台logging、GPU硬件、Git信息、Python解释器、Python库列表、代码目录

**2. ⚡️全面的框架集成**: [PyTorch](/zh/guide_cloud/integration/integration-pytorch.md)、[PyTorch Lightning](/zh/guide_cloud/integration/integration-pytorch-lightning.md)、[🤗HuggingFace Transformers](/zh/guide_cloud/integration/integration-huggingface-transformers.md)、[MMEngine](/zh/guide_cloud/integration/integration-mmengine.md)、[Ultralytics](/zh/guide_cloud/integration/integration-ultralytics.md)等主流框架

**3. 📦组织实验**: 集中式仪表板，快速管理多个项目与实验，通过整体视图速览训练全局

**4. 🆚比较结果**: 通过在线表格与对比图表比较不同实验的超参数和结果，挖掘迭代灵感

**5. 👥在线协作**: 您可以与团队进行协作式训练，支持将实验实时同步在一个项目下，您可以在线查看团队的训练记录，基于结果发表看法与建议

**6. ✉️分享结果**: 复制和发送持久的URL来共享每个实验，方便地发送给伙伴，或嵌入到在线笔记中

**7. 💻支持自托管**: 支持不联网使用，自托管的社区版同样可以查看仪表盘与管理实验


## 从哪里开始

- [快速开始](/zh/guide_cloud/general/quick-start.md): SwanLab入门教程，五分钟玩转实验跟踪！
- [API文档](/zh/api/api-index.md): 完整的API文档
- [在线支持](/zh/guide_cloud/community/online-support.md): 加入社区、反馈问题和联系我们
- [自托管](/zh/guide_cloud/self_host/offline-board.md): 自托管（离线版本）使用方式教程
- [案例](/zh/examples/mnist.md): 查看SwanLab与各个深度学习任务的案例

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
    - 产品: zeyi.lin@swanhub.co
    - 合作: shaohon_chen@115lab.club

- **[微信交流群](https://geektechstudio.feishu.cn/wiki/NIZ9wp5LRiSqQykizbGcVzUKnic?fromScene=spaceOverview)**：交流使用SwanLab的问题、分享最新的AI技术。

- **飞书群**: 我们的日常工作交流在飞书上，飞书群的回复会更及时。用飞书App扫描下方二维码即可：

<div align="center">
<img src="/assets/feishu-QR-Code.png" width=300>
</div>

- **微信公众号**:

<div align="center">
<img src="/assets/wechat.jpg" width=300>
</div>