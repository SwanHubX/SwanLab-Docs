---
layout: home

title: SwanLab官方文档
titleTemplate: 先进的AI团队协作与模型创新引擎

hero:
  name: SwanLab
  text: 先进的机器学习<br>团队协作创新引擎
  tagline: 一站式实验跟踪与训练可视化工具
  actions:
    - theme: brand
      text: 访问文档
      link: /guide_cloud/general/what-is-swanlab.md
    - theme: brand
      text: 官网
      link: https://dev101.swanlab.cn
    # - theme: alt
    #   text: 快速开始
    #   link: /zh/guide_cloud/general/quick-start.md
    - theme: alt
      text: 文档GitHub
      link: https://github.com/SwanHubX/SwanLab-Docs
  image:
    src: /page.png
    alt: VitePress

features:
  - icon: 🚢
    title: 快速开始
    details: 安装 SwanLab 并在几分钟内开始跟踪你的人工智能实验。
    link: /guide_cloud/general/quick-start.md
  
  - icon: 📚
    title: 案例
    details: SwanLab官方案例合集，更好地理解SwanLab在人工智能pipeline中扮演的角色和作用。
    link: /examples/mnist
  
  - icon: 🤗
    title: 集成
    details: 与HuggingFace Transformers、PyTorch Lightning、Hydra等主流框架的集成文档。
    link: /guide_cloud/integration/integration-pytorch-lightning.md

  - icon: ⚡️
    title: API文档
    details: Python库和CLI的完整API文档
    link: /api/api-index

---

<style>
:root {
  --vp-home-hero-name-color: transparent !important;
  --vp-home-hero-name-background: -webkit-linear-gradient(120deg, #637de8 50%, #63ca8c) !important;

  --vp-home-hero-image-background-image: linear-gradient(-45deg, #8d9956 50%, #47caff 50%) !important;
  --vp-home-hero-image-filter: blur(44px) !important;
}

@media (min-width: 640px) {
  :root {
    --vp-home-hero-image-filter: blur(56px);
  }
}

@media (min-width: 960px) {
  :root {
    --vp-home-hero-image-filter: blur(68px);
  }
}
</style>