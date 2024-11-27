---
layout: home

title: SwanLab Docs
titleTemplate: Advanced AI Team Collaboration and Model Innovation Engine

hero:
  name: SwanLab
  text: Track and Visualize Your AI Experiments
  tagline: One-stop Experiment Tracking and Training Visualization Tool
  actions:
    - theme: brand
      text: Documentation
      link: /en/guide_cloud/general/what-is-swanlab.md
    - theme: brand
      text: Website
      link: https://dev101.swanlab.cn
    # - theme: alt
    #   text: 快速开始
    #   link: /en/guide_cloud/general/quick-start.md
    - theme: alt
      text: Docs GitHub
      link: https://github.com/SwanHubX/SwanLab-Docs
  image:
    src: /page.png
    alt: VitePress

features:
  - icon: 🚢
    title: Quick Start
    details: Install SwanLab and start tracking your AI experiments in minutes.
    link: /en/guide_cloud/general/quick-start.md
  
  # - icon: 📚
  #   title: 案例
  #   details: SwanLab官方案例合集，更好地理解SwanLab在人工智能pipeline中扮演的角色和作用。
  #   link: /en/examples/mnist
  
  # - icon: 🤗
  #   title: 集成
  #   details: 与HuggingFace Transformers、PyTorch Lightning、Hydra等主流框架的集成文档。
  #   link: /en/guide_cloud/integration/integration-pytorch-lightning.md

  # - icon: ⚡️
  #   title: API文档
  #   details: Python库和CLI的完整API文档
  #   link: en/api/api-index

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