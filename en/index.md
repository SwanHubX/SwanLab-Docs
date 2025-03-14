---
layout: home

title: SwanLab Docs
titleTemplate: Advanced AI Team Collaboration and Model Innovation Engine

hero:
  name: SwanLab
  text: Track and Visualize Your AI Experiments
  tagline: A deep learning training tracking and visualization tool that supports both cloud and offline use, compatible with over 30 mainstream AI training frameworks.
  actions:
    - theme: brand
      text: Documentation
      link: /en/guide_cloud/general/what-is-swanlab.md
    - theme: brand
      text: Website
      link: https://swanlab.cn
    - theme: alt
      text: GitHub
      link: https://github.com/SwanHubX/SwanLab
  image:
    src: /page.png
    alt: VitePress

features:
  - title: 🚢 Quick Start
    details: Install SwanLab and start tracking your AI experiments in minutes.
    link: /en/guide_cloud/general/quick-start.md
  
  - title: 📚 Examples
    details: SwanLab official examples, better understand the role and function of SwanLab in the AI pipeline.
    link: /en/examples/mnist
  
  - title: 🤗 Integration
    details: Integration documentation with HuggingFace Transformers, PyTorch Lightning, Hydra, etc.
    link: /en/guide_cloud/integration/integration-pytorch-lightning.md

  - title: ⚡️ API Docs
    details: Complete API documentation for the Python library and CLI.
    link: en/api/api-index

  - title: 🔌 Plugin
    details: Extend the functionality of SwanLab.
    link: /en/plugin/plugin-index.md

  - title: 💻 Self-hosted
    details: Docker deployment and enterprise version.
    link: /en/guide_cloud/self_host/docker-deploy.md

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


<!-- 分割线 -->
<div style="text-align: center; margin-top: 120px; padding: 10px; color: var(--vp-c-text-2); font-size: 14px;">
  <div style="border-top: 1px solid var(--vp-c-divider); margin: 20px 0;"></div>
  <p style="margin: 0 0;">Emotion Machine (Beijing) Technology Co., Ltd.</p>
  <p style="margin: 0 0;">Copyright © 2024 SwanLab</p>
</div>