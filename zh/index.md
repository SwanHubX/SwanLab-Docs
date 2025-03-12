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
      text: 使用指南
      link: /guide_cloud/general/what-is-swanlab.md
    - theme: brand
      text: 官网
      link: https://swanlab.cn
    - theme: alt
      text: 文档GitHub
      link: https://github.com/SwanHubX/SwanLab-Docs
  image:
    src: /page.png
    alt: VitePress

features:
  - title: 🚢 快速开始
    details: 安装SwanLab，并在几分钟内掌握如何跟踪你的机器学习实验。
    link: /guide_cloud/general/quick-start.md

  - title: 🤗 框架集成
    details: 与HuggingFace Transformers、PyTorch Lightning、Hydra等主流框架的集成文档。
    link: /guide_cloud/integration/integration-huggingface-transformers.md
  
  - title: 📚 实战案例
    details: SwanLab官方案例合集，更好地理解SwanLab在人工智能pipeline中扮演的角色和作用。
    link: /examples/mnist
  
  - title: ⚡️ API文档
    details: Python库与命令行的完整API文档
    link: /api/api-index

  - title: 🔌 插件
    details: 扩展SwanLab的功能
    link: /plugin/plugin-index.md
  
  - title: 💻 私有化部署
    details: 离线查看实验结果，支持Docker部署
    link: /guide_cloud/self_host/docker-deploy.md

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
  <p style="margin: 0 0;">情感机器（北京）科技有限公司</p>
  <p style="margin: 0 0;"><a href="https://beian.miit.gov.cn/" target="_blank" style="color: var(--vp-c-text-2); text-decoration: none;">京ICP备2024101706号-1</a> · 版权所有 ©2024 SwanLab</p>
</div>