---
layout: home

title: SwanLab官方文档
titleTemplate: 先进的AI团队协作与模型创新引擎

hero:
  name: SwanLab
  text: 开源、现代化设计<br>训练跟踪与可视化工具
  tagline: 深度学习训练跟踪与可视化工具，同时支持云端/离线使用，适配40+主流AI训练框架
  actions:
    - theme: alt
      text: 使用指南
      link: /guide_cloud/general/what-is-swanlab.md
    - theme: alt
      text: 立即使用
      link: https://swanlab.cn
    - theme: github
      text: GitHub
      link: https://github.com/SwanHubX/SwanLab

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

<!-- 精选文章部分 第一行 -->
<div class="featured-articles">
  <h2>✨精选内容✨</h2>
  <div class="article-container">
    <div class="article-card">
      <a href="/guide_cloud/integration/integration-huggingface-transformers" class="article-link">
        <div class="article-cover">
          <img src="/assets/swanlab-love-hf.jpg" alt="SwanLab与HuggingFace Transformers的集成">
        </div>
        <h3>SwanLab与HuggingFace Transformers的集成</h3>
      </a>
      <p>了解如何在Transformers训练流程中无缝集成SwanLab，实现高效的实验跟踪。</p>
    </div>
    <div class="article-card">
      <a href="/examples/pretrain_llm" class="article-link">
        <div class="article-cover">
          <img src="/assets/examples/pretrain_llm/llm.png" alt="从零预训练一个自己的大模型">
        </div>
        <h3>从零预训练一个自己的大模型</h3>
      </a>
      <p>本文则从如何自己实战预训练一个大语言模型的角度，使用Wiki数据集进行一个简单的从零预训练工作。</p>
    </div>
    <div class="article-card">
      <a href="/plugin/notification-email" class="article-link">
        <div class="article-cover">
          <img src="/zh/plugin/notification-email/logo.jpg" alt="SwanLab邮件通知插件：掌握进度更及时">
        </div>
        <h3>SwanLab邮件通知插件：掌握进度更及时</h3>
      </a>
      <p>深入了解SwanLab的插件生态，快速接入你的邮件、飞书、钉钉、企业微信等IM系统，让掌握进度快人一步</p>
    </div>
  </div>
</div>

<!-- 精选文章部分 第二行 -->
<div class="featured-articles" style="margin: 30px auto 0;">
  <div class="article-container">
    <div class="article-card">
      <a href="/guide_cloud/integration/integration-llama-factory" class="article-link">
        <div class="article-cover">
          <img src="/zh/guide_cloud/integration/llama_factory/0.png" alt="SwanLab与LLaMA Factory的集成">
        </div>
        <h3>SwanLab与LLaMA Factory的集成</h3>
      </a>
      <p>了解如何在LLaMA Factory训练流程中无缝集成SwanLab，实现高效的实验跟踪。</p>
    </div>
    <div class="article-card">
      <a href="/examples/stable_diffusion" class="article-link">
        <div class="article-cover">
          <img src="/zh/examples/images/stable_diffusion/01.png" alt="Stable Diffusion文生图训练实战">
        </div>
        <h3>Stable Diffusion文生图训练实战</h3>
      </a>
      <p>本文介绍如何从零到一训练一个Stable Diffusion火影忍者模型。</p>
    </div>
          <div class="article-card">
        <a href="/plugin/notification-lark" class="article-link">
          <div class="article-cover">
            <img src="/zh/plugin/notification-lark/logo.jpg" alt="SwanLab飞书通知插件：训练信息流转更高效">
          </div>
          <h3>SwanLab飞书通知插件：训练信息流转更高效</h3>
        </a>
        <p>深入了解SwanLab的插件生态，快速接入你的邮件、飞书、钉钉、企业微信等IM系统，让掌握进度快人一步</p>
      </div>
  </div>
</div>

<!-- 精选文章部分 第三行 -->
<div class="featured-articles" style="margin: 30px auto 0;">
  <div class="article-container">
    <div class="article-card">
      <a href="/examples/robot/lerobot-guide" class="article-link">
        <div class="article-cover">
          <img src="/zh/examples/robot/assets/demo.png" alt="LeRobot具身智能教程">
        </div>
        <h3>LeRobot具身智能教程</h3>
      </a>
      <p>基于LeRobot机械臂的具身智能教程，从零开始，手把手教你如何使用SwanLab进行具身智能训练。</p>
    </div>
    <div class="article-card">
      <a href="/examples/pretrain_llm" class="article-link">
        <div class="article-cover">
          <img src="/assets/examples/pretrain_llm/llm.png" alt="从零预训练一个自己的大模型">
        </div>
        <h3>SwanLab与verl的集成</h3>
      </a>
      <p>本文则从如何自己实战预训练一个大语言模型的角度，使用Wiki数据集进行一个简单的从零预训练工作。</p>
    </div>
    <div class="article-card">
      <a href="/plugin/notification-email" class="article-link">
        <div class="article-cover">
          <img src="/zh/plugin/notification-email/logo.jpg" alt="SwanLab邮件通知插件：掌握进度更及时">
        </div>
        <h3>SwanLab邮件通知插件：掌握进度更及时</h3>
      </a>
      <p>深入了解SwanLab的插件生态，快速接入你的邮件、飞书、钉钉、企业微信等IM系统，让掌握进度快人一步</p>
    </div>
  </div>
</div>

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

/* 自定义主题按钮样式 */
.VPButton.alt {
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 6px;
}

.VPButton.alt::before {
  content: "";
  display: inline-block;
  width: 16px;
  height: 16px;
  background-image: url("/guide.svg");
  background-size: contain;
  background-repeat: no-repeat;
  filter: var(--icon-filter, none);
}

/* 黑夜模式适配 */
.dark .VPButton.alt::before {
  --icon-filter: invert(1);
}

/* 为"立即使用"按钮设置不同的图标 */
.VPButton.alt[href="https://swanlab.cn"]::before {
  background-image: url("/icon_single.svg");
}

/* 自定义主题按钮样式 */
.VPButton.github {
  color: white;
  background-color: #121826;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 6px;
}

.VPButton.github::before {
  content: "";
  display: inline-block;
  width: 16px;
  height: 16px;
  background-image: url("/github.svg");
  background-size: contain;
  background-repeat: no-repeat;
}

.VPButton.github:hover {
  color: white;
  background-color:rgb(39, 39, 39);
}

/* 精选文章样式 */
.featured-articles {
  max-width: 1200px;
  margin: 60px auto 0;
  padding: 0 24px;
}

.featured-articles h2 {
  text-align: center;
  font-size: 24px;
  margin-bottom: 32px;
  color: var(--vp-c-text-1);
  font-weight: 600;
}

.article-container {
  display: grid;
  grid-template-columns: repeat(1, 1fr);
  gap: 24px;
}

@media (min-width: 640px) {
  .article-container {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (min-width: 960px) {
  .article-container {
    grid-template-columns: repeat(3, 1fr);
  }
}

.article-card {
  background: var(--vp-c-bg-soft);
  border-radius: 12px;
  overflow: hidden;
  transition: transform 0.3s, box-shadow 0.3s;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
}

.article-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
}

.article-link {
  text-decoration: none !important;
  color: inherit;
  display: block;
  border-bottom: none !important;
}

.article-cover {
  height: 160px;
  overflow: hidden;
}

.article-cover img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.5s;
}

.article-card:hover .article-cover img {
  transform: scale(1.05);
}

.article-card h3 {
  padding: 16px 20px 8px;
  margin: 0;
  font-size: 16px;
  color: var(--vp-c-text-1);
  transition: color 0.3s;
  border-bottom: none !important;
  font-weight: 500;
  line-height: 1.4;
}

.article-link:hover h3 {
  color: var(--vp-c-brand);
}

.article-card p {
  padding: 0 20px 20px;
  margin: 0;
  font-size: 13px;
  color: var(--vp-c-text-2);
  line-height: 1.5;
}

.read-more {
  display: inline-block;
  margin: 0 20px 20px;
  font-size: 14px;
  font-weight: 500;
  color: var(--vp-c-brand);
  text-decoration: none;
}

.read-more:hover {
  text-decoration: underline;
}
</style>


<!-- 分割线 -->
<div style="text-align: center; margin-top: 60px; padding: 10px; color: var(--vp-c-text-2); font-size: 14px;">
  <div style="border-top: 1px solid var(--vp-c-divider); margin: 20px 0;"></div>
  <p style="margin: 0 0;">情感机器（北京）科技有限公司</p>
  <p style="margin: 0 0;"><a href="https://beian.miit.gov.cn/" target="_blank" style="color: var(--vp-c-text-2); text-decoration: none;">京ICP备2024101706号-1</a> · 版权所有 ©2024 SwanLab</p>
</div>