---
layout: home

title: SwanLab博客
titleTemplate: 先进的AI团队协作与模型创新引擎
---

<!-- 精选文章部分 -->
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

<!-- 精选文章部分 -->
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

/* AI文档助手按钮样式 */
.VPButton.ai-assistant {
  position: relative;
  color: white;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 6px;
  border: none;
  background: linear-gradient(-45deg, #54d3ff, #b17af0, #9f87f0, #5ac8ff);
  background-size: 300% 300%;
  box-shadow: 0 0 15px rgba(177, 122, 240, 0.5);
  animation: gradient-animation 3s ease infinite, pulse 1.5s infinite alternate;
  transition: all 0.3s ease;
  overflow: hidden;
}

.VPButton.ai-assistant::before {
  content: "";
  display: inline-block;
  width: 16px;
  height: 16px;
  background-image: url("/assets/chat-white.svg");
  background-size: contain;
  background-repeat: no-repeat;
}

.VPButton.ai-assistant:hover {
  transform: translateY(-2px);
  box-shadow: 0 0 30px rgba(177, 122, 240, 0.8);
  animation-play-state: paused;
}

@keyframes gradient-animation {
  0% {
    background-position: 0% 50%;
  }
  50% {
    background-position: 100% 50%;
  }
  100% {
    background-position: 0% 50%;
  }
}

@keyframes pulse {
  0% {
    box-shadow: 0 0 10px rgba(177, 122, 240, 0.5);
    transform: scale(1);
  }
  100% {
    box-shadow: 0 0 25px rgba(84, 211, 255, 0.8);
    transform: scale(1.02);
  }
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
