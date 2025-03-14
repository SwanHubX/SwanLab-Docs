---
layout: home

title: SwanLab官方文档
titleTemplate: 先进的AI团队协作与模型创新引擎

hero:
  name: SwanLab
  text: 开源、现代化设计<br>训练跟踪与可视化工具
  tagline: 深度学习训练跟踪与可视化工具，同时支持云端/离线使用，适配30+主流AI训练框架
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
  background-image: url("data:image/svg+xml;charset=utf-8;base64,PHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHdpZHRoPScyNCcgaGVpZ2h0PScyNCcgdmlld0JveD0nMCAwIDI0IDI0JyBmaWxsPSdub25lJyBzdHJva2U9J2N1cnJlbnRDb2xvcicgc3Ryb2tlLXdpZHRoPScyJyBzdHJva2UtbGluZWNhcD0ncm91bmQnIHN0cm9rZS1saW5lam9pbj0ncm91bmQnPjxwYXRoIGQ9J000IDE5LjV2LTE1QTIuNSAyLjUgMCAwIDEgNi41IDJIMTlhMSAxIDAgMCAxIDEgMXYxOGExIDEgMCAwIDEtMSAxSDYuNWExIDEgMCAwIDEgMC01SDIwJy8+PHBhdGggZD0nTTggMTFoOCcvPjxwYXRoIGQ9J004IDdoNicvPjwvc3ZnPg==");
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
  background-image: url("data:image/svg+xml;charset=utf-8;base64,PHN2ZyB3aWR0aD0nMzAnIGhlaWdodD0nMzUnIHZpZXdCb3g9JzAgMCAzMCAzNScgZmlsbD0nbm9uZScgeG1sbnM9J2h0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnJz48cGF0aCBkPSdNNC42MTQ3NSAxNi44MzMyQzQuNDUzOTUgMTYuOTAzMSA0LjMwMDE1IDE2Ljk4ODIgNC4xNTU0NSAxNy4wODcyTDQuMTA4ODUgMTcuMTI1M0M0LjA5MDY1IDE3LjEzNyA0LjA3MzY1IDE3LjE1MDUgNC4wNTgwNSAxNy4xNjU2QzAuOTk5NDQ5IDE5LjU0NDcgLTAuNjM4NzUxIDIzLjM4MDEgMC4yMzMyNDkgMjcuMjIzOUMwLjcyNTc0OSAyOS40MjgyIDEuOTYyOTUgMzEuMzk1MyAzLjczNjU1IDMyLjc5MzdDNS41MTAxNSAzNC4xOTIxIDcuNzExNTUgMzQuOTM2NCA5Ljk2OTk1IDM0LjkwMTFIMTAuMzE0OUMxMS43OTU1IDM0LjgzNzcgMTMuMjQyNyAzNC40NDA0IDE0LjU0ODEgMzMuNzM4OUMxNS44NTM1IDMzLjAzNzQgMTYuOTgzNCAzMi4wNDk3IDE3Ljg1MzMgMzAuODQ5OUMxOC43MjMxIDI5LjY1IDE5LjMxMDQgMjguMjY5IDE5LjU3MTIgMjYuODEwMkMxOS44MzIgMjUuMzUxMyAxOS43NTk2IDIzLjg1MjQgMTkuMzU5NCAyMi40MjU1QzE4Ljk2NzggMjEuMDIyMSAxOC4xNTI5IDE5LjcyODggMTcuNDM5NSAxOC40NjczQzE2LjE4NjUgMTYuMjUxOSAxNC45MzA2IDE0LjAzNzEgMTMuNjcxOSAxMS44MjMxQzEzLjI5OTQgMTEuMTY5MSAxMi45MjI2IDEwLjUxOTIgMTIuNTU0MyA5Ljg2NTE2QzEyLjA2OTYgOS4wMDE1NiAxMS4zMDk3IDguMDYxNzYgMTEuMjY5NSA3LjAzNzI2QzExLjIyOTMgNi4wMTI4NiAxMi4wNjMyIDUuMDI0MzYgMTMuMDgzNSA0LjgyNzQ2QzE1LjAwNTQgNC40NjEzNiAxNS45MDcxIDYuNTQ0MTYgMTUuOTExMyA2LjU0MTk2QzE2LjE0NDYgNy4wOTk2NiAxNi41ODA4IDcuNTQ3OTYgMTcuMTMxOSA3Ljc5NjM2QzE3LjY4MyA4LjA0NDY2IDE4LjMwNzggOC4wNzQ1NSAxOC44ODAxIDcuODc5OTVDMTkuNDUyNCA3LjY4NTM1IDE5LjkyOTQgNy4yODA3NSAyMC4yMTQ4IDYuNzQ3OTVDMjAuNTAwMyA2LjIxNTA1IDIwLjU3MjggNS41OTM3NiAyMC40MTc3IDUuMDA5NTZDMTkuOTQ3MSAzLjQ3Mjc2IDE4Ljk2NCAyLjE0MzY2IDE3LjYzMjIgMS4yNDM5NkMxNi40NjEgMC40MDYzNTYgMTUuMDUwMSAtMC4wMjk1NDQyIDEzLjYxMDUgMC4wMDE1NTU3NUMxMi42MDQzIDAuMDA3MDU1NzUgMTEuNjEwMyAwLjIyMjI1NSAxMC42OTIgMC42MzMzNTVDOS43NzM2NSAxLjA0NDM2IDguOTUwOTUgMS42NDIzNiA4LjI3NjU1IDIuMzg5MDZDNy41MTAxNSAzLjI0NTU2IDYuOTYxMDUgNC4yNzM4NiA2LjY3NTU1IDUuMzg3MTZDNi4zOTAwNSA2LjUwMDQ2IDYuMzc2NTUgNy42NjYwNSA2LjYzNjE1IDguNzg1NjZDNi43Mzk4NSA5LjIwNjE2IDYuODgxNTUgOS42MTYzNiA3LjA1OTQ1IDEwLjAxMTJDNy41NjUzNSAxMS4xNTg1IDguMzA4MzUgMTIuMjA0MSA4LjkzNDg1IDEzLjI4OTlDOS43MDM5NSAxNC42MjM0IDEwLjQ3MjkgMTUuOTU3NiAxMS4yNDIgMTcuMjkyNkMxMi4wMTEgMTguNjI3NSAxMi43ODcxIDE5Ljk2OCAxMy41NzAzIDIxLjMxNDJDMTQuMjA1MyAyMi40Mjc2IDE0Ljg5NzQgMjMuNDAxMiAxNC45NjczIDI0LjczOUMxNC45OTY3IDI1LjQ1MjUgMTQuODc4MSAyNi4xNjQ1IDE0LjYxOSAyNi44M0MxNC4zNTk5IDI3LjQ5NTUgMTMuOTY1OCAyOC4xMDAxIDEzLjQ2MTYgMjguNjA1OUMxMi45NTc0IDI5LjExMTcgMTIuMzU0IDI5LjUwNzcgMTEuNjg5MyAyOS43Njg5QzExLjAyNDYgMzAuMDMwMSAxMC4zMTMxIDMwLjE1MDkgOS41OTk0NSAzMC4xMjM3QzcuMTIyOTUgMzAuMDExNiA1LjI1Mzk1IDI4LjE3NjQgNC44NzcyNSAyNS43NzYxQzQuNDA1MjUgMjIuNzQ1MSA2Ljc2NTI1IDIxLjA4MTQgNi43NjUyNSAyMS4wODE0QzcuMjYyNjUgMjAuNzg4OCA3LjYzNzA1IDIwLjMyNTggNy44MTkxNSAxOS43NzgzQzguMDAxMjUgMTkuMjMwOCA3Ljk3ODc1IDE4LjYzNTggNy43NTU4NSAxOC4xMDM2QzcuNTMyODUgMTcuNTcxNCA3LjEyNDU1IDE3LjEzODEgNi42MDY1NSAxNi44ODM4QzYuMDg4NTUgMTYuNjI5NiA1LjQ5NTk1IDE2LjU3MTggNC45Mzg1NSAxNi43MjExQzQuODMxOTUgMTYuNzUxMSA0LjcyNzM1IDE2Ljc4NzggNC42MjUzNSAxNi44MzExJyBmaWxsPSdibGFjaycvPjxwYXRoIGQ9J00zMC4wMDIxIDI1LjA5MTZDMzAuMDAyOCAyNC4yMzYzIDI5LjkzMiAyMy4zODI0IDI5Ljc5MDUgMjIuNTM4OUMyOS41NzAzIDIxLjIwODQgMjkuMTU3NCAxOS45MTcxIDI4LjU2NDkgMTguNzA1NkMyOC4yMDkzIDE3Ljk4MzggMjcuNzYwNiAxNy4yNjIxIDI3LjA1NzggMTYuODY4NEMyNi43MDYgMTYuNjcgMjYuMzA4NSAxNi41NjY2IDI1LjkwNDYgMTYuNTY4NEMyNS41MDA3IDE2LjU3MDEgMjUuMTA0MiAxNi42NzY5IDI0Ljc1NCAxNi44NzgyQzI0LjQwMzkgMTcuMDc5NiAyNC4xMTIxIDE3LjM2ODYgMjMuOTA3NCAxNy43MTY4QzIzLjcwMjcgMTguMDY1IDIzLjU5MjEgMTguNDYwNSAyMy41ODY1IDE4Ljg2NDRDMjMuNTY3NSAxOS43NTU1IDI0LjA1MDEgMjAuNTU3NyAyNC40MDk5IDIxLjM3OUMyNS4wMDggMjIuNzQ3NyAyNS4yNzg5IDI0LjIzNzEgMjUuMjAxIDI1LjcyODhDMjUuMTIzMSAyNy4yMjA1IDI0LjY5ODYgMjguNjczNiAyMy45NjEyIDI5Ljk3MjZDMjMuOTE1OSAzMC4wNDk3IDIzLjg3NTYgMzAuMTI5NiAyMy44NDA1IDMwLjIxMThDMjMuNjkwNiAzMC41NjEgMjMuNjEyMiAzMC45MzY3IDIzLjYwOTggMzEuMzE2N0MyMy42MjU1IDMxLjg2MTcgMjMuODI4MyAzMi4zODQ3IDI0LjE4NCAzMi43OTc5QzI0LjUzOTggMzMuMjExMSAyNS4wMjY5IDMzLjQ4OTMgMjUuNTYzNSAzMy41ODU3QzI2Ljg3NzkgMzMuODIwNyAyNy45MjE0IDMyLjgzNDMgMjguNDczOSAzMS43MzM3QzI5LjQ4OTMgMjkuNjY3MyAzMC4wMTI0IDI3LjM5MzkgMzAuMDAyMSAyNS4wOTE2WicgZmlsbD0nYmxhY2snLz48cGF0aCBkPSdNMTguNDI4NSA5LjQ0NTc3QzE4LjM4NSA5LjQ1NDA3IDE4LjM0MTkgOS40NjQ2NyAxOC4yOTk0IDkuNDc3NDdDMTguMjEyOCA5LjUwMTY3IDE4LjEzNDggOS41NDk4NyAxOC4wNzQ1IDkuNjE2NDdDMTguMDE0MSA5LjY4MzE3IDE3Ljk3MzggOS43NjU0NyAxNy45NTgzIDkuODU0MDdDMTcuOTQyOCA5Ljk0MjY3IDE3Ljk1MjcgMTAuMDMzOCAxNy45ODY5IDEwLjExN0MxOC4wMjEgMTAuMjAwMiAxOC4wNzggMTAuMjcyIDE4LjE1MTIgMTAuMzI0MkMxOS4yMjc5IDExLjEzODMgMjAuMTM1OSAxMi4xNTQxIDIwLjgyNDYgMTMuMzE1QzIxLjAxMTcgMTMuNjMzOSAyMS4xNzQ1IDEzLjk2NjQgMjEuMzExNCAxNC4zMDk4QzIxLjQyNTcgMTQuNjA2MiAyMS41NzgxIDE0Ljk5MzUgMjEuOTYzNCAxNC45NzI0QzIyLjI2MzkgMTQuOTU3NSAyMi4zODY3IDE0LjcwMTQgMjIuNDY5MiAxNC40NDUzQzIyLjg2MTkgMTMuMTEzNSAyMi45ODQzIDExLjcxNjYgMjIuODI5MSAxMC4zMzY5QzIyLjcxODUgOS41ODUwNyAyMi41NDU3IDguODQzODcgMjIuMzEyNiA4LjEyMDY3QzIyLjI3NzMgNy45NDU1NyAyMi4xOTU0IDcuNzgzMjcgMjIuMDc1NSA3LjY1MDc3QzIxLjk5OTEgNy41ODE0NyAyMS45MDIzIDcuNTM4ODcgMjEuNzk5NyA3LjUyOTM3QzIxLjY5NyA3LjUxOTg3IDIxLjU5NCA3LjU0Mzg3IDIxLjUwNjEgNy41OTc4N0MyMS40MjI5IDcuNjY1MzcgMjEuMzUxNCA3Ljc0NjE3IDIxLjI5NDUgNy44MzcwN0MyMC45NDE3IDguMzExNjcgMjAuNDg4NSA4LjcwMjY4IDE5Ljk2NzMgOC45ODIxOEMxOS43MDQ1IDkuMTIxOTggMTkuNDI3OCA5LjIzNDA3IDE5LjE0MTggOS4zMTY1N0MxOC45MDI3IDkuMzg2NDcgMTguNjU5MiA5LjM5NzA3IDE4LjQyODUgOS40NDU3N1onIGZpbGw9JyNDMjFFMzEnLz48L3N2Zz4g");
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
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'%3E%3Cpath d='M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z'/%3E%3C/svg%3E");
  background-size: contain;
  background-repeat: no-repeat;
}

.VPButton.github:hover {
  color: white;
  background-color:rgb(39, 39, 39);
}
</style>

<!-- 分割线 -->
<div style="text-align: center; margin-top: 120px; padding: 10px; color: var(--vp-c-text-2); font-size: 14px;">
  <div style="border-top: 1px solid var(--vp-c-divider); margin: 20px 0;"></div>
  <p style="margin: 0 0;">情感机器（北京）科技有限公司</p>
  <p style="margin: 0 0;"><a href="https://beian.miit.gov.cn/" target="_blank" style="color: var(--vp-c-text-2); text-decoration: none;">京ICP备2024101706号-1</a> · 版权所有 ©2024 SwanLab</p>
</div>