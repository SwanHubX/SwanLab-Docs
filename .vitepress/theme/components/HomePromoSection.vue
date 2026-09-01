<script setup lang="ts">
/* ============================================================
 * 首页合作推广区 — CANN & SwanLab 大模型训练师成长计划
 *
 * 可选组件，无感下线：
 *   1. 临时下线 → 把下方 PROMO.enabled 置为 false（DOM 零残留）；
 *   2. 彻底移除 → 删除 zh/index.md 中的 <HomePromoSection /> 一行、
 *      本文件、以及 theme/index.ts 中的一行注册，三处互不牵连。
 *
 * 设计资源（/public/vendor/）：
 *   CANNlogo.svg / swanlab-logo.svg —— 白底圆形徽标（logo 置于白色
 *   圆形底上，暗色模式无需反色）；background-line.png —— 上半区
 *   的波浪装饰背景（透明底 PNG，用 background-image 引用，天然绕过
 *   medium-zoom 的图片缩放接管）。
 * 文案目前仅中文；若英文首页需要，在 en/index.md 插入标签并在此补
 * 英文文案即可。
 * ============================================================ */
const PROMO = {
  // 总开关：false 则整区不渲染。
  enabled: true,
  badge: "🤝 合作计划·PARTNERSHIP PROGRAM",
  titleLine1: "CANN & SwanLab",
  titleLine2: "大模型训练师成长计划",
  subtitle: "AI Trainer Growth Program",
  joinText: "我要加入",
  casesText: "查看案例",
  descTitle: "昇腾算力支持，助力 AI 模型训练实践",
  descBody:
    "本计划由华为 CANN 与 SwanLab 联合发起，面向有志于进入 AI 大模型训练领域的开发者、高校学生及团队，助力更多实践者成为 AI 时代的创新者。注册 SwanLab 并开通华为云账号，即可获得 100 卡时的 Ascend NPU 算力支持。通过使用算力、积累积分，可逐步解锁更多算力配额，持续学习与实践，不断提升 AI 模型训练能力。在这里，你可以体验昇腾训练框架，查阅完整的 CANN 技术文档，使用实战案例库，并获得专属技术支持，逐步掌握从入门到实践的 AI 模型训练全流程。无论你是计划入门大模型训练的开发者，还是希望验证国产 NPU 性能的算法工程师，都能在这里找到适合自己的成长路径与持续的算力支撑。",
};
</script>

<template>
  <section v-if="PROMO.enabled" class="home-promo">
    <div class="home-promo__divider"></div>
    <p class="home-promo__badge">{{ PROMO.badge }}</p>
    <div class="home-promo__card">
      <div class="home-promo__main">
        <div class="home-promo__logos">
          <span class="home-promo__logo-circle">
            <img class="no-zoomable" src="/vendor/CANNlogo.svg" alt="CANN" data-no-zoom />
          </span>
          <span class="home-promo__logo-x">×</span>
          <span class="home-promo__logo-circle">
            <img class="no-zoomable" src="/vendor/swanlab-logo.svg" alt="SwanLab" data-no-zoom />
          </span>
        </div>
        <div class="home-promo__info">
          <h2 class="home-promo__title">
            <span class="home-promo__title-line">{{ PROMO.titleLine1 }}</span>
            <span class="home-promo__title-line">{{ PROMO.titleLine2 }}</span>
          </h2>
          <p class="home-promo__subtitle">{{ PROMO.subtitle }}</p>
          <div class="home-promo__actions">
            <a
              class="home-promo__btn home-promo__btn--join"
              href="https://atomgit.com/cann/cann-learning-hub"
              target="_blank"
              rel="noopener noreferrer"
              >{{ PROMO.joinText }}</a
            >
            <a
              class="home-promo__btn home-promo__btn--cases"
              href="https://atomgit.com/cann/cann-learning-hub"
              target="_blank"
              rel="noopener noreferrer"
              >{{ PROMO.casesText }}</a
            >
          </div>
        </div>
      </div>
      <div class="home-promo__desc">
        <h3 class="home-promo__desc-title">{{ PROMO.descTitle }}</h3>
        <p class="home-promo__desc-body">{{ PROMO.descBody }}</p>
      </div>
    </div>
  </section>
</template>

<style scoped>
/* 设计参数（来自设计稿，1128×500 仅作比例参考，实际宽度随文档容器自适应）：
 * 灰色背景 #C3C3C340 仅包裹底部描述区（参考高度 186px）/
 * 红色按钮 #C7000B / 副标题 #B9B9B9 / 顶部 badge #B6121B */
.home-promo {
  margin: 0;
}

.home-promo__divider {
  margin: 32px 0;
  border-top: 1px solid var(--vp-c-divider);
}

.home-promo__badge {
  margin: 0 0 32px;
  padding: 0;
  text-align: center;
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0.5px;
  color: #b6121b;
}

.home-promo__card {
  border-radius: 24px;
  overflow: hidden;
  background: var(--vp-c-bg);
  box-shadow: 0 8px 40px rgba(0, 0, 0, 0.08);
  transition: box-shadow 0.3s ease;
}

.home-promo__card:hover {
  box-shadow: 0 16px 56px rgba(0, 0, 0, 0.14);
}

/* 上半区：白色底 + 波浪背景，左右各半、各自居中 */
.home-promo__main {
  display: flex;
  align-items: stretch;
  padding: 72px 48px 64px;
  background-image: url("/vendor/background-line.png");
  background-repeat: no-repeat;
  background-position: center bottom;
  background-size: 100% auto;
}

.home-promo__logos {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 20px;
}

.home-promo__logo-circle {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 136px;
  height: 136px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.65);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
}

.home-promo__logo-circle img {
  width: 80px;
  height: auto;
  margin: 0;
}

.home-promo__logo-x {
  font-size: 28px;
  color: #999999;
  line-height: 1;
}

.home-promo__info {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
}

.home-promo__title {
  margin: 0;
  padding: 0;
  border: none;
  font-size: 34px;
  font-weight: 700;
  line-height: 1.3;
  letter-spacing: normal;
  color: var(--vp-c-text-1);
}

.home-promo__title-line {
  display: block;
}

.home-promo__subtitle {
  margin: 12px 0 24px;
  padding: 0;
  font-size: 16px;
  font-weight: 700;
  color: #b9b9b9;
}

.home-promo__actions {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  justify-content: center;
}

.home-promo__btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 12px 36px;
  border-radius: 999px;
  font-size: 15px;
  font-weight: 600;
  color: #ffffff !important;
  text-decoration: none !important;
  transition: opacity 0.2s;
}

.home-promo__btn:hover {
  opacity: 0.88;
}

.home-promo__btn--join {
  background: #c7000b;
}

.home-promo__btn--cases {
  background: #121826;
}

/* 下半区：灰色背景仅包裹描述段落 */
.home-promo__desc {
  background: #c3c3c340;
  padding: 28px 48px 32px;
}

.home-promo__desc-title {
  margin: 0 0 16px;
  padding: 0;
  border: none;
  font-size: 18px;
  font-weight: 700;
  line-height: 1.5;
  color: var(--vp-c-text-1);
}

.home-promo__desc-body {
  margin: 0;
  padding: 0;
  font-size: 14px;
  line-height: 1.9;
  color: var(--vp-c-text-2);
}

@media (max-width: 768px) {
  .home-promo__main {
    flex-direction: column;
    gap: 32px;
    padding: 40px 24px 32px;
  }

  .home-promo__title {
    font-size: 26px;
  }

  .home-promo__desc {
    padding: 24px 24px 28px;
  }
}
</style>
