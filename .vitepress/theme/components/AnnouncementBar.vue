<script setup lang="ts">
import { computed, ref } from "vue";
import { useData } from "vitepress";

/* ============================================================
 * 顶部公告条 — 全站吸顶、可关闭、不持久化
 *
 * 关闭状态仅存于组件内存（ref）：刷新或重新打开网站时 banner 会再次
 * 出现；同一 SPA 会话内关闭后切换页面保持关闭，避免反复打扰。不写
 * localStorage，所以绝不会出现“关一次就再也看不到”的情况。
 *
 * 改文案/链接/开关 → 下方 ANNOUNCEMENT；logo → /public/trio-full.svg；
 * 视觉样式 → 本组件 <style> + styles/components.css 的 --vp-banner-height 联动段。
 * 多语言：text/cta 按语言各配一份，依据 VitePress 当前站点语言（useData().lang）
 * 切换；非英文一律回退到 zh 文案。
 * ============================================================ */
const ANNOUNCEMENT = {
  // 总开关：false 则整站不显示公告条。
  enabled: true,
  // 点击整条跳转的链接。
  url: "https://pytrio.com/home",
  // 按语言配置的文案（logo 已含 TRIO 字样，正文不再重复产品名）。
  text: {
    zh: "🎉 新产品上线!  无需 GPU 的 AI 后训练平台 · ",
    en: "🎉 New product launched! The GPU-free AI post-training platform · ",
  },
  // 右侧行动号召文案。
  cta: {
    zh: "立即体验 →",
    en: "Try it now →",
  },
};

const { lang } = useData();
const locale = computed<keyof typeof ANNOUNCEMENT.text>(() =>
  lang.value.startsWith("en") ? "en" : "zh",
);
const text = computed(() => ANNOUNCEMENT.text[locale.value]);
const cta = computed(() => ANNOUNCEMENT.cta[locale.value]);
// aria-label 也随语言切换，保持无障碍语义。
const closeLabel = computed(() => (locale.value === "en" ? "Close announcement" : "关闭公告"));

const visible = ref(ANNOUNCEMENT.enabled);
const dismiss = () => {
  visible.value = false;
};
</script>

<template>
  <div v-if="visible" class="announcement-bar">
    <a
      class="announcement-bar__inner"
      :href="ANNOUNCEMENT.url"
      target="_blank"
      rel="noopener noreferrer"
    >
      <img class="announcement-bar__logo" src="/trio-full.svg" alt="TRIO" />
      <span class="announcement-bar__text">{{ text }}</span>
      <span class="announcement-bar__cta">{{ cta }}</span>
    </a>
    <button class="announcement-bar__close" type="button" :aria-label="closeLabel" @click="dismiss">
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path
          d="M18 6 6 18M6 6l12 12"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
        />
      </svg>
    </button>
  </div>
</template>

<style scoped>
.announcement-bar {
  position: relative;
  display: none;
  /* 灰色玻璃半透明：用次级背景 bg-alt（比主背景 bg 更灰）+ alpha + 毛玻璃，
     与页面背景拉开区分；暗色模式随 bg-alt 自动适配为深灰玻璃。 */
  background: color-mix(in srgb, var(--vp-c-bg-alt) 82%, transparent);
  backdrop-filter: blur(12px) saturate(160%);
  -webkit-backdrop-filter: blur(12px) saturate(160%);
  border-bottom: 1px solid var(--vp-c-divider);
  font-size: 12px;
  line-height: 1;
}

/* md+ 才显示（移动端空间紧张，沿用主题内置 TopBanner 的断点） */
@media (min-width: 768px) {
  .announcement-bar {
    display: block;
  }
}

/* lg+ 吸顶：fixed 在页面最顶层，靠 --vp-banner-height + .Layout 的
 * padding-top（见 styles/components.css）让导航栏与正文下移腾位、不遮挡；
 * banner 关闭(v-if 移除)后 :has() 失配，腾位与变量回到 0，布局自动复原。
 * md（平板）时 banner 与导航栏都在文档流，无需 fixed。 */
@media (min-width: 1024px) {
  .announcement-bar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 60;
  }
}

.announcement-bar__inner {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  max-width: 1440px;
  height: 36px;
  margin: 0 auto;
  padding: 0 40px 0 16px;
  text-decoration: none;
}

.announcement-bar__logo {
  height: 20px;
  width: auto;
  flex-shrink: 0;
}

.announcement-bar__text {
  color: var(--vp-c-text-1);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.announcement-bar__cta {
  color: #ff4d4d;
  font-weight: 600;
  white-space: nowrap;
  flex-shrink: 0;
}

.announcement-bar__inner:hover .announcement-bar__cta {
  text-decoration: underline;
}

.announcement-bar__close {
  position: absolute;
  right: 10px;
  top: 50%;
  transform: translateY(-50%);
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  padding: 0;
  border: none;
  border-radius: 6px;
  background: transparent;
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition:
    color 0.2s,
    background-color 0.2s;
}

.announcement-bar__close:hover {
  color: var(--vp-c-text-1);
  background-color: var(--vp-c-default-soft);
}

.announcement-bar__close svg {
  width: 16px;
  height: 16px;
}
</style>
