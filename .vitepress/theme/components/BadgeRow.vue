<script setup lang="ts">
/* 页头 badge 行：Visualize in SwanLab、Open in Colab 等跳转徽章。
 *
 * 用法（在 Markdown 中直接写，若页首有 :::info 提示框则放在提示框下面）：
 *   <BadgeRow :badges="[
 *     { alt: 'Visualize in SwanLab', img: 'https://.../badge1.svg', href: 'https://swanlab.cn/...' },
 *     { alt: 'Open in Colab',        img: 'https://.../colab.svg',  href: 'https://colab.research.google.com/...' },
 *   ]" />
 *
 * 扩展方式：新增 badge 只需向数组追加一项 { alt, img, href }。
 * 渲染顺序由 priority() 强制收敛（SwanLab 系永远排在 Colab 之前，
 * 未识别的类型按声明顺序排在最后），作者手写顺序不影响最终展示。 */
import { computed } from "vue";

interface BadgeItem {
  alt: string;
  img: string;
  href: string;
}

const props = defineProps<{ badges: BadgeItem[] }>();

const priority = (img: string): number => {
  if (img.includes("badge1.svg")) return 0; // Visualize in SwanLab
  if (img.includes("colab")) return 1; // Open in Colab
  return 2; // 未来新增类型：保持声明顺序排在最后
};

const sortedBadges = computed(() =>
  props.badges
    .map((badge, index) => ({ badge, index }))
    .sort((a, b) => priority(a.badge.img) - priority(b.badge.img) || a.index - b.index)
    .map(({ badge }) => badge),
);
</script>

<template>
  <div class="badge-row">
    <a
      v-for="badge in sortedBadges"
      :key="badge.href"
      :href="badge.href"
      target="_blank"
      rel="noopener noreferrer"
    >
      <img :src="badge.img" :alt="badge.alt" />
    </a>
  </div>
</template>

<style scoped>
.badge-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.badge-row a {
  display: inline-flex;
}

/* 抵消 .vp-doc img 的全局边距，保证徽章严格同排 */
.badge-row img {
  margin: 0;
}
</style>
