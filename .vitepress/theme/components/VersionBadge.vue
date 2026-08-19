<script setup lang="ts">
/* 私有化部署文档 H1 旁的 APP 版本徽章。
 * 版本号由 `make bump-version V=vX.Y.Z` 统一替换
 * （匹配模式：<VersionBadge version="vX.Y.Z" />），请勿手写其他格式。
 * 描边样式：亮色模式黑字黑框、暗色模式白字白框（--vp-c-text-1 自动明暗反转），
 * 与 SwanLab 黑白品牌一致；无底色的内联小徽章形态，避免与顶部 AnnouncementBar
 * 公告条争夺视觉层级。 */
defineProps<{ version: string }>();
</script>

<template>
  <span class="version-badge">APP&nbsp;{{ version }}</span>
</template>

<style scoped>
.version-badge {
  display: inline-block;
  vertical-align: middle;
  margin-left: 0.35em;
  padding: 0.1em 0.55em;
  border: 1px solid var(--vp-c-text-1);
  border-radius: 999px;
  background: transparent;
  color: var(--vp-c-text-1);
  font-size: 0.36em;
  font-weight: 600;
  line-height: 1.6;
  letter-spacing: 0.04em;
  white-space: nowrap;
}
</style>
