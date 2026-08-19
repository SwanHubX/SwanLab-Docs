<script setup lang="ts">
/* 私有化部署文档 H1 旁的 APP 版本徽章。
 * 版本号由 `make bump-version V=vX.Y.Z` 统一替换
 * （匹配模式：<VersionBadge version="vX.Y.Z" />），请勿手写其他格式。
 * 配色用 --vp-c-text-1 / --vp-c-bg 反色：亮色模式黑底白字、暗色模式白底黑字，
 * 与 SwanLab 黑白品牌一致；内联小徽章形态，避免与顶部 AnnouncementBar 公告条
 * 争夺视觉层级。 */
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
  padding: 0.15em 0.55em;
  border-radius: 999px;
  background: var(--vp-c-text-1);
  color: var(--vp-c-bg);
  font-size: 0.36em;
  font-weight: 650;
  line-height: 1.6;
  letter-spacing: 0.04em;
  white-space: nowrap;
}
</style>
