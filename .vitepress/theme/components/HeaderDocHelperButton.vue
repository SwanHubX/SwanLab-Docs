<script setup>
import { useData } from 'vitepress'
const { isDark } = useData()
</script>

<template>
  <a
    class="vp-header-doc-helper-btn"
    href="https://chat.swanlab.cn"
    target="_blank"
    rel="noopener"
    aria-label="文档助手"
  >
    <img
      v-if="!isDark"
      src="/assets/chat-black.svg"
      alt="文档助手"
      class="doc-helper-logo"
    />
    <img
      v-else
      src="/assets/chat-white.svg"
      alt="文档助手"
      class="doc-helper-logo"
    />
    <span>文档助手</span>
  </a>
</template>

<style scoped>
.vp-header-doc-helper-btn {
  /* 按钮样式 */
  background-color: #ffffff;
  color: #374a52;
  border: 1px solid #e0e0e0;
  padding: 4px 12px;
  border-radius: 4px;
  cursor: pointer;
  margin-left: 10px; /* 与其他元素保持间距 */
  font-size: 14px;
  font-weight: 500;
  line-height: 1.5;
  height: 28px; /* 固定高度，避免占满header */
  display: flex; /* 改为flex以实现垂直居中 */
  align-items: center; /* 垂直居中 */
  justify-content: center; /* 水平居中 */
  margin-top: auto; /* 自动上边距 */
  margin-bottom: auto; /* 自动下边距 */
  gap: 6px; /* 图标和文字之间的间距 */
}

.vp-header-doc-helper-btn:hover {
  background-color: #f2f2f2;
  color: #397b89;
}

.doc-helper-logo {
  width: 14px;
  height: 14px;
}

/* 黑夜模式样式 */
:root.dark .vp-header-doc-helper-btn {
  background-color: #2a2a2a;
  color: #e0e0e0;
  border: 1px solid #3a3a3a;
}

:root.dark .vp-header-doc-helper-btn:hover {
  background-color: #3a3a3a;
  color: #e0e0e0;
}
</style>
