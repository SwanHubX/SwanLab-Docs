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
    aria-label="Document Copilot"
  >
    <img
      v-if="!isDark"
      src="/assets/chat-black.svg"
      alt="Document Copilot"
      class="doc-helper-logo"
    />
    <img
      v-else
      src="/assets/chat-white.svg"
      alt="Document Copilot"
      class="doc-helper-logo"
    />
    <span>Swanlab Copilot</span>
  </a>
</template>

<style scoped>
.vp-header-doc-helper-btn {
  background-color: #ffffff;
  color: #374a52;
  border: 1px solid #e0e0e0;
  padding: 4px 12px;
  border-radius: 4px;
  cursor: pointer;
  margin-left: 10px;
  font-size: 14px;
  font-weight: 500;
  line-height: 1.5;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-top: auto;
  margin-bottom: auto;
  gap: 6px;
}
.vp-header-doc-helper-btn:hover {
  background-color: #f2f2f2;
  color: #397b89;
}
.vp-header-doc-helper-btn .doc-helper-logo {
  width: 18px;
  height: 18px;
  margin-right: 8px;
  display: inline-block;
  vertical-align: middle;
}
.vp-header-doc-helper-btn span {
  display: inline-block;
  vertical-align: middle;
}


/* 黑夜模式样式 */
:root.dark .vp-header-doc-helper-btn {
  background-color: #2a2a2a;
  color: #e0e0e0;
  border: 1px solid #3a3a3a;
}

:root.dark .vp-header-doc-helper-btn:hover {
  background-color: #3a3a3a;
  color: #48a8b5;
}
</style>
