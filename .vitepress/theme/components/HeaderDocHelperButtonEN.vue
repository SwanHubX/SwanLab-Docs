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
    <span>Docs Copilot</span>
  </a>
</template>

<style scoped>
.vp-header-doc-helper-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: var(--sl-space-sm);
  height: 32px;
  padding: 0 var(--sl-space-md);
  font-size: var(--sl-font-size-button);
  font-weight: var(--sl-font-weight-medium);
  line-height: 1;
  white-space: nowrap;
  text-decoration: none;
  background-color: var(--sl-button-bg);
  color: var(--sl-button-text);
  border: 1px solid var(--sl-button-border);
  border-radius: var(--sl-radius-sm);
  cursor: pointer;
  transition: all var(--sl-transition-base);
  margin-left: var(--sl-space-sm);
  vertical-align: middle;
}

.vp-header-doc-helper-btn:hover {
  background-color: var(--sl-button-hover-bg);
  border-color: var(--sl-button-hover-border);
  color: var(--sl-button-hover-text);
  transform: translateY(-1px);
}

.vp-header-doc-helper-btn:active {
  transform: translateY(0);
}

.doc-helper-logo {
  width: 14px;
  height: 14px;
  flex-shrink: 0;
}
</style>