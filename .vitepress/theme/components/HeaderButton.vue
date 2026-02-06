<template>
    <button class="header-button" @click="goToWebsite">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" class="button-icon"><g fill="currentColor"><path d="M224 297.69 52.5 440.61C31.9 457.81 0 443.41 0 416V96c0-27.4 31.9-41.7 52.5-24.6L224 214.32z" opacity=".4"></path><path d="M500.5 231.41a32.11 32.11 0 0 1 0 49.2l-192 160c-20.6 17.2-52.5 2.8-52.5-24.6V96c0-27.4 31.9-41.7 52.5-24.6z"></path></g></svg>
      {{ text }}
    </button>
  </template>
  
  <script setup>
  defineProps({
    text: {
      type: String,
      default: '官网'
    },
  })

  function goToWebsite() {
    window.open('https://swanlab.cn', '_blank')
  }
  </script>
  
  <style scoped>
  .header-button {
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
    background-color: var(--sl-button-bg);
    color: var(--sl-button-text);
    border: 1px solid var(--sl-button-border);
    border-radius: var(--sl-radius-sm);
    cursor: pointer;
    transition: all var(--sl-transition-base);
    margin-left: var(--sl-space-sm);
    vertical-align: middle;
  }

  .header-button:hover {
    background-color: var(--sl-button-hover-bg);
    border-color: var(--sl-button-hover-border);
    color: var(--sl-button-hover-text);
    transform: translateY(-1px);
  }

  .header-button:active {
    transform: translateY(0);
  }

  .button-icon {
    width: 14px;
    height: 14px;
    flex-shrink: 0;
  }
  </style>