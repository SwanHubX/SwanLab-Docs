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

  .header-button:hover {
    background-color: #f2f2f2;
    color: #397b89;
  }

  .button-icon {
    width: 14px;
    height: 14px;
  }

  /* 黑夜模式样式 */
  :root.dark .header-button {
    background-color: #2a2a2a;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
  }

  :root.dark .header-button:hover {
    background-color: #3a3a3a;
    color: #48a8b5;
  }
  </style>