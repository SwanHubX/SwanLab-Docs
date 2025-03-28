<template>
  <button class="github-button" @click="goToGithub">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" class="button-icon">
      <path fill="currentColor" d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
    </svg>
    <span v-if="starCount !== '--'" class="star-count">{{ starCount }}</span>
  </button>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const props = defineProps({
  repo: {
    type: String,
    default: 'SwanHubX/SwanLab'
  },
  timeout: {
    type: Number,
    default: 3000 // 3秒超时
  }
})

const starCount = ref('--')

onMounted(async () => {
  try {
    // 创建一个可以被中断的fetch请求
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), props.timeout);
    
    const response = await fetch(`https://api.github.com/repos/${props.repo}`, {
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    
    const data = await response.json();
    if (data.stargazers_count !== undefined) {
      starCount.value = formatStarCount(data.stargazers_count);
    }
  } catch (error) {
    console.error('Failed to fetch GitHub stars:', error);
    // 超时或其他错误时，保持默认值或不显示
  }
})

function formatStarCount(count) {
  if (count >= 1000) {
    return (count / 1000).toFixed(1) + 'k'
  }
  return count.toString()
}

function goToGithub() {
  window.open(`https://github.com/${props.repo}`, '_blank')
}
</script>

<style scoped>
.github-button {
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

.github-button:hover {
  background-color: #f2f2f2;
  color: #397b89;
}

.button-icon {
  width: 16px;
  height: 16px;
}

.button-text {
  font-weight: 500;
}

.star-count {
  font-weight: 600;
}

/* 黑夜模式样式 */
:root.dark .github-button {
  background-color: #2a2a2a;
  color: #e0e0e0;
  border: 1px solid #3a3a3a;
}

:root.dark .github-button:hover {
  background-color: #3a3a3a;
  color: #48a8b5;
}
</style> 