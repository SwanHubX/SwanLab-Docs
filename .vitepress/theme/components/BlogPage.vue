<template>
  <div class="blog-page">
    <!-- 页面标题 -->
    <div class="blog-header">
      <div class="blog-header-bg"></div>
      <div class="blog-header-content">
        <h1 class="blog-title">SwanLab 博客</h1>
        <p class="blog-subtitle">探索机器学习实验跟踪的最新动态、教程和最佳实践</p>
      </div>
    </div>

    <!-- 分类筛选 -->
    <div class="category-filter">
      <button 
        v-for="(label, key) in categories" 
        :key="key"
        :class="['category-btn', { active: selectedCategory === key }]"
        @click="filterByCategory(key)"
      >
        {{ label }}
      </button>
    </div>

    <!-- 博客文章网格 -->
    <div class="blog-grid">
      <article 
        v-for="post in filteredPosts" 
        :key="post.link"
        class="blog-card"
        @click="navigateToPost(post.link)"
      >
        <div class="card-image">
          <div v-if="!post.image" class="card-image-placeholder">
            <div class="placeholder-icon">📝</div>
            <div class="placeholder-text">博客文章</div>
          </div>
          <img 
            v-else
            :src="post.image" 
            :alt="post.title"
            @error="handleImageError"
            @load="handleImageLoad"
            :class="{ 'image-loaded': true }"
          />
          <div class="card-category">{{ categories[post.category] }}</div>
        </div>
        
        <div class="card-content">
          <div class="card-meta">
            <span class="card-date">{{ formatDate(post.date) }}</span>
            <span class="card-author">{{ post.author }}</span>
          </div>
          
          <h3 class="card-title">{{ post.title }}</h3>
          <p class="card-description">{{ post.description }}</p>
          
          <div class="card-tags">
            <span 
              v-for="tag in post.tags.slice(0, 3)" 
              :key="tag" 
              class="tag"
            >
              {{ tag }}
            </span>
          </div>
        </div>
      </article>
    </div>

    <!-- 空状态 -->
    <div v-if="filteredPosts.length === 0" class="empty-state">
      <p>没有找到相关文章</p>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRouter } from 'vitepress'

const props = defineProps({
  posts: {
    type: Array,
    required: true
  },
  categories: {
    type: Object,
    required: true
  }
})

const router = useRouter()
const selectedCategory = ref('all')

const filteredPosts = computed(() => {
  if (selectedCategory.value === 'all') {
    return props.posts
  }
  return props.posts.filter(post => post.category === selectedCategory.value)
})

const filterByCategory = (category) => {
  selectedCategory.value = category
}

const navigateToPost = (link) => {
  router.go(link)
}

const formatDate = (dateString) => {
  const date = new Date(dateString)
  return date.toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  })
}

const handleImageError = (event) => {
  // 隐藏错误的图片，显示占位符
  event.target.style.display = 'none'
  const cardImage = event.target.parentElement
  const placeholder = cardImage.querySelector('.card-image-placeholder')
  if (placeholder) {
    placeholder.style.display = 'flex'
  }
}

const handleImageLoad = (event) => {
  // 图片加载成功，隐藏占位符
  const cardImage = event.target.parentElement
  const placeholder = cardImage.querySelector('.card-image-placeholder')
  if (placeholder) {
    placeholder.style.display = 'none'
  }
}
</script>

<style scoped>
.blog-page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem 1rem;
}

.blog-header {
  text-align: center;
  margin-bottom: 3rem;
}

.blog-title {
  font-size: 3rem;
  font-weight: 700;
  color: var(--vp-c-text-1);
  margin-bottom: 1rem;
  background: linear-gradient(135deg, var(--vp-c-brand-1), #6366f1);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.blog-subtitle {
  font-size: 1.25rem;
  color: var(--vp-c-text-2);
  max-width: 600px;
  margin: 0 auto;
  line-height: 1.6;
}

.category-filter {
  display: flex;
  justify-content: center;
  gap: 0.5rem;
  margin-bottom: 3rem;
  flex-wrap: wrap;
}

.category-btn {
  padding: 0.75rem 1.5rem;
  border: 2px solid var(--vp-c-divider);
  background: transparent;
  color: var(--vp-c-text-2);
  border-radius: 50px;
  cursor: pointer;
  transition: all 0.2s ease;
  font-weight: 500;
}

.category-btn:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

.category-btn.active {
  background: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
  color: white;
}

.blog-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 2rem;
  margin-bottom: 3rem;
  max-width: 100%;
}

.blog-card {
  background: var(--vp-c-bg-soft);
  border-radius: 16px;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.3s ease;
  border: 1px solid var(--vp-c-divider);
}

.blog-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
  border-color: var(--vp-c-brand-1);
}

.card-image {
  position: relative;
  height: 200px;
  overflow: hidden;
}

.card-image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.3s ease;
}

.card-image-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  background: linear-gradient(135deg, var(--vp-c-bg-soft-up), var(--vp-c-bg-soft));
  color: var(--vp-c-text-3);
  border: 2px dashed var(--vp-c-divider);
}

.placeholder-icon {
  font-size: 3rem;
  margin-bottom: 0.5rem;
  opacity: 0.7;
}

.placeholder-text {
  font-size: 0.875rem;
  font-weight: 500;
  opacity: 0.8;
}

.blog-card:hover .card-image img {
  transform: scale(1.05);
}

.card-category {
  position: absolute;
  top: 1rem;
  left: 1rem;
  background: var(--vp-c-brand-1);
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 20px;
  font-size: 0.875rem;
  font-weight: 500;
}

.card-content {
  padding: 1.5rem;
}

.card-meta {
  display: flex;
  gap: 1rem;
  margin-bottom: 1rem;
  font-size: 0.875rem;
  color: var(--vp-c-text-3);
}

.card-date,
.card-author {
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.card-title {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin-bottom: 0.75rem;
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.card-description {
  color: var(--vp-c-text-2);
  line-height: 1.6;
  margin-bottom: 1rem;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.card-tags {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.tag {
  background: var(--vp-c-bg-soft-up);
  color: var(--vp-c-text-2);
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 500;
}

.empty-state {
  text-align: center;
  padding: 3rem;
  color: var(--vp-c-text-3);
}

/* 响应式设计 */
@media (min-width: 1200px) {
  .blog-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (min-width: 768px) and (max-width: 1199px) {
  .blog-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 767px) {
  .blog-title {
    font-size: 2rem;
  }
  
  .blog-subtitle {
    font-size: 1rem;
  }
  
  .blog-grid {
    grid-template-columns: 1fr;
    gap: 1.5rem;
  }
  
  .category-filter {
    gap: 0.25rem;
  }
  
  .category-btn {
    padding: 0.5rem 1rem;
    font-size: 0.875rem;
  }
}
</style> 