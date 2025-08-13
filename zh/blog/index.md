---
title: SwanLab 博客
description: 探索机器学习实验跟踪的最新动态、教程和最佳实践
---

<script setup>
import { ref, onMounted } from 'vue'
import BlogPage from '../../.vitepress/theme/components/BlogPage.vue'
import { blogPosts, categories } from './blog-data'

const posts = ref(blogPosts)
const cats = ref(categories)
</script>

<BlogPage :posts="posts" :categories="cats" />
