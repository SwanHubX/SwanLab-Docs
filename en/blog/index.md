---
title: SwanLab Blog
description: Explore the latest updates, tutorials, and best practices in machine learning experiment tracking
---

<script setup>
import { ref, onMounted } from 'vue'
import BlogPageEN from '../../.vitepress/theme/components/BlogPageEN.vue'
import { blogPosts, categories } from './blog-data'

const posts = ref(blogPosts)
const cats = ref(categories)
</script>

<BlogPageEN :posts="posts" :categories="cats" /> 