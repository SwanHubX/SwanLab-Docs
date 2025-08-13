export interface BlogPost {
  title: string
  description: string
  date: string
  author: string
  category: 'release' | 'product' | 'how-to' | 'developer' | 'event' | 'company' | 'community'
  tags: string[]
  image?: string
  link: string
}

export const blogPosts: BlogPost[] = [
  {
    title: 'SwanLab v0.6.8 Release: New Experiment Tracking Experience',
    description: 'SwanLab v0.6.8 brings a completely new experiment tracking interface, supporting more data types and significant performance improvements to make your machine learning experiments more efficient.',
    date: '2024-12-20',
    author: 'SwanLab Team',
    category: 'release',
    tags: ['Release', 'Experiment Tracking', 'Performance'],
    image: '/assets/blog/swanlab-0.6.8.jpg',
    link: '/en/blog/swanlab-0.6.8-release'
  },
  {
    title: 'How to Build Efficient Machine Learning Workflows',
    description: 'Explore how to build end-to-end machine learning workflows using SwanLab, a complete guide from data preparation to model deployment.',
    date: '2024-12-15',
    author: 'Zhang Ming',
    category: 'how-to',
    tags: ['Workflow', 'Best Practices', 'Tutorial'],
    image: '', // No image to test placeholder
    link: '/en/blog/build-efficient-ml-workflow'
  },
  {
    title: 'SwanLab Integration Guide with Hugging Face',
    description: 'Deep dive into how to seamlessly integrate SwanLab with the Hugging Face ecosystem to accelerate your NLP project development.',
    date: '2024-12-10',
    author: 'Li Hua',
    category: 'how-to',
    tags: ['Integration', 'Hugging Face', 'NLP'],
    image: '', // No image to test placeholder
    link: '/en/blog/swanlab-huggingface-integration'
  },
  {
    title: 'Enterprise Machine Learning Experiment Management Solution',
    description: 'Learn how SwanLab Enterprise helps large teams manage complex machine learning experiments with a secure, scalable experiment tracking platform.',
    date: '2024-12-05',
    author: 'Wang Qiang',
    category: 'product',
    tags: ['Enterprise', 'Team Collaboration', 'Security'],
    image: '/assets/blog/enterprise-solution.jpg',
    link: '/en/blog/enterprise-ml-experiment-management'
  },
  {
    title: 'SwanLab Community Contribution Guide',
    description: 'Join the SwanLab open source community and learn how to contribute code, documentation, and ideas to advance machine learning tools together.',
    date: '2024-11-30',
    author: 'SwanLab Team',
    category: 'community',
    tags: ['Open Source', 'Contribution', 'Community'],
    image: '/assets/blog/community-contribution.jpg',
    link: '/en/blog/community-contribution-guide'
  },
  {
    title: 'Multimodal Experiment Tracking with SwanLab',
    description: 'Explore how to use SwanLab to track complex machine learning experiments involving multiple data types such as images, text, and audio.',
    date: '2024-11-25',
    author: 'Chen Jing',
    category: 'how-to',
    tags: ['Multimodal', 'Experiment Tracking', 'Tutorial'],
    image: '/assets/blog/multimodal-tracking.jpg',
    link: '/en/blog/multimodal-experiment-tracking'
  }
]

export const categories = {
  all: 'All',
  release: 'Release',
  product: 'Product',
  'how-to': 'How-to',
  developer: 'Developer',
  event: 'Event',
  company: 'Company',
  community: 'Community'
} 