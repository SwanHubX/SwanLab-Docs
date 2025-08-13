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
    title: 'SwanLab v0.6.8 发布：全新实验跟踪体验',
    description: 'SwanLab v0.6.8 带来了全新的实验跟踪界面，支持更多数据类型，性能大幅提升，让您的机器学习实验更加高效。',
    date: '2024-12-20',
    author: 'SwanLab Team',
    category: 'release',
    tags: ['发布', '实验跟踪', '性能优化'],
    image: '/assets/blog/swanlab-0.6.8.jpg',
    link: '/blog/swanlab-0.6.8-release'
  },
  {
    title: '如何构建高效的机器学习工作流',
    description: '探索如何使用 SwanLab 构建端到端的机器学习工作流，从数据准备到模型部署的完整指南。',
    date: '2024-12-15',
    author: '张明',
    category: 'how-to',
    tags: ['工作流', '最佳实践', '教程'],
    image: '', // 不设置图片，测试占位符
    link: '/blog/build-efficient-ml-workflow'
  },
  {
    title: 'SwanLab 与 Hugging Face 集成指南',
    description: '深入了解如何将 SwanLab 与 Hugging Face 生态系统无缝集成，加速您的 NLP 项目开发。',
    date: '2024-12-10',
    author: '李华',
    category: 'how-to',
    tags: ['集成', 'Hugging Face', 'NLP'],
    image: '', // 不设置图片，测试占位符
    link: '/blog/swanlab-huggingface-integration'
  },
  {
    title: '企业级机器学习实验管理解决方案',
    description: '了解 SwanLab 企业版如何帮助大型团队管理复杂的机器学习实验，提供安全、可扩展的实验跟踪平台。',
    date: '2024-12-05',
    author: '王强',
    category: 'product',
    tags: ['企业版', '团队协作', '安全'],
    image: '/assets/blog/enterprise-solution.jpg',
    link: '/blog/enterprise-ml-experiment-management'
  },
  {
    title: 'SwanLab 社区贡献指南',
    description: '加入 SwanLab 开源社区，了解如何贡献代码、文档和想法，共同推动机器学习工具的发展。',
    date: '2024-11-30',
    author: 'SwanLab Team',
    category: 'community',
    tags: ['开源', '贡献', '社区'],
    image: '/assets/blog/community-contribution.jpg',
    link: '/blog/community-contribution-guide'
  },
  {
    title: '使用 SwanLab 进行多模态实验跟踪',
    description: '探索如何使用 SwanLab 跟踪包含图像、文本、音频等多种数据类型的复杂机器学习实验。',
    date: '2024-11-25',
    author: '陈静',
    category: 'how-to',
    tags: ['多模态', '实验跟踪', '教程'],
    image: '/assets/blog/multimodal-tracking.jpg',
    link: '/blog/multimodal-experiment-tracking'
  }
]

export const categories = {
  all: '全部',
  release: '发布',
  product: '产品',
  'how-to': '教程',
  developer: '开发者',
  event: '活动',
  company: '公司',
  community: '社区'
} 