import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "My Awesome Project",
  description: "A VitePress Site",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Examples', link: '/changelog' }
    ],

    sidebar: [
      {
        text: '简介',
        items: [
          { text: '快速开始', link: '/quick-start' },
          { text: '更新日志', link: '/changelog' }
        ]
      },
      {
        text: 'AI实验跟踪',
        items: [
          { text: '创建一个实验', link: '/create-experiment' },
          { text: '用配置文件创建实验', link: '/create-experiment-by-configfile' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/vuejs/vitepress' }
    ],

    search: {
      provider: 'local'
    }
  }
})
