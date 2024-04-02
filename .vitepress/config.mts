import { defineConfig, type DefaultTheme} from 'vitepress'

var base_path_guide_cloud = '/zh/guide_cloud'
var base_path_guide_local = '/zh/guide_local'
var base_path_api = '/zh/api'


// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "SwanLab官方文档",
  description: "SwanLab官方文档, 提供最全面的使用指南和API文档",
  lang: 'zh-CN',
  head: [['link', { rel: 'icon', href: '/assets/icon.svg' }]],

  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    // 顶部logo
    logo: '/assets/icon.svg',

    // 导航栏配置
    nav: [
      { text: 'SwanLab', link: 'https://swanlab.pro' },
      { text: '用户指南/云端', link: base_path_guide_cloud + '/changelog' },
      { text: '用户指南/本地', link: base_path_guide_local +  '/changelog' },
      { text: 'API文档', link: base_path_api + '/changelog' },
      { text: 'v0.2.4',  items: [
        { text: '更新日志', link: base_path_guide_cloud + '/changelog' },
        { text: '参与贡献', link: '/' },
      ] },
    ],

    // 最后更新于配置
    lastUpdated: {
      text: '更新于',
      formatOptions: {
        dateStyle: 'full',
        timeStyle: 'medium'
      },
    },

    // 编辑此页配置
    editLink: {
      pattern: 'https://github.com/swanhubx/swanlab-docs/edit/main/:path',
      text: '在GitHub编辑此页面'
    },

    // 侧边栏配置
    outline: {
      label: '本页目录',
      level: 'deep',
    },

    // 侧边栏配置
    sidebar: {
      '/zh/guide_cloud/':{base: '/zh/guide_cloud/', items: sidebarGuideCloud(),},
      '/zh/guide_local/':{base: '/zh/guide_local/', items: sidebarGuideLocal(),},
      '/zh/api/':{base: '/zh/api/', items: sidebarAPI(),},
    },

    // 页脚配置
    socialLinks: [
      { icon: 'github', link: 'https://github.com/swanhubx/swanlab' }
    ],

    // 搜索配置
    search: {
      provider: 'local'
    }
  }
})


function sidebarGuideCloud(): DefaultTheme.SidebarItem[] {
  return [{
    text: '简介',
    // collapsed: false,
    items: [
      { text: '什么是SwanLab？', link: '/what-is-swanlab' },
      { text: '快速开始', link: '/quick-start' },
      { text: '更新日志', link: '/changelog' }
    ]
  },
  {
    text: 'AI实验跟踪',
    // collapsed: false,
    items: [
      { text: '什么是AI实验跟踪？', link: '/what-is-experiment-track' },
      { text: '创建一个实验', link: '/create-experiment' },
      { text: '用配置文件创建实验', link: '/create-experiment-by-configfile' },
      { text: '设置实验配置', link: '/create-experiment' },
      { text: '记录实验指标', link: '/create-experiment' },
      { text: '记录多媒体数据', link: '/create-experiment' },
      { text: '查看实验结果', link: '/create-experiment' },
      { text: '结束一个实验', link: '/create-experiment' },
      { text: '限制与性能', link: '/create-experiment' },
      { text: 'FAQ', link: '/create-experiment' },
    ]
  },
  {
    text: '集成',
    // collapsed: false,
    items: [
      { text: '添加SwanLab到任何库', link: '/create-experiment' },
      { text: 'Hydra', link: '/create-experiment' },
      { text: 'PyTorch', link: '/create-experiment-by-configfile' },
    ]
  },
  {
    text: '关于我们',
    // collapsed: false,
    items: [
      { text: '在线支持', link: '/create-experiment' },
      { text: 'SwanLab团队', link: '/create-experiment' },
    ]
  },]
}

function sidebarGuideLocal(): DefaultTheme.SidebarItem[] {
  return [{
    text: '简介',
    // collapsed: false,
    items: [
      { text: '什么是SwanLab？', link: '/what-is-swanlab' },
      { text: '快速开始', link: '/quick-start' },
      { text: '更新日志', link: '/changelog' }
    ]
  },
  {
    text: 'AI实验跟踪',
    // collapsed: false,
    items: [
      { text: '什么是AI实验跟踪？', link: '/what-is-experiment-track' },
      { text: '创建一个实验', link: '/create-experiment' },
      { text: '用配置文件创建实验', link: '/create-experiment-by-configfile' },
      { text: '设置实验配置', link: '/create-experiment' },
      { text: '记录实验指标', link: '/create-experiment' },
      { text: '记录多媒体数据', link: '/create-experiment' },
      { text: '查看实验结果', link: '/create-experiment' },
      { text: '结束一个实验', link: '/create-experiment' },
      { text: '限制与性能', link: '/create-experiment' },
      { text: 'FAQ', link: '/create-experiment' },
    ]
  },
  {
    text: '集成',
    // collapsed: false,
    items: [
      { text: '添加SwanLab到任何库', link: '/create-experiment' },
      { text: 'Hydra', link: '/create-experiment' },
      { text: 'PyTorch', link: '/create-experiment-by-configfile' },
    ]
  },
  {
    text: '关于我们',
    // collapsed: false,
    items: [
      { text: '在线支持', link: '/create-experiment' },
      { text: 'SwanLab团队', link: '/create-experiment' },
    ]
  },]
}

function sidebarAPI(): DefaultTheme.SidebarItem[] {
  return [{
    text: '简介',
    // collapsed: false,
    items: [
      { text: '什么是SwanLab？', link: '/what-is-swanlab' },
      { text: '快速开始', link: '/quick-start' },
      { text: '更新日志', link: '/changelog' }
    ]
  },
  {
    text: 'AI实验跟踪',
    // collapsed: false,
    items: [
      { text: '什么是AI实验跟踪？', link: '/what-is-experiment-track' },
      { text: '创建一个实验', link: '/create-experiment' },
      { text: '用配置文件创建实验', link: '/create-experiment-by-configfile' },
      { text: '设置实验配置', link: '/create-experiment' },
      { text: '记录实验指标', link: '/create-experiment' },
      { text: '记录多媒体数据', link: '/create-experiment' },
      { text: '查看实验结果', link: '/create-experiment' },
      { text: '结束一个实验', link: '/create-experiment' },
      { text: '限制与性能', link: '/create-experiment' },
      { text: 'FAQ', link: '/create-experiment' },
    ]
  },
  {
    text: '集成',
    // collapsed: false,
    items: [
      { text: '添加SwanLab到任何库', link: '/create-experiment' },
      { text: 'Hydra', link: '/create-experiment' },
      { text: 'PyTorch', link: '/create-experiment-by-configfile' },
    ]
  },
  {
    text: '关于我们',
    // collapsed: false,
    items: [
      { text: '在线支持', link: '/create-experiment' },
      { text: 'SwanLab团队', link: '/create-experiment' },
    ]
  },]
}