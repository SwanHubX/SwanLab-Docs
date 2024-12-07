import { defineConfig, type DefaultTheme} from 'vitepress'

var base_path_guide_cloud = '/guide_cloud'
var base_path_examples = '/examples'
var base_path_api = '/api'

// https://vitepress.dev/reference/site-config
export const zh = defineConfig({
  title: "SwanLab官方文档",
  description: "SwanLab官方文档, 提供最全面的使用指南和API文档",
  lang: 'zh-CN',

  head: [
    ['link', { rel: 'icon', type:"image/svg+xml", href: '/icon.svg' }],
    ['link', { rel: 'icon', type:"image/png", href: '/icon.png' }],
  ],

  // markdown: {
  //   lineNumbers: true
  // },

  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    // 左上角logo
    logo: '/icon.svg',

    // 导航栏配置
    nav: [
      { 
        text: '用户指南',
        link: base_path_guide_cloud + '/general/what-is-swanlab',
        activeMatch: '/guide_cloud/',
      },
      { 
        text: '案例',
        link: base_path_examples + '/mnist',
        activeMatch: '/examples/',
      },
      { 
        text: 'API',
        link: base_path_api + '/api-index',
        activeMatch: '/api/',
        },
      { text: 'v0.3.28',  items: [
        { text: '更新日志', link: base_path_guide_cloud + '/general/changelog' },
        { text: '参与贡献', link: 'https://github.com/SwanHubX/SwanLab/blob/main/CONTRIBUTING.md' },
        { text: '建议反馈', link: 'https://geektechstudio.feishu.cn/share/base/form/shrcn8koDFRcH2mMcBYMh9tiKfI'}
      ] },
      { text: '在线交流', link: '/guide_cloud/community/online-support'},
      { text: '官网', link: 'https://swanlab.cn' },
    ],

    // 最后更新于配置
    lastUpdated: {
      text: '更新于',
      formatOptions: {
        dateStyle: 'full',
        timeStyle: 'medium'
      },
    },

    langMenuLabel: '多语言',

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
      '/guide_cloud/':{base: '/guide_cloud/', items: sidebarGuideCloud(),},
      '/examples/':{base: '/examples/', items: sidebarExamples(),},
      '/api/':{base: '/api/', items: sidebarAPI(),},
    },

    // 页脚配置
    socialLinks: [
      { icon: 'github', link: 'https://github.com/swanhubx/swanlab' }
    ],
  }
})


function sidebarGuideCloud(): DefaultTheme.SidebarItem[] {
  return [{
    text: '简介',
    // collapsed: false,
    items: [
      { text: '欢迎使用SwanLab', link: 'general/what-is-swanlab' },
      { text: '快速开始', link: 'general/quick-start' },
      { text: '团队使用', link: 'general/organization' },
      { text: '更新日志', link: 'general/changelog' }
    ]
  },
  {
    text: '📚 实验跟踪',
    // collapsed: false,
    items: [
      { text: '什么是实验跟踪', link: 'experiment_track/what-is-experiment-track' },
      { text: '创建一个实验', link: 'experiment_track/create-experiment' },
      { text: '用配置文件创建实验', link: 'experiment_track/create-experiment-by-configfile' },
      { text: '设置实验配置', link: 'experiment_track/set-experiment-config' },
      { text: '记录指标', link: 'experiment_track/log-experiment-metric' },
      { text: '记录多媒体数据', link: 'experiment_track/log-media' },
      { text: '查看实验结果', link: 'experiment_track/view-result' },
      { text: '结束一个实验', link: 'experiment_track/finish-experiment' },
      { text: '用Jupyter Notebook跟踪实验', link: 'experiment_track/jupyter-notebook' },
      { text: '限制与性能', link: 'experiment_track/limit-and-performance' },
      { text: '实验元数据', link: 'experiment_track/experiment-metadata' },
      { text: 'FAQ', link: 'experiment_track/FAQ' },
    ]
  },
  {
    text: '💻 自托管',
    // collapsed: false,
    items: [
      { text: '离线看板', link: 'self_host/offline-board' },
      { text: '远程访问教程', link: 'self_host/remote-view' },
    ]
  },
  {
    text: '⚡️ 集成',
    // collapsed: false,
    items: [
      { text: 'Ascend NPU & MindSpore', link: 'integration/integration-ascend.md' },
      { text: 'Argparse', link:'integration/integration-argparse' },
      { text: 'Fastai', link: 'integration/integration-fastai' },
      { text: 'HuggingFace Accelerate', link: 'integration/integration-huggingface-accelerate' },
      { text: 'HuggingFace Transformers', link: 'integration/integration-huggingface-transformers' },
      { text: 'Hydra', link: 'integration/integration-hydra' },
      { text: 'Keras', link: 'integration/integration-keras' },
      { text: 'MMEngine', link: 'integration/integration-mmengine' },
      { text: 'MMPretrain', link: 'integration/integration-mmpretrain' },
      { text: 'MMDetection', link: 'integration/integration-mmdetection' },
      { text: 'MMSegmentation', link: 'integration/integration-mmsegmentation' },
      { text: 'OpenAI', link: 'integration/integration-openai' },
      { text: 'Omegaconf', link: 'integration/integration-omegaconf' },
      { text: 'PaddleDetection', link: 'integration/integration-paddledetection' },
      { text: 'PaddleYOLO', link: 'integration/integration-paddleyolo' },
      { text: 'PyTorch', link: 'integration/integration-pytorch' },
      { text: 'PyTorch Lightning', link: 'integration/integration-pytorch-lightning' },
      { text: 'PyTorch torchtune', link: 'integration/integration-pytorch-torchtune' },
      { text: 'Sentence Transformers', link: 'integration/integration-sentence-transformers'},
      { text: 'Stable Baseline3', link: 'integration/integration-sb3' },
      { text: 'Swift', link: 'integration/integration-swift' },
      { text: 'Tensorboard', link: 'integration/integration-tensorboard'},
      { text: 'Ultralytics', link: 'integration/integration-ultralytics' },
      { text: 'Weights & Biases', link: 'integration/integration-wandb'},
      { text: 'Xtuner', link: 'integration/integration-xtuner'},
      { text: 'ZhipuAI', link: 'integration/integration-zhipuai'},
    ]
  },
  {
    text: '👥 社区',
    // collapsed: false,
    items: [
      { text: '在线支持', link: 'community/online-support'},
      { text: 'Github徽章', link: 'community/github-badge'},
      { text: '论文引用', link: 'community/paper-cite'},
      { text: '贡献代码', link: 'community/contributing-code'},
      { text: '贡献官方文档', link: 'community/contributing-docs'},
      { text: '贡献者', link: 'community/contributor'},
      { text: '关于我们', link: 'community/emotion-machine'},
    ]
  },]
}

function sidebarExamples(): DefaultTheme.SidebarItem[] {
  return [{
    text: '入门',
    // collapsed: false,
    items: [
      { text: 'Hello_World', link: 'hello_world' },
      { text: 'MNIST手写体识别', link: 'mnist' },
    ]
  },
  {
    text: '计算机视觉',
    // collapsed: false,
    items: [
      { text: 'FashionMNIST', link: 'fashionmnist' },
      { text: 'Resnet猫狗分类', link: 'cats_dogs_classification' },    
      { text: 'Yolo目标检测', link: 'yolo' },  
      { text: 'QwenVL多模态大模型微调', link: 'qwen_vl_coco'},
    ]
  },
  {
    text: '自然语言处理',
    // collapsed: false,
    items: [  
      { text: 'BERT文本分类', link: 'bert' },  
      { text: 'Qwen微调案例', link: 'qwen_finetune' },  
      { text: 'LLM预训练', link: 'pretrain_llm' },  
    ]
  },
  {
    text: '音频',
    // collapsed: false,
    items: [
      { text: '音频分类', link: 'audio_classification' },
    ]
  },
  {
    text: '时间序列',
    // collapsed: false,
    items: [
      { text: 'LSTM股票预测', link: 'lstm_stock'},
    ]
  },
]
}

function sidebarAPI(): DefaultTheme.SidebarItem[] {
  return [{
    text: 'CLI',
    // collapsed: false,
    items: [
      { text: 'swanlab watch', link: 'cli-swanlab-watch' },
      { text: 'swanlab login', link: 'cli-swanlab-login' },
      { text: 'swanlab logout', link: 'cli-swanlab-logout' },
      { text: 'swanlab convert', link: 'cli-swanlab-convert' },
      { text: '(内测中) swanlab remote gpu', link: 'cli-swanlab-remote-gpu' },
      { text: '其他', link: 'cli-swanlab-other' },
    ]
  },
  {
    text: 'Python SDK',
    // collapsed: false,
    items: [
      { text: 'init', link: 'py-init' },
      { text: 'log', link: 'py-log' },
      { text: '多媒体数据', items: [
        { text: 'Image', link: 'py-Image' },
        { text: 'Audio', link: 'py-Audio' },
        { text: 'Text', link: 'py-Text' },
      ]},
      { text: 'login', link: 'py-login' },
      { text: 'integration', link: 'py-integration' },
      { text: 'converter', link: 'py-converter' },
    ]
  },]
}