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
    // 左上角logo，支持明暗模式
    logo: {
      light: '/icon_docs.svg',
      dark: '/icon_docs_dark.svg'
    },
    siteTitle: false,

    // 导航栏配置
    nav: [
      { 
        text: '用户指南',
        link: base_path_guide_cloud + '/general/what-is-swanlab',
        // activeMatch: '/guide_cloud/',
      },
      {
        text: '框架集成',
        link: base_path_guide_cloud + '/integration/integration-huggingface-transformers',
        activeMatch: '/guide_cloud/integration/',
      },
      { 
        text: '实战案例',
        link: base_path_examples + '/mnist',
        activeMatch: '/examples/',
      },
      { 
        text: 'API文档',
        link: base_path_api + '/api-index',
        activeMatch: '/api/',
        },
      { text: 'v0.4.11',  items: [
        { text: '更新日志', link: base_path_guide_cloud + '/general/changelog' },
        { text: '建议反馈', link: 'https://geektechstudio.feishu.cn/share/base/form/shrcn8koDFRcH2mMcBYMh9tiKfI'},
        { text: '贡献文档', link: 'https://github.com/SwanHubX/SwanLab-Docs' },
      ]
      },
      {
        component: 'HeaderButton',
      },
      {
        component: 'HeaderGithubButton',
      }

    ],

    // 最后更新于配置
    lastUpdated: {
      text: '更新于',
      formatOptions: {
        dateStyle: 'full',
        timeStyle: 'medium'
      },
    },

    // 丰富中文化配置
    docFooter: {
      prev: '上一页',
      next: '下一页'
    },

    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '主题',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    skipToContentLabel: '跳转到内容',
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
      '/guide_cloud/integration/':{base: '/guide_cloud/integration/', items: sidebarIntegration(),},
      '/api/':{base: '/api/', items: sidebarAPI(),},
    },

    // 页脚配置
    socialLinks: [
      { icon: 'bilibili', link: 'https://space.bilibili.com/386591517' },
      { icon: 'wechat', link: '/guide_cloud/community/online-support.html' },
      // { icon: 'github', link: 'https://github.com/swanhubx/swanlab' },
    ],
  }
})


function sidebarGuideCloud(): DefaultTheme.SidebarItem[] {
  return [
    {
    text: '简介',
    // collapsed: false,
    items: [
      { text: '欢迎使用SwanLab', link: 'general/what-is-swanlab' },
      { text: '快速开始', link: 'general/quick-start' },
      { text: '团队使用', link: 'general/organization' },
      { text: '更新日志', link: 'general/changelog' },
      { text: '🔥手机看实验', link: 'general/app' },
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
      { text: '系统硬件监控（支持昇腾）', link: 'experiment_track/system-monitor' },
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
    text: '👥 社区',
    // collapsed: false,
    items: [
      { text: '在线支持', link: 'community/online-support'},
      { text: 'Github徽章', link: 'community/github-badge'},
      { text: '论文引用', link: 'community/paper-cite'},
      { text: '贡献代码', link: 'community/contributing-code'},
      { text: '贡献官方文档', link: 'community/contributing-docs'},
      { text: '关于我们', link: 'community/emotion-machine'},
    ]
  },]
}

function sidebarIntegration(): DefaultTheme.SidebarItem[] {
  return [
  { text: '将SwanLab集成到你的库', link: 'integration-any-library' },
  {
    text: 'A-G',
    // collapsed: false,
    items: [
      { text: 'Argparse', link:'integration-argparse' },
      { text: 'Ascend NPU & MindSpore', link: 'integration-ascend' },
      { text: 'DiffSynth-Studio', link: 'integration-diffsynth-studio' },
      { text: 'EasyR1', link: 'integration-easyr1' },
      { text: 'Fastai', link: 'integration-fastai' },
    ]
  },
  {
    text: 'H-N',
    // collapsed: false,
    items: [
      { text: 'HuggingFace Accelerate', link: 'integration-huggingface-accelerate' },
      { text: 'HuggingFace Transformers', link: 'integration-huggingface-transformers' },
      { text: 'HuggingFace Trl', link: 'integration-huggingface-trl' },
      { text: 'Hydra', link: 'integration-hydra' },
      { text: 'Keras', link: 'integration-keras' },
      { text: 'LightGBM', link: 'integration-lightgbm'},
      { text: 'LLaMA Factory', link: 'integration-llama-factory'},
      { text: 'MLFlow', link: 'integration-mlflow'},
      { text: 'MMEngine', link: 'integration-mmengine' },
      { text: 'MMPretrain', link: 'integration-mmpretrain' },
      { text: 'MMDetection', link: 'integration-mmdetection' },
      { text: 'MMSegmentation', link: 'integration-mmsegmentation' },
      { text: 'Modelscope Swift', link: 'integration-swift' },
    ]
  },
  {
    text: 'O-T',
    // collapsed: false,
    items: [
      { text: 'OpenAI', link: 'integration-openai' },
      { text: 'Omegaconf', link: 'integration-omegaconf' },
      { text: 'PaddleDetection', link: 'integration-paddledetection' },
      { text: 'PaddleYOLO', link: 'integration-paddleyolo' },
      { text: 'PyTorch', link: 'integration-pytorch' },
      { text: 'PyTorch Lightning', link: 'integration-pytorch-lightning' },
      { text: 'PyTorch torchtune', link: 'integration-pytorch-torchtune' },
      { text: 'Sentence Transformers', link: 'integration-sentence-transformers'},
      { text: 'Stable Baseline3', link: 'integration-sb3' },
      { text: 'Tensorboard', link: 'integration-tensorboard'},
    ]
  },
  {
    text: 'U-Z',
    // collapsed: false,
    items: [
      { text: 'Ultralytics', link: 'integration-ultralytics' },
      { text: 'Unsloth', link: 'integration-unsloth' },
      { text: 'Verl', link: 'integration-verl' },
      { text: 'Weights & Biases', link: 'integration-wandb'},
      { text: 'XGBoost', link: 'integration-xgboost'},
      { text: 'Xtuner', link: 'integration-xtuner'},
      { text: 'ZhipuAI', link: 'integration-zhipuai'},
    ]
  }]
}


function sidebarExamples(): DefaultTheme.SidebarItem[] {
  return [{
    text: '入门',
    // collapsed: false,
    items: [
      { text: 'Hello_World', link: 'hello_world' },
      { text: 'MNIST手写体识别', link: 'mnist' },
      { text: 'FashionMNIST', link: 'fashionmnist' },
    ]
  },
  {
    text: '计算机视觉',
    // collapsed: false,
    items: [
      { text: 'Resnet猫狗分类', link: 'cats_dogs_classification' },    
      { text: 'Yolo目标检测', link: 'yolo' },  
      { text: 'QwenVL多模态大模型微调', link: 'qwen_vl_coco'},
      { text: 'Stable Diffusion文生图微调', link: 'stable_diffusion'},
    ]
  },
  {
    text: '自然语言处理',
    // collapsed: false,
    items: [  
      { text: 'BERT文本分类', link: 'bert' },  
      { text: 'Qwen微调案例', link: 'qwen_finetune' },  
      { text: 'LLM预训练', link: 'pretrain_llm' },  
      { text: 'openMind大模型微调', link: 'openMind' },  
    ]
  },
  {
    text: '强化学习',
    // collapsed: false,
    items: [
      { text: 'DQN推车倒立摆', link: 'dqn_cartpole' },
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
      // { text: '(内测中) swanlab remote gpu', link: 'cli-swanlab-remote-gpu' },
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
      { text: 'run', link: 'py-run' },
      { text: 'login', link: 'py-login' },
      { text: 'integration', link: 'py-integration' },
      { text: 'converter', link: 'py-converter' },
      { text: 'sync_wandb', link: 'py-sync-wandb' },
      { text: 'sync_tensorboard', link: 'py-sync-tensorboard' },
      { text: '其他', link: 'py-other' },
    ]
  },
  {
    text: '其他',
    // collapsed: false,
    items: [
      { text: '环境变量', link: 'environment-variable' },
    ]
  }
]
}