import { defineConfig, type DefaultTheme} from 'vitepress'

var base_path_guide_cloud = '/en/guide_cloud'
var base_path_examples = '/en/examples'
var base_path_api = '/en/api'

// https://vitepress.dev/reference/site-config
export const en = defineConfig({
  title: "SwanLab Docs",
  description: "SwanLab Official Documentation, providing the most comprehensive user guide and API documentation",
  lang: 'en-US',

  head: [
    ['link', { rel: 'icon', type:"image/svg+xml", href: '/icon.svg' }],
    ['link', { rel: 'icon', type:"image/png", href: '/icon.png' }],
  ],

  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    // 左上角logo
    logo: '/icon.svg',

    // 导航栏配置
    nav: [
      { 
        text: 'User Guide',
        link: base_path_guide_cloud + '/general/what-is-swanlab',
        activeMatch: '/en/guide_cloud/',
      },
      { 
        text: 'Example',
        link: base_path_examples + '/mnist',
        activeMatch: '/en/examples/',
      },
      { 
        text: 'API',
        link: base_path_api + '/api-index',
        activeMatch: '/en/api/',
        },
      { text: 'v0.3.27',  items: [
        { text: 'changelog', link: base_path_guide_cloud + '/general/changelog' },
        { text: 'Contribution', link: 'https://github.com/SwanHubX/SwanLab/blob/main/CONTRIBUTING.md' },
        { text: 'Feedback', link: 'https://geektechstudio.feishu.cn/share/base/form/shrcn8koDFRcH2mMcBYMh9tiKfI'}
      ] },
    //   { text: '在线交流', link: '/en/guide_cloud/community/online-support'},
      { text: 'Website', link: 'https://swanlab.cn' },
    ],

    // 最后更新于配置
    lastUpdated: {
      text: 'Updated at',
      formatOptions: {
        dateStyle: 'full',
        timeStyle: 'medium'
      },
    },

    // 编辑此页配置
    editLink: {
      pattern: 'https://github.com/swanhubx/swanlab-docs/edit/main/:path',
      text: 'Edit this page on GitHub'
    },

    // 侧边栏配置
    outline: {
      label: 'On this page',
      level: 'deep',
    },

    // 侧边栏配置
    sidebar: {
      '/en/guide_cloud/':{base: '/en/guide_cloud/', items: sidebarGuideCloud(),},
      '/en/examples/':{base: '/en/examples/', items: sidebarExamples(),},
      '/en/api/':{base: '/en/api/', items: sidebarAPI(),},
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
    text: 'Introduction',
    // collapsed: false,
    items: [
      { text: 'What is SwanLab', link: 'general/what-is-swanlab' },
      { text: 'Quick Start', link: 'general/quick-start' },
      { text: 'Team Usage', link: 'general/organization' },
      { text: 'Changelog', link: 'general/changelog' }
    ]
  },
  {
    text: '📚 Experiment Track',
    // collapsed: false,
    items: [
      { text: 'What is experiment tracking?', link: 'experiment_track/what-is-experiment-track' },
      { text: 'Create an experiment', link: 'experiment_track/create-experiment' },
      { text: 'Create by config file', link: 'experiment_track/create-experiment-by-configfile' },
      { text: 'Set config', link: 'experiment_track/set-experiment-config' },
      { text: 'Log metric', link: 'experiment_track/log-experiment-metric' },
      { text: 'Log media metric', link: 'experiment_track/log-media' },
      { text: 'View result', link: 'experiment_track/view-result' },
      { text: 'Finish experiment', link: 'experiment_track/finish-experiment' },
      { text: 'Jupyter Notebook', link: 'experiment_track/jupyter-notebook' },
      { text: 'Limitations and Performance', link: 'experiment_track/limit-and-performance' },
      { text: 'Experiment metadata', link: 'experiment_track/experiment-metadata' },
      { text: 'FAQ', link: 'experiment_track/FAQ' },
    ]
  },
  {
    text: '💻 Self-hosted',
    // collapsed: false,
    items: [
      { text: 'Offline board', link: 'self_host/offline-board' },
      { text: 'Remote access tutorial', link: 'self_host/remote-view' },
    ]
  },
  {
    text: '⚡️ Integration',
    // collapsed: false,
    items: [
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
    text: '👥 Community',
    // collapsed: false,
    items: [
      { text: 'Online support', link: 'community/online-support'},
      { text: 'Github badge', link: 'community/github-badge'},
      { text: 'Paper citation', link: 'community/paper-cite'},
      { text: 'Contributing code', link: 'community/contributing-code'},
      { text: 'Contributing docs', link: 'community/contributing-docs'},
      { text: 'Contributor', link: 'community/contributor'},
      { text: 'About us', link: 'community/emotion-machine'},
    ]
  },]
}



function sidebarExamples(): DefaultTheme.SidebarItem[] {
  return [{
    text: 'Quick Start',
    // collapsed: false,
    items: [
      { text: 'Hello_World', link: 'hello_world' },
      { text: 'MNIST Handwriting Recognition', link: 'mnist' },
    ]
  },
  {
    text: 'Computer Vision',
    // collapsed: false,
    items: [
      { text: 'FashionMNIST', link: 'fashionmnist' },
      { text: 'Cats and Dogs Classification', link: 'cats_dogs_classification' },    
      { text: 'Yolo Object Detection', link: 'yolo' },  
    ]
  },
  {
    text: 'NLP',
    // collapsed: false,
    items: [  
      { text: 'BERT Text Classification', link: 'bert' },  
      { text: 'Qwen Finetune Case', link: 'qwen_finetune' },  
      { text: 'LLM Pretraining', link: 'pretrain_llm' },  
    ]
  },
  {
    text: 'Audio',
    // collapsed: false,
    items: [
      { text: 'Audio Classification', link: 'audio_classification' },
    ]
  },
  {
    text: 'Time Series',
    // collapsed: false,
    items: [
      { text: 'LSTM Stock Prediction', link: 'lstm_stock'},
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
      { text: '(Beta) swanlab remote gpu', link: 'cli-swanlab-remote-gpu' },
      { text: 'Other', link: 'cli-swanlab-other' },
    ]
  },
  {
    text: 'Python SDK',
    // collapsed: false,
    items: [
      { text: 'init', link: 'py-init' },
      { text: 'log', link: 'py-log' },
      { text: 'Media data', items: [
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