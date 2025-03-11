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
    logo: {
      light: '/icon_docs.svg',
      dark: '/icon_docs_dark.svg'
    },
    siteTitle: false,

    // 导航栏配置
    nav: [
      { 
        text: 'User Guide',
        link: base_path_guide_cloud + '/general/what-is-swanlab',
      },
      {
        text: 'Integration',
        link: base_path_guide_cloud + '/integration/integration-huggingface-transformers',
        activeMatch: '/en/guide_cloud/integration/',
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
      { text: 'Contact Us', link: '/en/guide_cloud/community/online-support'},
      { text: 'v0.4.12',  items: [
        { text: 'changelog', link: base_path_guide_cloud + '/general/changelog' },
        { text: 'Feedback', link: 'https://geektechstudio.feishu.cn/share/base/form/shrcn8koDFRcH2mMcBYMh9tiKfI'},
        { text: 'Contribute Docs', link: 'https://github.com/SwanHubX/SwanLab-Docs' },
      ] },
      {
        component: 'HeaderButtonEN',
      },
      {
        component: 'HeaderGithubButton',
      }
      // { text: 'Website', link: 'https://swanlab.cn' },
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
      '/en/guide_cloud/integration/':{base: '/en/guide_cloud/integration/', items: sidebarIntegration(),},
      '/en/examples/':{base: '/en/examples/', items: sidebarExamples(),},
      '/en/api/':{base: '/en/api/', items: sidebarAPI(),},
    },

    // 页脚配置
    socialLinks: [
      { icon: 'bilibili', link: 'https://space.bilibili.com/386591517' },
      { icon: 'wechat', link: '/en/guide_cloud/community/online-support.html' },
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
      { text: 'Changelog', link: 'general/changelog' },
      { text: '🔥 Mobile SwanLab', link: 'general/app' },
    ]
  },
  {
    text: '📚 Experiment Tracking',
    // collapsed: false,
    items: [
      { text: 'What is experiment tracking?', link: 'experiment_track/what-is-experiment-track' },
      { text: 'Create an experiment', link: 'experiment_track/create-experiment' },
      { text: 'Create by config file', link: 'experiment_track/create-experiment-by-configfile' },
      { text: 'Set config', link: 'experiment_track/set-experiment-config' },
      { text: 'Log metric', link: 'experiment_track/log-experiment-metric' },
      { text: 'Log media metric', link: 'experiment_track/log-media' },
      { text: 'System Hardware Monitoring', link: 'experiment_track/system-monitor' },
      { text: 'View result', link: 'experiment_track/view-result' },
      { text: 'Finish experiment', link: 'experiment_track/finish-experiment' },
      { text: 'Jupyter Notebook', link: 'experiment_track/jupyter-notebook' },
      { text: 'Limitations and Performance', link: 'experiment_track/limit-and-performance' },
      { text: 'Experiment metadata', link: 'experiment_track/experiment-metadata' },
      { text: 'FAQ', link: 'experiment_track/FAQ' },
    ]
  },
  {
    text: '🚀 Self-hosted',
    // collapsed: false,
    items: [
      { text: 'Docker deployment', link: 'self_host/docker-deploy' },
    ]
  },
  {
    text: '💻 Offline board',
    collapsed: true,
    items: [
      { text: 'Offline board', link: 'self_host/offline-board' },
      { text: 'Remote access tutorial', link: 'self_host/remote-view' },
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
      { text: 'About us', link: 'community/emotion-machine'},
    ]
  },]
}

function sidebarIntegration(): DefaultTheme.SidebarItem[] {
  return [
  { text: 'Add SwanLab into Your Lib', link: 'integration-any-library' },
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
    text: 'Quick Start',
    // collapsed: false,
    items: [
      { text: 'Hello_World', link: 'hello_world' },
      { text: 'MNIST Handwriting Recognition', link: 'mnist' },
      { text: 'FashionMNIST', link: 'fashionmnist' },
    ]
  },
  {
    text: 'Computer Vision',
    // collapsed: false,
    items: [
      { text: 'Cats and Dogs Classification', link: 'cats_dogs_classification' },    
      { text: 'Yolo Object Detection', link: 'yolo' },  
      { text: 'QwenVL Finetune', link: 'qwen_vl_coco' },
      { text: 'Stable Diffusion Finetune', link: 'stable_diffusion' },
    ]
  },
  {
    text: 'Reinforcement Learning',
    // collapsed: false,
    items: [
      { text: 'DQN CartPole', link: 'dqn_cartpole' },
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
      // { text: '(Beta) swanlab remote gpu', link: 'cli-swanlab-remote-gpu' },
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
      { text: 'run', link: 'py-run' },
      { text: 'login', link: 'py-login' },
      { text: 'integration', link: 'py-integration' },
      { text: 'converter', link: 'py-converter' },
      { text: 'sync_wandb', link: 'py-sync-wandb' },
      { text: 'sync_tensorboard', link: 'py-sync-tensorboard' },
      { text: 'register_callback', link: 'py-register-callback' },
      { text: 'Other', link: 'py-other' },
    ]
  },
  {
    text: 'Other',
    // collapsed: false,
    items: [
      { text: 'Environment Variables', link: 'environment-variable' },
    ]
  }
]
}
