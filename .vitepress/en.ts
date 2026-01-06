import { defineConfig, type DefaultTheme} from 'vitepress'

type SidebarItemBadge = { badge?: { type?: string; text?: string } }
type SidebarItemEx = DefaultTheme.SidebarItem & SidebarItemBadge

var base_path_guide_cloud = '/en/guide_cloud'
var base_path_examples = '/en/examples'
var base_path_api = '/en/api'
var base_path_plugin = '/en/plugin'

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
        text: 'Guide',
        link: base_path_guide_cloud + '/general/what-is-swanlab',
      },
      {
        text: 'Integration',  items: [
          { text: 'All 40+ Frameworks', link: base_path_guide_cloud + '/integration'},
          { text: 'Transformers', link: base_path_guide_cloud + '/integration/integration-huggingface-transformers'},
          { text: 'Lightning', link: base_path_guide_cloud + '/integration/integration-pytorch-lightning'},
          { text: 'LLaMA Factory', link: base_path_guide_cloud + '/integration/integration-llama-factory'},
          { text: 'Swift', link: base_path_guide_cloud + '/integration/integration-swift'},
          { text: 'Ultralytics', link: base_path_guide_cloud + '/integration/integration-ultralytics'},
          { text: 'veRL', link: base_path_guide_cloud + '/integration/integration-verl'},
          { text: 'Sb3', link: base_path_guide_cloud + '/integration/integration-sb3'},
        ]
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
      { text: 'Plugin', link: base_path_plugin + '/plugin-index'},
      { text: 'v0.7.6',  items: [
        { text: 'Changelog', link: base_path_guide_cloud + '/general/changelog' },
        { text: 'Community', link: 'https://swanlab.cn/benchmarks' },
        { text: 'Join Us', link: 'https://rcnpx636fedp.feishu.cn/wiki/BxtVwAc0siV0xrkCbPTcldBEnNP' },
        { text: 'Feedback', link: 'https://geektechstudio.feishu.cn/share/base/form/shrcn8koDFRcH2mMcBYMh9tiKfI' },
        { text: 'Docs Github', link: 'https://github.com/SwanHubX/SwanLab-Docs' },
      ] },
      {
        component: 'HeaderDocHelperButtonEN',
      },
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
      '/en/plugin/':{base: '/en/plugin/', items: sidebarPlugin(),},
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


function sidebarGuideCloud(): SidebarItemEx[] {
  return [{
    text: 'Introduction',
    // collapsed: false,
    items: [
      { text: 'What is SwanLab', link: 'general/what-is-swanlab' },
      { text: 'Quick Start', link: 'general/quick-start' },
      { text: 'Team Usage', link: 'general/organization' },
      { text: 'Changelog', link: 'general/changelog' },
      { text: '🔥 Mobile SwanLab', link: 'general/app'},
    ]
  },
  {
    text: '📚 Experiment Tracking',
    // collapsed: false,
    items: [
      { text: 'What is experiment tracking?', link: 'experiment_track/what-is-experiment-track'},
      { text: 'Create an experiment', link: 'experiment_track/create-experiment' },
      { text: 'Set config', link: 'experiment_track/set-experiment-config' },
      { text: 'Log metric', link: 'experiment_track/log-experiment-metric' },
      { text: 'Log media metric', items:[
        { text: 'Log Media', link: 'experiment_track/log-media' },
        { text: 'Log Custom Chart', link: 'experiment_track/log-custom-chart' },
        { text: 'Log Custom 3D Chart', link: 'experiment_track/log-custom-3dchart'},
      ]},
      {
        text: 'Log compute metric',
        collapsed: true,
        items: [
          { text: 'Log PR Curve', link: 'experiment_track/compute_metric/log-pr-curve' },
          { text: 'Log ROC Curve', link: 'experiment_track/compute_metric/log-roc-curve' },
          { text: 'Log Confusion Matrix', link: 'experiment_track/compute_metric/log-confusion-matrix' },
        ]
      },
      { text: 'Smooth line plots', link: 'experiment_track/smooth-algorithms' },
      { text: 'System Hardware Monitoring', link: 'experiment_track/system-monitor' },
      { text: 'Set tag', link: 'experiment_track/set-experiment-tag' },
      { text: 'View result', link: 'experiment_track/view-result' },
      { text: 'Finish experiment', link: 'experiment_track/finish-experiment' },
      { text: 'Email Notifications', link: 'experiment_track/send-notification' },
      { text: 'Jupyter Notebook', link: 'experiment_track/jupyter-notebook' },
      { text: 'Limitations and Performance', link: 'experiment_track/limit-and-performance' },
      { text: 'FAQ', link: 'experiment_track/FAQ' },
    ]
  },
  { 
    text: 'Tips',
    items: [
      { text: 'Resume experiment', link: 'experiment_track/resume-experiment' },
      { text: 'Upload offline experiment data', link: 'experiment_track/sync-logfile' },
      { text: 'Add project collaborator', link: 'experiment_track/add-collaborator' },
      { text: 'Access SwanLab on internal computing nodes', link: 'experiment_track/ssh-portforwarding' },
      { text: 'Avoid API key conflicts', link: 'experiment_track/api-key-conflict' },
      { text: 'Use OpenAPI to get experiment data', link: 'experiment_track/use-openapi' },
      { text: 'Webhook setup', link: 'experiment_track/webhook-setup' },
      { text: 'Experiment metadata', link: 'experiment_track/experiment-metadata' },
    ]
  },
  {
    text: '🚀 Self-hosted',
    // collapsed: false,
    items: [
      { text: 'Kubernetes deployment', link: 'self_host/kubernetes-deploy' },
      { text: 'Docker deployment', link: 'self_host/docker-deploy' }, 
      { text: 'Migration from Docker to Kubernetes', link: 'self_host/migration-docker-kubernetes' },
      { text: 'Offline Deployment', link: 'self_host/offline-deployment' },
      { text: 'Team/Enterprise', link: 'self_host/enterprise-version' },
      { text: 'FAQ', link: 'self_host/faq' },
    ]
  },
  {
    text: '💻 Offline board',
    // collapsed: true,
    items: [
      { text: 'Offline board', link: 'self_host/offline-board' },
      { text: 'Remote access tutorial', link: 'self_host/remote-view' },
      { text: 'Offline board API', link: 'self_host/offline-board-api' },
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
      { text: 'Areal', link: 'integration-areal' },
      { text: 'Ascend NPU & MindSpore', link: 'integration-ascend' },
      { text: 'Catboost', link: 'integration-catboost'},
      { text: 'DiffSynth-Studio', link: 'integration-diffsynth-studio' },
      { text: 'EasyR1', link: 'integration-easyr1' },
      { text: 'EvalScope', link: 'integration-evalscope' },
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
      { text: 'MLX LM', link: 'integration-mlx-lm' },
      { text: 'MMEngine', link: 'integration-mmengine' },
      { text: 'MMPretrain', link: 'integration-mmpretrain' },
      { text: 'MMDetection', link: 'integration-mmdetection' },
      { text: 'MMSegmentation', link: 'integration-mmsegmentation' },
      { text: 'Modelscope Swift', link: 'integration-swift' },
      { text: 'NVIDIA-NeMo RL', link: 'integration-nvidia-nemo-rl' },
    ]
  },
  {
    text: 'O-T',
    // collapsed: false,
    items: [
      { text: 'OpenAI', link: 'integration-openai' },
      { text: 'Omegaconf', link: 'integration-omegaconf' },
      { text: 'PaddleDetection', link: 'integration-paddledetection' },
      { text: 'PaddleNLP', link: 'integration-paddlenlp' },
      { text: 'PaddleYOLO', link: 'integration-paddleyolo' },
      { text: 'PyTorch', link: 'integration-pytorch' },
      { text: 'PyTorch Lightning', link: 'integration-pytorch-lightning' },
      { text: 'PyTorch torchtune', link: 'integration-pytorch-torchtune' },
      { text: 'Ray', link: 'integration-ray' },
      { text: 'RLinf', link: 'integration-rlinf'},
      { text: 'ROLL', link: 'integration-roll' },
      { text: 'Sentence Transformers', link: 'integration-sentence-transformers'},
      { text: 'Specforge', link: 'integration-specforge'},
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
      { text: 'MNIST', link: 'mnist' },
      { text: 'FashionMNIST', link: 'fashionmnist' },
      { text: 'CIFAR10', link: 'cifar10' },
    ]
  },
  {
    text: 'Computer Vision',
    // collapsed: false,
    items: [
      { text: 'Cats and Dogs Classification', link: 'cats_dogs_classification' },    
      { text: 'Yolo Object Detection', link: 'yolo' },  
      { text: 'UNet Medical Image Segmentation', link: 'unet-medical-segmentation'},
      { text: 'QwenVL Finetune', link: 'qwen_vl_coco' },
      { text: 'Stable Diffusion Finetune', link: 'stable_diffusion' },
    ]
  },
  {
    text: 'NLP',
    // collapsed: false,
    items: [  
      { text: 'BERT Text Classification', link: 'bert' },  
      { text: 'LLM Pretraining', link: 'pretrain_llm' },  
      { text: 'GLM4 Instruct Finetune', link: 'glm4-instruct'},
      { text: 'Qwen2 NER', link: 'ner'},
      { text: 'Qwen Finetune Case', link: 'qwen_finetune' },
      { text: 'Qwen3 Medical', link: 'qwen3-medical' },
    ]
  },
  {
    text: 'Robot',
    items: [
      { text: 'LeRobot Guide', link: 'robot/lerobot-guide' },
    ]
  },
  {
    text: 'Reinforcement Learning',
    // collapsed: false,
    items: [
      { text: 'DQN CartPole', link: 'dqn_cartpole' },
      { text: 'GRPO LLM RL', link: 'qwen_grpo' },
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
      { text: 'RNN Fundamentals: Variable-Length Sequences', link: 'rnn_tutorial_1'},
      { text: 'RNN Sequence Prediction', link: 'rnn_tutorial_2'},
    ]
  },
  {
    text: 'Community Contributions',
    //collapsed: false,
    items: [
      { text: 'How to Choose Open Source License', link: 'how-to-choose-open-source-license' },
      { text: 'Swanlab-rag', link: 'agent/swanlab-rag' },
      { text: 'Vit-KNO Weather Forecasting', link: 'ViT-KNO' },
    ]
  }
]
}

function sidebarAPI(): DefaultTheme.SidebarItem[] {
  return [
  {
    text: 'Develop',
    // collapsed: false,
    items: [
      { text: 'OpenAPI', link: 'py-openapi' },
      { text: 'Environment Variables', link: 'environment-variable' },
    ]
  },
  {
    text: 'CLI',
    // collapsed: false,
    items: [
      { text: 'swanlab watch', link: 'cli-swanlab-watch' },
      { text: 'swanlab login', link: 'cli-swanlab-login' },
      { text: 'swanlab logout', link: 'cli-swanlab-logout' },
      { text: 'swanlab convert', link: 'cli-swanlab-convert' },
      { text: 'swanlab sync', link: 'cli-swanlab-sync' },
      { text: 'swanlab offline', link: 'cli-swanlab-offline' },
      { text: 'swanlab online', link: 'cli-swanlab-online' },
      { text: 'swanlab local', link: 'cli-swanlab-local' },
      { text: 'swanlab disabled', link: 'cli-swanlab-disabled' },
      { text: 'Other', link: 'cli-swanlab-other' },
    ]
  },
  {
    text: 'Python SDK',
    // collapsed: false,
    items: [
      { text: 'init', link: 'py-init' },
      { text: 'log', link: 'py-log' },
      { text: 'Settings', link: 'py-settings' },
      { text: 'Media data', items: [
        { text: 'Image', link: 'py-Image' },
        { text: 'Audio', link: 'py-Audio' },
        { text: 'Text', link: 'py-Text' },
        { text: 'Video', link: 'py-video' },
        { text: 'Echarts', link: 'py-echarts' },
        { text: 'Object3D', link: 'py-object3d' },
        { text: 'Molecule', link: 'py-molecule' },
      ]},
      {
        text: 'Metrics', items: [
          { text: 'pr_curve', link: 'py-pr_curve' },
          { text: 'roc_curve', link: 'py-roc_curve' },
          { text: 'confusion_matrix', link: 'py-confusion_matrix' },
        ]
      },
      { text: 'run', link: 'py-run' },
      { text: 'login', link: 'py-login' },
      { text: 'integration', link: 'py-integration' },
      { text: 'converter', link: 'py-converter' },
      { text: 'sync_wandb', link: 'py-sync-wandb' },
      { text: 'sync_tensorboard', link: 'py-sync-tensorboard' },
      { text: 'sync_mlflow', link: 'py-sync-mlflow' },
      { text: 'register_callback', link: 'py-register-callback' },
      { text: 'Other', link: 'py-other' },
    ]
  },
]
}

function sidebarPlugin(): DefaultTheme.SidebarItem[] {
  return [
  {
    text: '🔧 Make your custom plugin',
    link: 'custom-plugin',
  },
  {
    text: '✈️ Notification',
    // collapsed: false,
    items: [
      { text: 'Email', link: 'notification-email' },
      { text: 'Lark', link: 'notification-lark' },
      { text: 'Dingtalk', link: 'notification-dingtalk' },
      { text: 'WXWork', link: 'notification-wxwork' },
      { text: 'Discord', link: 'notification-discord' },
      { text: 'Slack', link: 'notification-slack' },
      { text: 'Bark', link: 'notification-bark' },
      { text: 'Telegram', link: 'notification-telegram' },
    ]
  },
  {
    text: '📝 Writer',
    // collapsed: false,
    items: [
      { text: 'File Logger', link: 'writer-filelogdir' },
      { text: 'CSV Table', link: 'writer-csv' },
    ]
  },
]
}