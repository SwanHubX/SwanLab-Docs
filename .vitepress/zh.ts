import { defineConfig, type DefaultTheme } from 'vitepress'

var base_path_guide_cloud = '/guide_cloud'
var base_path_examples = '/examples'
var base_path_api = '/api'
var base_path_plugin = '/plugin'
var base_path_course = '/course'

// https://vitepress.dev/reference/site-config
export const zh = defineConfig({
  title: "SwanLab官方文档",
  description: "SwanLab官方文档, 提供最全面的使用指南和API文档",
  lang: 'zh-CN',

  head: [
    ['link', { rel: 'icon', type: "image/svg+xml", href: '/icon.svg' }],
    ['link', { rel: 'icon', type: "image/png", href: '/icon.png' }],
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
        text: '指南',
        link: base_path_guide_cloud + '/general/what-is-swanlab',
        // activeMatch: '/guide_cloud/',
      },
      {
        text: '案例',
        link: base_path_examples + '/mnist',
        activeMatch: '/examples/',
      },
      {
        text: '集成', items: [
          { text: '全部40+框架', link: base_path_guide_cloud + '/integration' },
          { text: 'Transformers', link: base_path_guide_cloud + '/integration/integration-huggingface-transformers' },
          { text: 'Lightning', link: base_path_guide_cloud + '/integration/integration-pytorch-lightning' },
          { text: 'LLaMA-Factory', link: base_path_guide_cloud + '/integration/integration-llama-factory' },
          { text: 'MS-Swift', link: base_path_guide_cloud + '/integration/integration-swift' },
          { text: 'veRL', link: base_path_guide_cloud + '/integration/integration-verl' },
          { text: 'Ultralytics', link: base_path_guide_cloud + '/integration/integration-ultralytics' },
          { text: 'Sb3', link: base_path_guide_cloud + '/integration/integration-sb3' },
        ]
      },
      {
        text: '课程',items:[
          {'text': '提示词工程', link: base_path_course + '/prompt_engineering_course/01-preface/README.md'},
          {'text': '大模型训练', link:base_path_course+'/llm_train_course/00-preface/README.md'},
        ],
      },
      {
        text: 'API',
        link: base_path_api + '/api-index',
        activeMatch: '/api/',
      },
      {
        text: '插件',
        link: base_path_plugin + '/plugin-index',
        activeMatch: '/plugin/',
      },
      { text: 'v0.7.12',  items: [
        { text: '更新日志', link: base_path_guide_cloud + '/general/changelog' },
        { text: '基线社区', link: 'https://swanlab.cn/benchmarks' },
        { text: '加入我们', link: 'https://rcnpx636fedp.feishu.cn/wiki/BxtVwAc0siV0xrkCbPTcldBEnNP' },
        { text: '建议反馈', link: 'https://geektechstudio.feishu.cn/share/base/form/shrcn8koDFRcH2mMcBYMh9tiKfI'},
        { text: '文档仓库', link: 'https://github.com/SwanHubX/SwanLab-Docs' },
        { text: '交流群', link: base_path_guide_cloud + '/community/online-support' },
      ]
      },
      {
        component: 'HeaderDocHelperButton',
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
      '/guide_cloud/': { base: '/guide_cloud/', items: sidebarGuideCloud(), },
      '/examples/': { base: '/examples/', items: sidebarExamples(), },
      '/guide_cloud/integration/': { base: '/guide_cloud/integration/', items: sidebarIntegration(), },
      '/api/': { base: '/api/', items: sidebarAPI(), },
      '/plugin/': { base: '/plugin/', items: sidebarPlugin(), },
      '/course/prompt_engineering_course/': { base: '/course/prompt_engineering_course/', items: sidebarCoursePromptEngineering(), },
      '/course/llm_train_course/': { base: '/course/llm_train_course/', items: sidebarCourseLLMTrain(), },
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
      ]
    },
  {
      text: '📚 实验记录与分析',
      // collapsed: false,
      items: [
        { text: '什么是实验记录', link: 'experiment_track/what-is-experiment-track'},
        { text: '创建一个实验', link: 'experiment_track/create-experiment' },
        { text: '设置实验配置', link: 'experiment_track/set-experiment-config' },
        { text: '记录指标', link: 'experiment_track/log-experiment-metric' },
        { text: '折线图平滑', link: 'experiment_track/smooth-algorithms' },
        {
          text: '记录多维数据', 
          collapsed: true,
          items: [
            { text: '媒体类型', link: 'experiment_track/log-media' },
            { text: '自定义图表', link: 'experiment_track/log-custom-chart' },
            { text: '自定义3D图表', link: 'experiment_track/log-custom-3dchart'},
          ]
        },
        {
          text: '记录计算指标', collapsed: true, items: [
            { text: 'PR曲线', link: 'experiment_track/compute_metric/log-pr-curve' },
            { text: 'ROC曲线', link: 'experiment_track/compute_metric/log-roc-curve' },
            { text: '混淆矩阵', link: 'experiment_track/compute_metric/log-confusion-matrix' },
          ]
        },
        { text: '记录分布式训练指标', link: 'experiment_track/log-distributed-training'},
        { text: '查看实验结果', link: 'experiment_track/view-result' },
        { text: '结束一个实验', link: 'experiment_track/finish-experiment' },
        { text: 'GPU监控', link: 'experiment_track/system-monitor' },
        { text: '邮件/第三方通知', link: 'experiment_track/send-notification' },
        { text: '常见问题', link: 'experiment_track/FAQ' },
      ]
    },
    {
      text: '🦄 实验管理',
      items: [
        { text: '实验标签', link: 'experiment_track/set-experiment-tag'},
        { text: '实验分组', link: 'experiment_track/grouping' },
      ]
    },
    { text: '📈 高级特性', 
      items: [
        { text: '恢复实验/断点续训', link: 'experiment_track/resume-experiment' },
        { text: '上传离线实验数据', link: 'experiment_track/sync-logfile' },
        { text: '添加项目协作者', link: 'experiment_track/add-collaborator' },
        { text: '使用API获取实验数据', link: 'experiment_track/use-openapi' },
      ]
    },
    {
      text: '🚄 技巧',
      items: [
        { text: '手机看实验', link: 'general/app'},
        { text: '内网节点访问SwanLab', link: 'experiment_track/ssh-portforwarding' },
        { text: '多人共用服务器避免密钥冲突', link: 'experiment_track/api-key-conflict' },
        { text: '更多技巧', 
          collapsed: true,
          items: [
          { text: 'Webhook设置', link: 'experiment_track/webhook-setup' },
          { text: '获取实验元数据', link: 'experiment_track/experiment-metadata' },
          { text: 'Notebook跟踪实验', link: 'experiment_track/jupyter-notebook' },
        ]},
    ] },
    {
      text: '🚀 自托管',
      // collapsed: false,
      items: [
        { text: 'Kubernetes部署（推荐）', link: 'self_host/kubernetes-deploy' },
        { text: "Docker部署", link: "self_host/docker-deploy" },
        { text: '从Docker迁移至K8S', link: 'self_host/migration-docker-kubernetes' },
        { text: "纯离线环境部署", link: "self_host/offline-deployment" },
        { text: '团队/企业版', link: 'self_host/enterprise-version' },
        {
          text: "第三方部署",
          collapsed: true,
          items: [
            { text: '阿里云计算巢', link: 'self_host/alibabacloud-computenest' },
            { text: '腾讯云云应用', link: 'self_host/tencentcloud-app' },
          ]
        },
        { text: '常见问题', link: 'self_host/faq' },
      ]
    },
    {
      text: '💻 离线看板',
      collapsed: true,
      items: [
        { text: '使用离线看板', link: 'self_host/offline-board' },
        { text: '远程访问离线看板', link: 'self_host/remote-view' },
        { text: '离线看板接口文档', link: 'self_host/offline-board-api' },
      ]
    },
    {
      text: '👥 社区',
      // collapsed: false,
      items: [
        { text: '在线支持', link: 'community/online-support' },
        { text: 'Github徽章', link: 'community/github-badge' },
        // { text: '论文引用', link: 'community/paper-cite'},
        // { text: '贡献代码', link: 'community/contributing-code'},
        // { text: '贡献官方文档', link: 'community/contributing-docs'},
        { text: '关于我们', link: 'community/emotion-machine' },
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
        { text: 'Argparse', link: 'integration-argparse' },
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
        { text: 'LightGBM', link: 'integration-lightgbm' },
        { text: 'LLaMA-Factory', link: 'integration-llama-factory' },
        { text: 'LLaMA-Factory Online', link: 'integration-llama-factory-online' },
        { text: 'MindSpeed-RL', link: 'integration-mindspeed-rl' },
        { text: 'MLFlow', link: 'integration-mlflow' },
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
        { text: 'RLinf', link: 'integration-rlinf' },
        { text: 'ROLL', link: 'integration-roll' },
        { text: 'Sentence Transformers', link: 'integration-sentence-transformers' },
        { text: 'Specforge', link: 'integration-specforge' },
        { text: 'Stable Baseline3', link: 'integration-sb3' },
        { text: 'Tensorboard', link: 'integration-tensorboard' },
      ]
    },
    {
      text: 'U-Z',
      // collapsed: false,
      items: [
        { text: 'Ultralytics', link: 'integration-ultralytics' },
        { text: 'Unsloth', link: 'integration-unsloth' },
        { text: 'Verl', link: 'integration-verl' },
        { text: 'Weights & Biases', link: 'integration-wandb' },
        { text: 'XGBoost', link: 'integration-xgboost' },
        { text: 'Xtuner', link: 'integration-xtuner' },
        { text: 'ZhipuAI', link: 'integration-zhipuai' },
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
      { text: 'Cifar10图像分类', link: 'cifar10' },
    ]
  },
  {
    text: '计算机视觉',
    // collapsed: false,
    items: [
      { text: 'Resnet猫狗分类', link: 'cats_dogs_classification' },
      { text: 'Yolo目标检测', link: 'yolo' },
      { text: 'UNet医学影像分割', link: 'unet-medical-segmentation' },
      { text: 'QwenVL多模态大模型微调', link: 'qwen_vl_coco' },
      { text: 'Qwen3-smVL模型拼接微调', link: 'qwen3_smolvlm_muxi' },
      { text: 'Stable Diffusion文生图微调', link: 'stable_diffusion' },
    ]
  },
  {
    text: '自然语言处理',
    // collapsed: false,
    items: [
      { text: 'BERT文本分类', link: 'bert' },
      { text: 'LLM预训练', link: 'pretrain_llm' },
      { text: 'GLM4指令微调', link: 'glm4-instruct' },
      { text: 'Qwen下游任务训练', link: 'qwen_finetune' },
      { text: 'NER命名实体识别', link: 'ner' },
      { text: 'Qwen3医学模型微调', link: 'qwen3-medical' },
      { text: '通过微调给Qwen3起新名字', link: 'mlx_lm_finetune' },
      { text: '扩散语言模型', link: 'diffusion_language_model' },
    ]
  },
  {
    text: '机器人',
    items: [
      { text: 'LeRobot 入门', link: 'robot/lerobot-guide' },
    ]
  },
  {
    text: '强化学习',
    // collapsed: false,
    items: [
      { text: 'DQN推车倒立摆', link: 'dqn_cartpole' },
      { text: 'GRPO大模型强化学习', link: 'qwen_grpo' },
      { text: '数独游戏GRPO训练', link: 'sudoku_grpo' },
    ]
  },
  {
    text: '音频',
    // collapsed: false,
    items: [
      { text: '音频分类', link: 'audio_classification' },
      { text: '原神派蒙语音克隆', link: 'genshin-paimon-cosyvoice-sft' },
    ]
  },
  {
    text: '时间序列',
    // collapsed: false,
    items: [
      { text: 'LSTM股票预测', link: 'lstm_stock' },
      { text: 'RNN教程', items: [
        { text: '原理简介', link: 'rnn_tutorial_1' },
        { text: '序列预测模型构建', link: 'rnn_tutorial_2' },
      ]},
    ]
  },
  {
    text: '社区供稿',
    collapsed: false,
    items: [
      { text: '如何为你的大模型选择开源许可证', link: 'how-to-choose-open-source-license' },
      { text: 'openMind大模型微调', link: 'openMind' },
      { text: 'SwanLab RAG文档助手', link: 'agent/swanlab-rag' },
      { text: 'PaddleNLP大模型微调实战', link: 'paddlenlp_finetune' },
      { text: 'Vit-KNO 气象预测', link: 'ViT-KNO' },
    ]
  }
  ]
}

function sidebarAPI(): DefaultTheme.SidebarItem[] {
  return [
  {
      text: '开发',
      // collapsed: false,
      items: [
        { text: '开放接口', link: 'py-api' },
        { text: '开放接口（旧版）', link: 'py-openapi' },
        { text: '环境变量', link: 'environment-variable' },
      ]
  },  
  {
    text: '命令行',
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
      { text: '其他', link: 'cli-swanlab-other' },
    ]
  },
  {
    text: 'Python SDK',
    // collapsed: false,
    items: [
      { text: 'init', link: 'py-init' },
      { text: 'log', link: 'py-log' },
      { text: 'Settings', link: 'py-settings' },
      {
        text: '多媒体数据', items: [
          { text: 'Image', link: 'py-Image' },
          { text: 'Audio', link: 'py-Audio' },
          { text: 'Text', link: 'py-Text' },
          { text: 'Video', link: 'py-video' },
          { text: 'ECharts', link: 'py-echarts' },
          { text: 'Object3D', link: 'py-object3d' },
          { text: 'Molecule', link: 'py-molecule' },
        ]
      },
      {
        text: '指标数据', items: [
          { text: 'pr_curve', link: 'py-pr_curve' },
          { text: 'roc_curve', link: 'py-roc_curve' },
          { text: 'confusion_matrix', link: 'py-confusion_matrix' },
        ]
      },
      { text: 'run', link: 'py-run' },
      { text: 'login', link: 'py-login' },
      { text: 'integration', link: 'py-integration' },
      { text: 'converter', link: 'py-converter' },
      {
        text: '同步其他工具',
        items: [
          { text: 'sync_wandb', link: 'py-sync-wandb' },
          { text: 'sync_tensorboard', link: 'py-sync-tensorboard' },
          { text: 'sync_mlflow', link: 'py-sync-mlflow' },
        ]
      },
      { text: 'register_callback', link: 'py-register-callback' },
      { text: '其他', link: 'py-other' },
    ]
  },
  ]
}

function sidebarPlugin(): DefaultTheme.SidebarItem[] {
  return [
    {
      text: '🔧 制作自定义插件',
      link: 'custom-plugin',
    },
    {
      text: '✈️ 通知类',
      // collapsed: false,
      items: [
        { text: '邮件', link: 'notification-email' },
        { text: '飞书', link: 'notification-lark' },
        { text: '钉钉', link: 'notification-dingtalk' },
        { text: '企业微信', link: 'notification-wxwork' },
        { text: 'Discord', link: 'notification-discord' },
        { text: 'Slack', link: 'notification-slack' },
        { text: 'Bark', link: 'notification-bark' },
        { text: 'Telegram', link: 'notification-telegram' },
      ]
    },
    {
      text: '📝 记录类',
      // collapsed: false,
      items: [
        { text: '文件记录器', link: 'writer-filelogdir' },
        { text: 'CSV表格', link: 'writer-csv' },
      ]
    },
  ]
}

function sidebarCoursePromptEngineering(): DefaultTheme.SidebarItem[] {
  return [
    {
      text: '📕提示词工程课程',
      items: [
        { text: '前言', link: '01-preface/README.md' },
        { text: '指南', link: '02-prompt_guide/README.md' },
        { text: '第一章 环境安装', link: '03-environmental_installation_platform_preparation/README.md' },
        { text: '第二章 模型选择', link: '04-model_types/README.md' },
        { text: '第三章 提示词撰写技巧', link: '05-tips_for_prompt/README.md', items: [
          { text: '3.1 模型参数设置', link: '05-tips_for_prompt/1.model_parameter_settings.md' },
          { text: '3.2 提示词结构', link: '05-tips_for_prompt/2.prompt_structure.md' },
          { text: '3.3 提示词要素', link: '05-tips_for_prompt/3.prompt_elements.md' },
          { text: '3.4 其他提示词技巧', link: '05-tips_for_prompt/4.other_prompt_techniques.md' },
        ]},
        { text: '第四章 常见任务示例', link: '06-common_task_examples/README.md' },
        { text: '第五章 多模态大模型提示词', link: '07-multimodal_prompt/README.md' },
        { text: '第六章 合成数据', link: '08-synthetic_data/README.md' , items: [
          { text: '6.1 预训练合成数据', link: '08-synthetic_data/1.pretrain_data.md' },
          { text: '6.2 微调合成数据', link: '08-synthetic_data/2.instruct_data.md' },
          { text: '6.3 推理合成数据', link: '08-synthetic_data/3.reasoning_data.md' },
        ]},
        { text: '第七章 RAG检索', link: '09-RAG/README.md' },
        { text: '第八章 Agent实践', link: '10-Agent/README.md' , items: [
          { text: '8.1 函数调用实践', link: '10-Agent/1.function_calling.md' },
          { text: '8.2 MCP实践', link: '10-Agent/2.mcp_usage.md' },
          { text: '8.3 多Agents简介', link: '10-Agent/3.multi_agents.md' },
        ]},
        { text: '第九章 项目实战', link: '11-swanlab_rag/README.md' , items:[
          { text: '9.1 Swanlab-RAG实战', link: '11-swanlab_rag/1.swanlab-rag.md' },
        ]},
      ]
    },
  ]
}


function sidebarCourseLLMTrain(): DefaultTheme.SidebarItem[] {
  return [
    {
      text: '📖大模型训练课程',
      items: [
        { text: '简介', link: '00-preface/README.md' },
        { text: '第一章 传统模型', items: [
          { text: '1.1 Bert文本分类', link: '01-traditionmodel/1.bert/README.md' },
          { text: '1.2 LSTM股票预测', link: '01-traditionmodel/2.lstm/README.md' },
          { text: '1.3 RNN教程', items: [
            { text: '原理简介', link: '01-traditionmodel/3.rnn/rnn_tutorial_1.md' },
            { text: '序列预测模型构建', link: '01-traditionmodel/3.rnn/rnn_tutorial_2.md' },
              ]
            },
          ]
        },
        { text: '第二章 预训练', items: [
          { text: '2.1 LLM预训练', link: '02-pretrain/1.qwen-pretrain/README.md' },
        ]},
        { text: '第三章 微调', items: [
          { text: '3.1 Qwen文本分类', link: '03-sft/1.text_classification/README.md' },
          { text: '3.2 Qwen命名体识别', link: '03-sft/2.ner/README.md' },
          { text: '3.3 GLM4指令微调', link: '03-sft/3.glm4-instruct/README.md' },
          { text: '3.4 Qwen3医学模型微调', link: '03-sft/4.qwen3-medical-finetune/README.md' },
          { text: '3.5 Mac上微调Qwen3模型', link: '03-sft/5.mac-qwen3-finetune/README.md' },
          { text: '3.6 llamafactory框架QLoRA微调', items: [
            { text: 'QLoRA原理', link: '03-sft/6.llamafactory-finetune/lora1.md' },
            { text: 'QLoRA微调实战', link: '03-sft/6.llamafactory-finetune/lora2.md' },
          ]},
          { text: '3.7 deepseek模型lora微调', link: '03-sft/7.deepseek-lora/README.md' },
          { text: '3.8 其他框架微调', items: [
            { text: 'PaddleNLP', link: '03-sft/8.other_frameworks/paddlenlp_finetune.md' },
            { text: 'ms-swift', link: '03-sft/8.other_frameworks/ms-swift.md' },
          ]}
            ]
          },
        { text: '第四章 强化学习',items: [
          { text: '4.1 Qwen复现R1-Zero', link: '04-reinforce/2.qwen_grpo/README.md' },
          { text: '4.2 数独游戏GRPO训练', link: '04-reinforce/3.sudoku_grpo/README.md' },
        ]},
        { text: '第五章 评估', items: [
          { text: '5.1 EvalScope使用', link: '05-eval/1.evalscope/README.md' },
        ]},
        { text: '第六章 视觉大模型', items: [
          { text: '6.1 Qwen2-VL微调', link: '06-multillm/1.qwen_vl_coco/README.md' },
          { text: '6.2 Qwen3-smVL模型拼接微调', link: '06-multillm/2.qwen3_smolvlm_muxi/README.md' },
          { text: '6.3 Qwen2.5-VL目标检测微调', link: '06-multillm/4.grounding/README.md' },
      ]},
      { text: '第七章 音频大模型', items: [
          { text: '7.1 CosyVoice微调派蒙语音', link: '07-audio/1.cosyvoice-sft/README.md' },
      ]},
      { text: '第八章 扩散模型', items: [
        { text: '8.1 LLaDA模型预训练和微调', link: '08-diffusion/1.llada/README.md' },
      ]}
      ]
    }
  ]
}