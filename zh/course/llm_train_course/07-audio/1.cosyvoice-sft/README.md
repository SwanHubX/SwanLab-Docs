# 用CosyVoice2实现派蒙语音的微调

[![SVG Banners](https://svg-banners.vercel.app/api?type=origin&text1=CosyVoice%20SFT%20🤠&text2=Text-to-Speech%20💖%20Genshin%20Paimon&width=800&height=300)](https://github.com/Akshay090/svg-banners)

作者信息：情感机器实验室研究员-李馨雨 邮箱：wind.340171@gmail.com

代码：[https://github.com/828Tina/cosyvoice-paimon-sft](https://github.com/828Tina/cosyvoice-paimon-sft)

数据集：[https://www.modelscope.cn/datasets/aihobbyist/Genshin_Dataset](https://www.modelscope.cn/datasets/aihobbyist/Genshin_Dataset)

模型：[https://www.modelscope.cn/models/iic/CosyVoice2-0.5B](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B)

SwanLab结果：[https://swanlab.cn/@LiXinYu/cosyvoice-sft/overview](https://swanlab.cn/@LiXinYu/cosyvoice-sft/overview)

## 简介

<div style="display:flex;justify-content:center;">
  <figure style="text-align:center;margin:0;">
    <img src="./cosyvoice/paimon4.png" style="width:100%">
  </figure>
</div>

一直以来我们的教程基本都是自然语言领域的训练，多模态领域也基本就是图像相关，很少研究音频模型，本次教程我们就研究研究如何来训练一波音频模型。

模型我们选择通义实验室发布的CosyVoice2模型来微调原神中的派蒙语音，我们会非常详细地按照步骤教会大家如何训练CosyVoice。并且作者也会写出自己对于CosyVoice原理的理解，希望能帮到各位读者。

> 特此声明：本次教程仅作为AI模型训练，数据集均来自开源数据集。

<div style="min-width: 600px;">
  <h3 style="padding: 12px 16px; margin: 0; text-align: center; color: #050505ff; border-bottom: 1px solid #787878ff; width: 100%;">音频效果对照表（全是派蒙）</h3>
  <div style="overflow-x: auto; width: 100%;">
    <table style="width: 100%; border-collapse: collapse; min-width: 1100px;">
      <thead>
        <tr style="background: #f5f5f5;">
          <th style="padding: 8px 12px; text-align: left; border: 1px solid #e0e0e0; text-align: center;color: #050505ff;">Text</th>
          <th style="padding: 8px 12px; text-align: left; border: 1px solid #e0e0e0; text-align: center;color: #050505ff;">CosyVoice2</th>
          <th style="padding: 8px 12px; text-align: left; border: 1px solid #e0e0e0; text-align: center;color: #050505ff;">CosyVoice2 SFT</th>
          <th style="padding: 8px 12px; text-align: left; border: 1px solid #e0e0e0; text-align: center;color: #050505ff;">CosyVoice3</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="padding: 8px 12px; border: 1px solid #e0e0e0;word-wrap: break-word; word-break: break-all; max-width: 200px;"><strong>游戏内文本：</strong>欸！蒙德城的修女还能负责抓捕可疑人士的吗？！</td>
          <td style="padding: 8px 12px; border: 1px solid #e0e0e0;"><audio controls style="width: 280px;"> <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/examples/cosyvoice2/zero_shot_0.wav" type="audio/wav">  </audio></td>
          <td style="padding: 8px 12px; border: 1px solid #e0e0e0;"><audio controls style="width: 280px;"> <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/examples/cosyvoice2sft/paimon_sft_inference_0.wav" type="audio/wav">  </audio></td>
          <td style="padding: 8px 12px; border: 1px solid #e0e0e0;"><audio controls style="width: 280px;"> <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/examples/cosyvoice3/zero_shot_0.wav" type="audio/wav">  </audio></td>
        </tr>
        <tr>
          <td style="padding: 8px 12px; border: 1px solid #e0e0e0;word-wrap: break-word; word-break: break-all; max-width: 200px;"><strong>现实文本：</strong>现代科技让世界变得更加紧密相连。</td>
          <td style="padding: 8px 12px; border: 1px solid #e0e0e0;"><audio controls style="width: 280px;"> <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/examples/cosyvoice2/zero_shot_1.wav" type="audio/wav">  </audio></td>
          <td style="padding: 8px 12px; border: 1px solid #e0e0e0;"><audio controls style="width: 280px;"> <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/examples/cosyvoice2sft/paimon_sft_inference_1.wav" type="audio/wav">  </audio></td>
          <td style="padding: 8px 12px; border: 1px solid #e0e0e0;"><audio controls style="width: 280px;"> <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/examples/cosyvoice3/zero_shot_1.wav" type="audio/wav">  </audio></td>
        </tr>
        <tr>
          <td style="padding: 8px 12px; border: 1px solid #e0e0e0;word-wrap: break-word; word-break: break-all; max-width: 200px;"><strong>情绪化文本：</strong>啊啊啊！真是让人生气！他怎么可以这样说！我明明不是那样的人！</td>
          <td style="padding: 8px 12px; border: 1px solid #e0e0e0;"><audio controls style="width: 280px;"> <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/examples/cosyvoice2/zero_shot_2.wav" type="audio/wav">  </audio></td>
          <td style="padding: 8px 12px; border: 1px solid #e0e0e0;"><audio controls style="width: 280px;"> <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/examples/cosyvoice2sft/paimon_sft_inference_2.wav" type="audio/wav">  </audio></td>
          <td style="padding: 8px 12px; border: 1px solid #e0e0e0;"><audio controls style="width: 280px;"> <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/examples/cosyvoice3/zero_shot_2.wav" type="audio/wav">  </audio></td>
        </tr>
        <tr>
          <td style="padding: 8px 12px; border: 1px solid #e0e0e0;word-wrap: break-word; word-break: break-all; max-width: 200px;"><strong>方言文本：</strong>用四川话说：走哦兄弟，楼下新开的火锅店巴适得板，味道绝了，我们整起，保证吃得你肚皮圆滚滚！</td>
          <td style="padding: 8px 12px; border: 1px solid #e0e0e0;"><audio controls style="width: 280px;"> <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/examples/cosyvoice2/instruction_zero_shot.wav" type="audio/wav">  </audio></td>
          <td style="padding: 8px 12px; border: 1px solid #e0e0e0;"><audio controls style="width: 280px;"> <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/examples/cosyvoice2sft/instruction_zero_shot.wav" type="audio/wav">  </audio></td>
          <td style="padding: 8px 12px; border: 1px solid #e0e0e0;">待补充...</td>
        </tr>
        <tr>
          <td style="padding: 8px 12px; border: 1px solid #e0e0e0;word-wrap: break-word; word-break: break-all; max-width: 200px;"><strong>跨语言文本：</strong>Has the author of my issue seen it?</td>
          <td style="padding: 8px 12px; border: 1px solid #e0e0e0;"><audio controls style="width: 280px;"> <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/examples/cosyvoice2/cross_lingual_zero_shot_0.wav" type="audio/wav">  </audio></td>
          <td style="padding: 8px 12px; border: 1px solid #e0e0e0;"><audio controls style="width: 280px;"> <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/examples/cosyvoice2sft/cross_lingual_zero_shot_0.wav" type="audio/wav">  </audio></td>
          <td style="padding: 8px 12px; border: 1px solid #e0e0e0;">待补充...</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

## CosyVoice原理

未完待续...

## 零样本推理

关于零样本推理，既可以参考官方给出的[代码](https://github.com/FunAudioLLM/CosyVoice/blob/main/example.py)，也可以直接运行我给出的`inference.sh`，直接运行下面的代码，就可以直接生成零样本推理结果，以及如果你有微调后的模型，可以利用微调后的模型进行推理，生成音频。

```bash
bash inference.sh
```

这里说明下参数含义：

**零样本推理**

如果仅进行零样本推理，注意脚本中的进程设置为0:

```bash
stage=0
stop_stage=0
```

参数含义分别如下：

- `--tts_text`:你需要模型生成的文本内容对应的地址，需要注意的是内部有以下标签：
    
```json
{
  "paimon":{
      "zero-shot": [
      "卖唱的怎么又跑去喝酒了！",
    ],
    "cross-lingual": [
      "Hello, my name is Paimon. I am a character from the game Genshin Impact. I love to explore the world with my traveler friend."
    ],
    "instruction-zero-shot": {
      "instruction":"用广东话说这句话",
      "text":"我觉得璃月的火锅很好吃！"
    },
    "sft_inference":[
      "卖唱的怎么又跑去喝酒了！",
    ]
  }
}
```

`paimon`:你的说话人身份，这里按照名字来识别会比较方便

`zero-shot`:推理类型为零样本生成对应的文本

`cross-lingual`:推理类型为跨语言生成对应的文本

`instruction-zero-shot`:推理类型为方言生成对应的文本

`sft_inference`:推理类型为微调后的模型推理文本

- `--model_dir`:你的模型地址，这里用原始模型地址

- `--spk_id`:说话人身份，这里为`paimon`

- `--test_data_dir`:参考声线对应的数据地址

- `--example_id`:参考声线数据的编号，比如`1_4`中的4

- `--target_sr`:采样频率

- `--result_dir`:推理结果保存地址

- `--task_type`:推理类型

> 注意：CosyVoice支持的多语言类型仅为中文、英文、日文、韩文、中文方言（粤语、四川话、上海话、天津话、武汉话等）这些，CosyVoice3支持的更多些

**微调后推理**

如果是微调后模型推理，需要注意进程设置为1:

```
stage=1
stop_stage=1
```

参数和零样本推理基本一致，只有一个参数需要注意：

- `emb_path`:这个为说话人对应的embedding，我设置为对训练数据取平均后的embedding，也就是`./data/train/spk2embedding.pt`文件地址，如果你希望用单独的一个数据对应的embedding，可以替换`spk2embedding.pt`为`utt2embedding.pt`。

> 注意⚠️：在运行这个代码前，最好确保run.sh中的进程4已经运行，不然找不到对应模型地址。

## SFT代码

完整的代码其实用的是官方给的[example](https://github.com/FunAudioLLM/CosyVoice/tree/main/examples/libritts/cosyvoice2)，只要环境和配置设置正确，直接可以用，不过我对其进行了一点小小的改造🤏，具体的我们下面详细讲述。

### 1. 环境安装

- 克隆代码

```bash
git clone --recursive https://github.com/...
cd Genshin_CosyVoice_sft
git submodule update --init --recursive
```

- 安装环境

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

<div style="background:#e7f8ff;color:#000;padding:12px 16px;border-left:4px solid #20c0ff;">
本次教程我们使用5090来训练模型，由于cosyvoice这个模型也比较早了，当时的pytorch版本没有那么高，没有什么bug，但是5090安装的驱动CUDA适配的pytorch版本较高，就导致我实际在跑的时候报了很多bug，下面我会依次说明都出了哪些问题并如何解决。（我的代码应该是可以直接用的，但是如果报环境错误，可以根据下面给出的信息来查看是否是对应的bug）
</div>

---

**环境安装Bug汇总**

1. 缺少一个模型库`whisper`

在实际跑训练代码的时候，在数据集预处理阶段中，可能会报下面的错误：

```bash
AttributeError: module 'whisper' has no attribute 'log_mel_spectrogram'
```

这可能是`whisper`版本错了，但是安装的时候要注意，库的名字全称是`openai-whisper`，我们直接在[Releases](https://github.com/openai/whisper/releases)中找到最新的，运行下面的代码（`requirements.txt`我已经修改，所以可能不会报这个错误，但是如果在数据集预处理阶段有问题，可以来查看是否是这个库的问题）：

```bash
pip install openai-whisper==v20250625
```

2. `Pytorch`版本问题

我用的GPU是RTX 5090，如果按照cosyvoice官方给的`requirements.txt`直接安装环境，会有如下错误：

```bash
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
If you want to use the NVIDIA GeForce RTX 5090 GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/

  warnings.warn(
/home/lxy/miniconda3/envs/tts/lib/python3.10/site-packages/torch/cuda/__init__.py:235: UserWarning: 
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
If you want to use the NVIDIA GeForce RTX 5090 GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/
```

这个警告的核心问题是PyTorch 安装版本与 RTX 5090 显卡的 CUDA 算力不兼容：RTX 5090 的 CUDA 算力为 `sm_120`，而当前安装的 PyTorch 仅支持 `sm_50/sm_60/sm_70/sm_75/sm_80/sm_86/sm_90` 这些旧算力版本，无法适配该显卡的新算力规格。

这个问题其实主要集中于比较新的GPU中，如果想要解决，升级 PyTorch 到支持 `sm_120` 的版本（需匹配对应 CUDA Toolkit 和显卡驱动），或从源码编译 PyTorch 并指定 `sm_120` 算力，确保与 RTX 5090 的硬件规格适配。具体可以查看这个👉[CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus)

<div style="display:flex;justify-content:center;">
  <figure style="text-align:center;margin:0;">
    <img src="./cosyvoice/paimon5.png" style="width:100%">
  </figure>
</div>

**由于我们的驱动CUDA安装版本为12.8，因此我们选择直接升级`Pytorch`版本到能够有匹配的算力版本，由于我没有找到具体的`Pytorch`算力支持表，因此我是直接试出来的，要求`Pytorch`版本为2.8--cuda128，对应的`torchvision`和`torchaudio`也选择匹配的版本就行。**

3. `torchaudio`安装不完整

上面`Pytorch`安装完后，可能会报下面的错误：

<div style="display:flex;justify-content:center;">
  <figure style="text-align:center;margin:0;">
    <img src="./cosyvoice/paimon6.png" style="width:100%">
  </figure>
</div>

这是因为缺少`torchcodec`模块：在数据加载器（Dataloader）工作进程中，`torchaudio的_torchcodec.py`文件尝试导入`AudioDecoder`类，但系统找不到名为`torchcodec`的模块，触发了`ModuleNotFoundError`。

这个模块缺失导致数据加载流程中断，进而引发后续的迭代（如shuffle/sort/dynamic_batch等数据预处理步骤）无法正常执行。

问题根源是`torchaudio`安装不完整（`torchcodec`是其依赖组件），或当前环境中`torchaudio`版本与 `PyTorch` 不兼容，需重新安装匹配版本的`torchcodec`以补全依赖模块。

**我们的解决办法是直接安装对应的`torchcodec`**

```bash
pip install --pre torchcodec \
  --index-url https://download.pytorch.org/whl/nightly/cu128
```

如果遇到比如pytorch和torchcodec版本不匹配的问题，可以参考这个链接：[https://github.com/meta-pytorch/torchcodec?tab=readme-ov-file#installing-cpu-only-torchcodec](https://github.com/meta-pytorch/torchcodec?tab=readme-ov-file#installing-cpu-only-torchcodec)

4. 代码问题

其实严格来说，这个并不是代码问题，单纯是因为`CosyVoice`时间比较早，当时的`Pytorch`和现在相比有一点点不太一样...如果直接运行官方的`run.sh`会报下面的错误：

<div style="display:flex;justify-content:center;">
  <figure style="text-align:center;margin:0;">
    <img src="./cosyvoice/paimon7.png" style="width:100%">
  </figure>
</div>

定位到指定地点：

```python
try:
    # dist.monitored_barrier(group=group_join,
    #                        timeout=group_join.options._timeout)
```

因为前面已经将`Pytorch`改成了高版本2.8，`options`已经被删掉了，在官方的文档中并没有看到这个参数👉[Pytorch Docs](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.new_group)，那么这个问题我们只能从原理入手：

我们找到[monitored_barrier](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.monitored_barrier)，原理如下：

torch.distributed.monitored_barrier 是一个 集体阻塞式同步原语，在分布式进程组内提供 `可配置超时 + 失败报告` 的能力。其核心语义为：

1. 全局屏障
   所有属于 group 的 ranks 必须调用该函数并到达同步点，否则整个集体操作永远不会完成。
2. 超时监测
   通过 timeout 参数设定最大阻塞时间；若在该时间内任意 rank 未能完成握手，则抛出 RuntimeError 并 立即中断所有参与者，防止因个别 rank 挂起导致的 分布式死锁。
3. 失败定位
   由 rank 0 担任协调者，负责收集各 rank 的握手消息；超时后会在 rank 0 侧生成详细的失败日志，指出未响应的 rank 列表，便于调试定位。
4. 使用约束
   - 只能在 NCCL / GLOO / MPI 等支持点对点通信的后端使用；
   - 非零 rank 在函数内部执行 send/recv 与 rank 0 握手；
   - rank 0 执行 all-reduce-like 的握手计数，因而函数返回前会 阻塞 host 线程；
   - 由于涉及 host-side 同步，性能开销大，仅推荐用于 调试、负载均衡检测、checkpoint 一致性校验 等场景。

综上，monitored_barrier 在保持 barrier 语义的同时，增加了 `超时控制 + 故障报告` 机制，是分布式训练中诊断`rank`间同步异常的重要工具。

🤔总之，这里本质就是一个超时检测，具体要设定一个时间，那么我直接设置一个足够多的固定时间即可：

```python
# we try to join all rank in both ddp and deepspeed mode, in case different rank has different lr
# 固定 30 秒，或从环境变量读
timeout = datetime.timedelta(seconds=30)   # ← 直接构造
try:
    # dist.monitored_barrier(group=group_join,
    #                        timeout=group_join.options._timeout)
    dist.monitored_barrier(group=group_join, timeout=timeout)
    return False
```

然后就可以解决问题了😄。

### 2. 数据处理

数据处理这一步，其实官方已经给出了脚本，在`run.sh`中的前4步中，如下所示：

```bash
data_dir=/data/tts-data/paimon_4k
pretrained_model_dir=/data/tts-models/cosyvoice2-0.5b
output_model_dir=/data/tts-models/cosyvoice2-0.5b-paimon

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation, prepare wav.scp/text/utt2spk/spk2utt"
  # 下面的文件夹要换成我们自己的数据集名称
  for x in test train; do
    mkdir -p data/$x
    python local/prepare_data.py --src_dir $data_dir/$x --des_dir data/$x
  done
fi

# 存成 utt2embedding.pt 和 spk2embedding.pt，后面做说话人相关任务（区分、聚类、多说话人 TTS 等）直接拿来用。
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extract campplus speaker embedding, you will get spk2embedding.pt and utt2embedding.pt in data/$x dir"
  for x in test train; do
    tools/extract_embedding.py --dir data/$x \
      --onnx_path $pretrained_model_dir/campplus.onnx
  done
fi

# 把每条语音（≤30 s）喂进一个 ONNX 版的 「 Whisper-风格语音离散化 tokenizer 」，输出一串整数编号（speech token）
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extract discrete speech token, you will get utt2speech_token.pt in data/$x dir"
  for x in test train; do
    tools/extract_speech_token.py --dir data/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx
  done
fi

# 把准备好的数据整理成训练时需要的 parquet 格式，方便后续大规模分布式训练时高效读取
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for x in test train; do
    mkdir -p data/$x/parquet
    tools/make_parquet_list.py --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --src_dir data/$x \
      --des_dir data/$x/parquet
  done
fi
```

核心在于序号`1`和序号`2`，分别是我们在原理中讲到的`speaker-vector`也就是`embedding`，和`speech-tokens`。其中`speech-tokens`是`llm`模型训练的时候预测出来对照计算loss和accuracy的部分，而`speaker-vector`用在`flow`模型的训练中，用于匹配说话人音色。`speaker-vector`后续也可以作为推理使用。

而如果想要运行这段代码，你的数据集要符合一些要求，说起来比较抽象，我直接展示下你需要喂给脚本中的数据集格式：

```python
├── your data_dir/
│   ├── test/
│   │   ├── 1_1.wav
│   │   ├── 1_1.normalized.txt
│   │   ├── 1_2.wav
│   │   ├── 1_2.normalized.txt
│   │   ├── 1_3.wav
│   │   ├── 1_3.normalized.txt
│   │   └── ...
│   └── train/
│       ├── 1_1.wav
│       ├── 1_1.normalized.txt
│       ├── 1_2.wav
│       ├── 1_2.normalized.txt
│       ├── 1_3.wav
│       ├── 1_3.normalized.txt
│       └── ...
```

其中`test`和`train`分别是验证集和训练集，名字其实并不是必要的，但是要和脚本`for x in test train; do`中的名字保持一致。

**当然最重要的是，`.wav`和`.normalized.txt`内容一定要对应，否则训练的时候可能会出现乱码。**

OK啊，我们知道要喂给脚本的数据集具体长什么样子，那么我们来简单的处理一波：

首先，我们从魔搭社区中选择[原神语音数据集](https://www.modelscope.cn/datasets/aihobbyist/Genshin_Dataset)，因为这是个集合了所有角色的语音包，太大了，所以，我们选择我们需要的角色，这里你当然可以选择其他自己喜欢的角色，但是由于我们的教程是派蒙语音，所以我们选择派蒙语音包，点击直接下载👉[派蒙语音包](https://pan.acgnai.top/d/datasets/%E5%8E%9F%E7%A5%9E/%E4%B8%AD%E6%96%87/%E6%B4%BE%E8%92%99.7z?sign=Xj4ykQUYkiq6mz0aa4ntMJYzCPGURSxlE7bQFd4Zk9Y=:0)。

<div style="display:flex;justify-content:center;">
  <figure style="text-align:center;margin:0;">
    <img src="./cosyvoice/paimon8.png" style="width:80%">
  </figure>
</div>

解压缩了之后大概是这样，其中带变量语音是包含旅行者名字的特殊字符`{NICKNAME}`（当然可能有其他的特殊字符，这里不过多赘述）。其他语音是一些语气对应的音频，比如叹气的声音但是没有文本的那种。

如果数据足够多，我们就不需要带变量语音和其他语音，只用正常的语音数据即可，这里就只用正常语音，其他的不用的直接删掉就行。

然后由于数据集中wav和txt（lab文件）分开了，我们需要按照顺序保存成上面我们说的样子，也就是一一对应，原理很简单，只要找名字一样的文件就行，不过这里为了方便，我直接提供了预处理代码，只要输入对应的数据地址就行：

```bash
python ./tools/data_save_process.py \
    --spk_id paimon \
    --orig_data_path your original data path \
    --target_data_path your target data path
```

大致结果如下：

<div style="display:flex;justify-content:center;">
  <figure style="text-align:center;margin:0;">
    <img src="./cosyvoice/paimon9.png" style="width:100%">
  </figure>
</div>

我们训练的时候用其中5000条就行，也就是一个文件夹里的数据足够，再多了除了增加训练时间外没什么必要。

然后训练的时候可以按照9:1的比例分割数据作为`train`和`test`，这里为了方便，我直接用前4000条数据集作为训练集，后400条作为验证集，然后按照之前说的`your data_dir`里保存这两部分数据集，用`data_dir`地址使用脚本里前5步（这里多了一步也就是最后一步是我写的，用于`swanlab`每一个epoch展示音频结果使用，本质也就是将`spk2embedding.pt`中的信息保存到模型文件中）

你会分别得到下面说的文件，**注意：如果有些文件没有，请务必检查是哪一步没保存下来，下面所有的文件后续训练都要使用，因此不能缺少任何一个文件。**

<div style="display:flex;justify-content:center;">
  <figure style="text-align:center;margin:0;">
    <img src="./cosyvoice/paimon10.png" style="width:100%">
  </figure>
</div>

只要上述所代表的训练集和验证集里这些文件都有，那么接下来你就可以开始训练了😄。

### 3. 训练代码

由于`Cosyvoice`官方已经给出了训练脚本👉[run.sh](https://github.com/FunAudioLLM/CosyVoice/blob/main/examples/libritts/cosyvoice2/run.sh#L55)，其中步骤5是训练部分，步骤6其实是对生成的结果求平均，训练阶段可以步骤5和步骤6按顺序来。但是由于我们进行了简单的改造，所以实际运行我们自己的代码即可👉[ours](https://github.com/828Tina/cosyvoice-paimon-sft/blob/main/run.sh#L86)

我们修改`run.sh`的开头，然后运行`run.sh`训练模型：

```bash
### 修改训练脚本run.sh
# 只训练
stage=5
stop_stage=5
...

# 训练+取平均保存模型权重
stage=5
stop_stage=6
...

# 修改模型地址
pretrained_model_dir=/data/tts-models/cosyvoice2-0.5b
output_model_dir=/data/tts-models/cosyvoice2-0.5b-paimon
...

# 指定训练的卡
CUDA_VISIBLE_DEVICES=your GPU id
...
```

另外，超参数设置在`./conf/cosyvoice2-paimon.yaml`中，我们需要修改的仅为`train_conf`和`swan`部分，其中`train_conf`为超参数，`swan`为训练观测参数。在`run.sh`中记得设置`--config conf/cosyvoice2-paimon.yaml`。

<div style="background:#e7f8ff;color:#000;padding:12px 16px;border-left:4px solid #20c0ff;">
很凑巧的是，在写这篇教程的时候，cosyvoice3发布了😂，我们检查了代码，发现有如下修改：<br>
1. 模型地址在脚本中修改了<br>
2. cosyvoice部分，主要是加入了instruct的处理<br>
因此我直接训练了一波看看效果如何，其中超参数文件用<strong>cosyvoice3-paimon.yaml</strong>。
</div>

<div style="display:flex;justify-content:center;">
  <figure style="text-align:center;margin:0;">
    <img src="./cosyvoice/paimon11.png" style="width:100%">
  </figure>
</div>

```bash
### 启动训练
bash run.sh
```

### 4. SwanLab设置

官方使用的是`tensorboard`来观测loss等变化，为了方便起见，我们选择一个能直接在线观察train loss变化的`SwanLab`，然后我们对代码进行一点改造。

在`./utils/train_utils.py`中找到`tensorboard`使用的地方，然后将tensorboard生成的log迁移到`swanlab`中：

```python
from torch.utils.tensorboard import SummaryWriter
import swanlab

...

def init_summarywriter(args,swan_config):
    writer = None
    if int(os.environ.get('RANK', 0)) == 0:
        swanlab.login(api_key=swan_config['api_key'])
        swanlab.init(
            project=swan_config['project_name'],
            experiment_name=swan_config['experiment_name'],
            description=swan_config['description'],
            config=args.__dict__
        )
        swanlab.sync_tensorboard_torch()
        os.makedirs(args.model_dir, exist_ok=True)
        writer = SummaryWriter(args.tensorboard_dir)
    return writer
```

然后在`run.sh`中添加一个新的键`swan`，需要设置如下参数：

```yaml
# swanlab
swan:
    api_key: your swanlab api_key
    project_name: paimon-cosyvoice
    experiment_name: cosyvoice2-paimon-llm+flowtrain
    description: paimon dataset llm+flow training
```

> 如果为了不想每次输入`api_key`导致信息泄露，可以删掉该参数，在外部只要最开始`login`一次就行

**SwanLab**有一个功能`Audio`，可以直接在网页端上传并听音频文件，因此我们需要设置在每个epoch结束后，保存模型并进行推理，生成音频播放到`SwanLab`中，代码我已经给出，因此不需要读者做太复杂的操作，仅需要注意一个小的地方。

**实际训练的时候，不需要读者做什么操作，只需要运行训练脚本即可，但是如果想了解具体原理，可以看下面的内容。**

首先我们需要确定模型训练时保存每次模型权重的位置：

[https://github.com/828Tina/cosyvoice-paimon-sft/blob/main/cosyvoice/utils/executor.py#L192](https://github.com/828Tina/cosyvoice-paimon-sft/blob/main/cosyvoice/utils/executor.py#L192)

确认了代码地址之后，我们就可以补充后续的代码。

因为每个epoch会保存checkpoint，我们需要将每一个.pt文件转换成对应模型训练的权重文件llm.pt&flow.pt。

```python
# 保存的模型地址：exp/cosyvoice2/llm...
save_model_dir = info_dict['model_dir']
# 将当前的保存好的模型名称转换成llm.pt或者flow.pt并保存，原始模型不能覆盖
# 要先读取，然后保存
current_checkpoint_name = f"{model_name}.pt"
# 具体的保存的新的模型的地址：exp/cosyvoice2/llm/torch_ddp/epoch_{}_whole.pt
current_checkpoint_path = os.path.join(save_model_dir, current_checkpoint_name)
state_dict = torch.load(current_checkpoint_path, map_location='cpu')
if 'model' in state_dict:
    state_dict = state_dict['model']
state_dict = {k: v for k, v in state_dict.items() if k not in {'epoch', 'step'}}
# 新名字
# 模型名字，要根据模型类型来确定，llm、flow
model_type = info_dict['model']
target_model_name = f"{model_type}.pt"
target_model_path = os.path.join(save_model_dir, target_model_name)
# 改名字，但是原始模型不能覆盖,新的地址：exp/cosyvoice2/llm/torch_ddp/llm.pt
torch.save(state_dict, target_model_path)
```

然后把llm.pt&flow.pt转移到我们的`output_model_dir`文件中替换原始的权重文件。

```python
# 最终的目标模型地址：/data/tts-models/cosyvoice2-0.5b-mydeimos/llm
final_target_model_path = os.path.join(info_dict['target_model_path'],model_type)
# 具体旧的模型的地址，但是最终是新模型的保存地址:/data/tts-models/cosyvoice2-0.5b-mydeimos/llm/llm.pt
final_model_path = os.path.join(final_target_model_path, target_model_name)
# 替换
shutil.copyfile(target_model_path, final_model_path)
```

最后根据我们的模型地址，使用`CosyVoice`给出的`sft`后对应的推理函数`inference_sft`（不是零样本推理）推理生成音频文件，上传到`swanlab`中即可。

具体效果如下：

<div style="display:flex;justify-content:center;">
  <figure style="text-align:center;margin:0;">
    <img src="./cosyvoice/paimon12.png" style="width:100%">
  </figure>
</div>

## SwanLab观测结果

本次实验我们总共进行了两个模型的训练，分别是`llm`和`flow`模型，然后在训练`flow`模型的时候，除了展示本来`flow`模型的音频效果，还补充了`llm`和`flow`训练后权重组合的音频效果。

可以直接看完整结果，并且可以在线听👉[https://swanlab.cn/@LiXinYu/cosyvoice-sft/overview](https://swanlab.cn/@LiXinYu/cosyvoice-sft/overview)

**仅llm训练的曲线图和音频结果**

<div style="display:flex;justify-content:center;">
  <figure style="text-align:center;margin:0;">
    <img src="./cosyvoice/paimon13.png" style="width:100%">
  </figure>
</div>

<div style="display:flex;justify-content:center;">
  <figure style="text-align:center;margin:0;">
    <img src="./cosyvoice/paimon14.png" style="width:100%">
  </figure>
</div>

<div style="display:flex;justify-content:center;">
  <figure style="text-align:center;margin:0;">
    <img src="./cosyvoice/paimon15.png" style="width:100%">
  </figure>
</div>

**flow训练的曲线图和音频结果**

<div style="display:flex;justify-content:center;">
  <figure style="text-align:center;margin:0;">
    <img src="./cosyvoice/paimon16.png" style="width:100%">
  </figure>
</div>

<div style="display:flex;justify-content:center;">
  <figure style="text-align:center;margin:0;">
    <img src="./cosyvoice/paimon17.png" style="width:100%">
  </figure>
</div>

其中`Train`就是训练集训练的过程，`CV`可以理解为每个`epoch`完成后进行验证集做前向传播的过程。

这里需要注意几点⚠️：

1. 对于准确度`acc`，只有`llm`才有，`flow`是没有的，这是因为`llm`其实是根据输入的文本预测对应语音的speech tokens，这个tokens其实就可以理解为字典里的编号，那么离散的编号数字当然可以求准确度，同时也保证在反向传播过程中结合loss提升模型训练的效果。`flow`模型理论上是根据speech tokens生成对应的梅尔频谱图，这种非离散状态的输出无法做准确度计算，因此只有loss。
2. `flow`里有个`llm_flow`的效果展示，这是我做的一点小改动🤏，用于在`llm`训练好后，在`flow`训练的时候可以同时对比只有`flow`微调以及两个都微调后音频效果，方便随时观察音频效果变化。
3. `example: same as the last one`这个是我在训练数据中随便找了一个数据作为参考，因为可能有些读者并没有玩过原神，不清楚派蒙的声音，因此做了个对比。

## 微调效果测试

在进行推理的时候，我们需要将说话人对应的音色`embedding`作为输入，用`instruct_sft`来生成对应目标文本的音频文件，而如果只微调`llm`或者`flow`单独一个模型，效果并不是很好，这是因为单独一个分别有自己的任务要完成。

- `llm`聚焦 “文本与音频的语义、指令对齐”，不直接学习声学细节：

    1. 文本到语义令牌的映射：将输入文本（含自然语言指令）转化为离散的 speech tokens，确保内容一致性（比如文本 “开心地打招呼” 对应包含 “开心” 情绪倾向的语义令牌）。
    2. 指令解析与执行逻辑：学习理解 “用粤语说”“快速朗读” 等指令，将指令信息编码到语义令牌中，为后续声学生成提供高层指导。
    3. 语言与格式适配：掌握多语言语法、文本归一化规则（如数字、符号的口语化转换），以及多音字、生僻词的发音逻辑（通过发音修复模块辅助）。

- `flow`模型聚焦 “将语义令牌转化为真实音频波形”，核心学习目标是声学细节的生成与还原：
    1. 声学特征建模：基于 speech tokens，生成对应的 Mel 频谱（音频的核心声学特征），涵盖频率分布、时长匹配等基础声学属性。
    2. 环境与噪声适配：学习处理真实场景中的噪声、背景音，生成符合场景的自然音频（比如还原参考音频的轻微背景噪声）。
    3. 细粒度声学控制：响应 LLM 传递的指令信息，调整语速、音量等声学参数（如 “快速朗读” 对应更快的帧生成节奏）。

我们简单总结下📖：

|模型组件|核心学习目标|是否学习音频韵律/音色/语调|角色定位|
|:---:|:---|:---|:---|
|llm|文本-语义-指令对齐|否（仅传递高层倾向信息）|内容与指令理解者|
|flow|声学细节生成、个性化特征还原|	是（主导学习）|音频生成执行者|

如果文字表述比较抽象的话，可以听下下面展示出来的音频结果，分别是单独模型组件和两个模型组件一起作用时的结果。

| 微调方案  | 音频效果演示 & 对应自定义文本   |
| ------------ | ------------------------------- |
| 仅llm微调后      | <audio controls style="width: 400px;"> <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/single_example/paimon_sft_inference_llm.wav" type="audio/wav">  </audio> <br> **文本内容**：有任何问题都可以来找我，我可是提瓦特最棒的、最厉害的、知道的最多的向导！     |
| 仅flow微调后     | <audio controls style="width: 400px;"> <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/single_example/paimon_sft_inference_flow.wav" type="audio/wav">  </audio> <br> **文本内容**：有任何问题都可以来找我，我可是提瓦特最棒的、最厉害的、知道的最多的向导！    |
| llm和flow都微调后 | <audio controls style="width: 400px;"> <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/single_example/paimon_sft_inference_llmflow.wav" type="audio/wav">  </audio> <br> **文本内容**：有任何问题都可以来找我，我可是提瓦特最棒的、最厉害的、知道的最多的向导！ |

说实话，关于派蒙的语音，仅`llm`就比较像，这个是将训练后的`llm.pt`替换原始的`llm.pt`，其他的不变，而这样的推理结果比较像原始音频的主要原因大概率是预训练阶段就有派蒙的数据，因此模型早就记住了派蒙的语音特征，这点在零样本推理中尤为明显，我们使用某个音频作为参照，用零样本推理生成一段派蒙语音对比下。

其中零样本推理使用的是完全没有微调过的原始模型，SFT则是将`llm`和`flow`都微调过后的结果。下面依次展示原始训练数据中的派蒙语音和零样本推理、微调音频效果对照表。

<table style="width: 100%; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="padding: 12px 15px; text-align: center; border: 1px solid #e0e0e0; background: #f8f8f8; font-weight: 600; min-width: 150px; font-size: 16px;">原始数据编号</th>
      <th style="padding: 12px 15px; text-align: left; border: 1px solid #e0e0e0; background: #f8f8f8; font-weight: 600; min-width: 585px; font-size: 16px; ">音频效果演示 & 游戏文本</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 12px 15px; text-align: center; border: 1px solid #e0e0e0; background: #fff; font-size: 16px;">1_4</td>
      <td style="padding: 12px 15px;  border: 1px solid #e0e0e0; word-wrap: break-word; word-break: break-word; background: #fff;">
        <audio controls style="width: 100%; max-width: 400px; margin-bottom: 8px;">
          <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/paimon_inference/train_data/1_4.wav" type="audio/wav">
        </audio> <br>
        <strong>文本内容</strong>：说起来刚刚就一直听到「阿贝多先生」，他也是蒙德城的炼金术士吗？
      </td>
    </tr>
    <tr>
      <td style="padding: 12px 15px; text-align: center; border: 1px solid #e0e0e0; background: #fafafa; font-size: 16px;">1_12</td>
      <td style="padding: 12px 15px; border: 1px solid #e0e0e0; word-wrap: break-word; word-break: break-word; background: #fafafa;">
        <audio controls style="width: 100%; max-width: 400px; margin-bottom: 8px;">
          <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/paimon_inference/train_data/1_12.wav" type="audio/wav">
        </audio> <br>
        <strong>文本内容</strong>：没什么没什么，只是平时他总是站在这里，有点奇怪而已。
      </td>
    </tr>
  </tbody>
</table>

---

<!-- 外层容器：控制滚动+基础样式（兼容大部分平台） -->
<div style="overflow-x: auto; max-width: 100%; border: 1px solid #cdccccff;">
<!-- 标题嵌入容器内 -->
  <h3 style="padding: 12px 16px; margin: 0; text-align: center; color: #333; border-bottom: 1px solid #787878ff;">零样本&微调音频效果对照表</h3>
  <!-- 表格：标准化写法，避免渲染异常 -->
  <table style="width: 100%; border-collapse: collapse; min-width: 800px;">
    <!-- 表头 -->
    <thead>
      <tr style="background: #f5f5f5;">
        <th style="padding: 8px 12px; text-align: left; border: 1px solid #e0e0e0;">Text</th>
        <th style="padding: 8px 12px; text-align: left; border: 1px solid #e0e0e0;">Zero-Shot</th>
        <th style="padding: 8px 12px; text-align: left; border: 1px solid #e0e0e0;">SFT(llm+flow)</th>
      </tr>
    </thead>
    <!-- 表体 -->
    <tbody>
      <tr>
        <td style="padding: 8px 12px; border: 1px solid #e0e0e0;word-wrap: break-word; word-break: break-all; max-width: 230px;">卖唱的怎么又跑去喝酒了！</td>
        <td style="padding: 8px 12px; border: 1px solid #e0e0e0;"><audio controls style="width: 280px;"> <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/paimon_inference/zero-shot/zero_shot_0.wav" type="audio/wav">  </audio></td>
        <td style="padding: 8px 12px; border: 1px solid #e0e0e0;"><audio controls style="width: 280px;"> <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/paimon_inference/llm_flow/paimon_sft_inference_0.wav" type="audio/wav">  </audio></td>
      </tr>
      <tr>
        <td style="padding: 8px 12px; border: 1px solid #e0e0e0;word-wrap: break-word; word-break: break-all; max-width: 230px;">你知道昨天那个新闻吗？</td>
        <td style="padding: 8px 12px; border: 1px solid #e0e0e0;"><audio controls style="width: 280px;"> <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/paimon_inference/zero-shot/zero_shot_1.wav" type="audio/wav">  </audio></td>
        <td style="padding: 8px 12px; border: 1px solid #e0e0e0;"><audio controls style="width: 280px;"> <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/paimon_inference/llm_flow/paimon_sft_inference_1.wav" type="audio/wav">  </audio></td>
      </tr>
      <tr>
        <td style="padding: 8px 12px; border: 1px solid #e0e0e0;word-wrap: break-word; word-break: break-all; max-width: 230px;">音频大模型是用海量声音数据训练、能一次性听懂生成说话音乐噪声的“万能声学大脑”。</td>
        <td style="padding: 8px 12px; border: 1px solid #e0e0e0;"><audio controls style="width: 280px;"> <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/paimon_inference/zero-shot/zero_shot_2.wav" type="audio/wav">  </audio></td>
        <td style="padding: 8px 12px; border: 1px solid #e0e0e0;"><audio controls style="width: 280px;"> <source src="https://github.com/828Tina/cosyvoice-paimon-sft/raw/main/examples/audios/paimon_inference/llm_flow/paimon_sft_inference_2.wav" type="audio/wav">  </audio></td>
      </tr>
    </tbody>
  </table>
</div>

以上三段文字都是我写的自定义文本：
- 第一条是贴合原神派蒙会说的话（当然不排除可能在游戏里真的说过😂，如果有，可以当作用训练数据作为参考）；
- 第二条表示疑问的语气；
- 第三条直接完全脱离原神游戏里的内容，以陈述句形式来展示效果

可以从实际效果长听出，零样本推理其实已经具备了一点派蒙的声线和说话习惯，但是相比于真实声线，其实还是差很多的。而微调过后的模型，对于派蒙声音的模仿就要好得多，如此也可以证明微调的效果。

> 如果想要更真实的数据支撑推理的效果，可以参考👉[https://github.com/FunAudioLLM/CV3-Eval](https://github.com/FunAudioLLM/CV3-Eval)，这是官方发布的针对CosyVoice3的模型推理效果评测，应该对CosyVoice2也可以用，我之后有时间研究下，这里先放个未完待续...


## 参考文献

[1]. [https://developer.nvidia.com/cuda/gpus](https://developer.nvidia.com/cuda/gpus)

[2]. [https://github.com/FunAudioLLM/CosyVoice/tree/main/examples/libritts/cosyvoice2](https://github.com/FunAudioLLM/CosyVoice/tree/main/examples/libritts/cosyvoice2) 

[3]. [https://funaudiollm.github.io/pdf/CosyVoice_v1.pdf](https://funaudiollm.github.io/pdf/CosyVoice_v1.pdf)

[4]. [https://arxiv.org/abs/2412.10117](https://arxiv.org/abs/2412.10117)