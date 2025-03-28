# 音频分类

:::info
音频分类、音频处理入门
:::

[![](/assets/badge1.svg)](https://swanlab.cn/@ZeyiLin/PyTorch\_Audio\_Classification-simple/charts)

音频分类任务是指将音频信号按照其内容的类别归属进行划分。例如，区分一段音频是音乐、语音、环境声音（如鸟鸣、雨声、机器运转声）还是动物叫声等。其目的是通过自动分类的方式，高效地对大量音频数据进行组织、检索和理解。

![alt text](/assets/examples/audio_classification/example-audio-classification-1.png)

在现在音频分类的应用场景，比较多的是在音频标注、音频推荐这一块。同时，这也是一个非常好的入门音频模型训练的任务。

在本文中，我们会基于PyTorch框架，使用 ResNet系列模型在 GTZAN 数据集上进行训练，同时使用[SwanLab](https://swanlab.cn)监控训练过程、评估模型效果。

* Github：[https://github.com/Zeyi-Lin/PyTorch-Audio-Classification](https://github.com/Zeyi-Lin/PyTorch-Audio-Classification)
* 数据集：[https://pan.baidu.com/s/14CTI_9MD1vXCqyVxmAbeMw?pwd=1a9e](https://pan.baidu.com/s/14CTI_9MD1vXCqyVxmAbeMw?pwd=1a9e) 提取码: 1a9e
* SwanLab实验日志：[https://swanlab.cn/@ZeyiLin/PyTorch\_Audio\_Classification-simple/charts](https://swanlab.cn/@ZeyiLin/PyTorch\_Audio\_Classification-simple/charts)
* 更多实验日志：[https://swanlab.cn/@ZeyiLin/PyTorch\_Audio\_Classification/charts](https://swanlab.cn/@ZeyiLin/PyTorch\_Audio\_Classification/charts)



## 1. 音频分类逻辑

本教程对音频分类任务的逻辑如下：

1. 载入音频数据集，数据集为音频WAV文件与对应的标签
2. 以8:2的比例划分训练集和测试集
3. 使用`torchaudio`库，将音频文件转换为梅尔频谱图，本质将其转换为图像分类任务
4. 使用ResNet模型对梅尔频谱图进行训练迭代
5. 使用SwanLab记录训练和测试阶段的loss、acc变化，并对比不同实验之间的效果差异

## 2. 环境安装

本案例基于**Python>=3.8**，请在您的计算机上安装好Python。

我们需要安装以下这几个Python库：

```python
torch
torchvision
torchaudio
swanlab
pandas
scikit-learn
```

一键安装命令：

```shellscript
pip install torch torchvision torchaudio swanlab pandas scikit-learn
```



## 3. GTZAN数据集准备

本任务使用的数据集为GTZAN，这是一个在音乐流派识别研究中常用的公开数据集。GTZAN数据集包含 1000 个音频片段，每个音频片段的时长为 30 秒，共分为 10 种音乐流派：包括布鲁斯（Blues）、古典（Classical）、乡村（Country）、迪斯科（Disco）、嘻哈（Hip Hop）、爵士（Jazz）、金属（Metal）、流行（Pop）、雷鬼（Reggae）、摇滚（Rock），且每种流派都有 100 个音频片段。

![alt text](/assets/examples/audio_classification/example-audio-classification-2.png)

GTZAN数据集是在 2000-2001 年从各种来源收集的，包括个人 CD、收音机、麦克风录音等，代表了各种录音条件下的声音。

**数据下载方式（大小1.4GB）：**

1. 百度网盘下载：链接: [https://pan.baidu.com/s/14CTI\_9MD1vXCqyVxmAbeMw?pwd=1a9e](https://pan.baidu.com/s/14CTI\_9MD1vXCqyVxmAbeMw?pwd=1a9e) 提取码: 1a9e
2. 通过Kaggle下载：[https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
3. 在Hyper超神经网站下载BT种子进行下载：[https://hyper.ai/cn/datasets/32001](https://hyper.ai/cn/datasets/32001)

> 注意，数据集中有一个音频是损坏的，在百度网盘版本里已经将其剔除。

下载完成后，解压到项目根目录下即可。



## 4. 生成数据集CSV文件

我们将数据集中的音频文件路径和对应的标签，处理成一个`audio_dataset.csv`文件，其中第一列为文件路径，第二列为标签：

（这一部分先不执行，在完整代码里会带上）

```python
import os
import pandas as pd

def create_dataset_csv():
    # 数据集根目录
    data_dir = './GTZAN/genres_original'
    data = []
    
    # 遍历所有子目录
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            # 遍历子目录中的所有wav文件
            for audio_file in os.listdir(label_dir):
                if audio_file.endswith('.wav'):
                    audio_path = os.path.join(label_dir, audio_file)
                    data.append([audio_path, label])
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(data, columns=['path', 'label'])
    df.to_csv('audio_dataset.csv', index=False)
    return df


# 生成或加载数据集CSV文件
if not os.path.exists('audio_dataset.csv'):
    df = create_dataset_csv()
else:
    df = pd.read_csv('audio_dataset.csv')
```

处理后，你会在根目录下看到一个`audio_dataset.csv`文件：

![alt text](/assets/examples/audio_classification/example-audio-classification-3.png)


## 5. 配置训练跟踪工具SwanLab

SwanLab 是一款开源、轻量的 AI 实验跟踪工具，提供了一个跟踪、比较、和协作实验的平台。SwanLab 提供了友好的 API 和漂亮的界面，结合了超参数跟踪、指标记录、在线协作、实验链接分享等功能，让您可以快速跟踪 AI 实验、可视化过程、记录超参数，并分享给伙伴。

![alt text](/assets/examples/audio_classification/example-audio-classification-4.png)

配置SwanLab的方式很简单：

1. 注册一个账号：[https://swanlab.cn](https://swanlab.cn)
2. 在安装好swanlab后（pip install swanlab），登录：

```bash
swanlab login
```

在提示输入API Key时，去[设置页面](https://swanlab.cn/settings/overview)复制API Key，粘贴后按回车即可。

![alt text](/assets/examples/audio_classification/example-audio-classification-5.png)



## 6. 完整代码

开始训练时的目录结构：

```
|--- train.py
|--- GTZAN
```

train.py做的事情包括：

1. 生成数据集csv文件
2. 加载数据集和resnet18模型（ImageNet预训练）
3. 训练20个epoch，每个epoch进行训练和评估
4. 记录loss和acc，以及学习率的变化情况，在swanlab中可视化



train.py：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import swanlab


def create_dataset_csv():
    # 数据集根目录
    data_dir = './GTZAN/genres_original'
    data = []
    
    # 遍历所有子目录
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            # 遍历子目录中的所有wav文件
            for audio_file in os.listdir(label_dir):
                if audio_file.endswith('.wav'):
                    audio_path = os.path.join(label_dir, audio_file)
                    data.append([audio_path, label])
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(data, columns=['path', 'label'])
    df.to_csv('audio_dataset.csv', index=False)
    return df

# 自定义数据集类
class AudioDataset(Dataset):
    def __init__(self, df, resize, train_mode=True):
        self.audio_paths = df['path'].values
        # 将标签转换为数值
        self.label_to_idx = {label: idx for idx, label in enumerate(df['label'].unique())}
        self.labels = [self.label_to_idx[label] for label in df['label'].values]
        self.resize = resize
        self.train_mode = train_mode  # 添加训练模式标志
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        # 加载音频文件
        waveform, sample_rate = torchaudio.load(self.audio_paths[idx])
        
        # 将音频转换为梅尔频谱图
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=640,
            n_mels=128
        )
        mel_spectrogram = transform(waveform)

        # 确保数值在合理范围内
        mel_spectrogram = torch.clamp(mel_spectrogram, min=0)
        
        # 转换为3通道图像格式 (为了适配ResNet)
        mel_spectrogram = mel_spectrogram.repeat(3, 1, 1)
        
        # 确保尺寸一致
        resize = torch.nn.AdaptiveAvgPool2d((self.resize, self.resize))
        mel_spectrogram = resize(mel_spectrogram)
        
        return mel_spectrogram, self.labels[idx]

# 修改ResNet模型
class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()
        # 加载预训练的ResNet
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # 修改最后的全连接层
        self.resnet.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss/len(train_loader)
        train_acc = 100.*correct/total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss/len(val_loader)
        val_acc = 100.*correct/total
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录训练和验证指标
        swanlab.log({
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "train/epoch": epoch,
            "train/lr": current_lr
        })
            
        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')

# 主函数
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    run = swanlab.init(
        project="PyTorch_Audio_Classification-simple",
        experiment_name="resnet18",
        config={
            "batch_size": 16,
            "learning_rate": 1e-4,
            "num_epochs": 20,
            "resize": 224,
        },
    )
    
    # 生成或加载数据集CSV文件
    if not os.path.exists('audio_dataset.csv'):
        df = create_dataset_csv()
    else:
        df = pd.read_csv('audio_dataset.csv')
    
    # 划分训练集和验证集
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        label_train, label_val = train_test_split(label_df, test_size=0.2, random_state=42)
        train_df = pd.concat([train_df, label_train])
        val_df = pd.concat([val_df, label_val])
    
    # 创建数据集和数据加载器 
    train_dataset = AudioDataset(train_df, resize=run.config.resize, train_mode=True)
    val_dataset = AudioDataset(val_df, resize=run.config.resize, train_mode=False)
    
    train_loader = DataLoader(train_dataset, batch_size=run.config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # 创建模型
    num_classes = len(df['label'].unique())  # 根据实际分类数量设置
    print("num_classes", num_classes)
    model = AudioClassifier(num_classes).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=run.config.learning_rate)  
    
    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=run.config.num_epochs, device=device)

if __name__ == "__main__":
    main()
```

看到下面的输出，则代表训练开始：

![alt text](/assets/examples/audio_classification/example-audio-classification-6.png)

访问打印的swanlab链接，可以看到训练的全过程：

![alt text](/assets/examples/audio_classification/example-audio-classification-7.png)

可以看到Reset18模型，且不加任何策略的条件下，在训练集的准确率为99.5%，验证集的准确率最高为71.5%，val loss在第3个epoch开始反而在上升，呈现「过拟合」的趋势。



## 7. 进阶代码

下面是我训出验证集准确率87.5%的实验，具体策略包括：

1. 将模型换成resnext101\_32x8d
2. 将梅尔顿图的resize提高到512
3. 增加warmup策略
4. 增加时间遮蔽、频率屏蔽、高斯噪声、随机响度这四种数据增强策略
5. 增加学习率梯度衰减策略

![alt text](/assets/examples/audio_classification/example-audio-classification-8.png)

进阶代码（需要24GB显存，如果要降低显存消耗的话，可以调低batch\_size）：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import swanlab
import random
import numpy as np

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dataset_csv():
    # 数据集根目录
    data_dir = './GTZAN/genres_original'
    data = []
    
    # 遍历所有子目录
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            # 遍历子目录中的所有wav文件
            for audio_file in os.listdir(label_dir):
                if audio_file.endswith('.wav'):
                    audio_path = os.path.join(label_dir, audio_file)
                    data.append([audio_path, label])
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(data, columns=['path', 'label'])
    df.to_csv('audio_dataset.csv', index=False)
    return df

# 自定义数据集类
class AudioDataset(Dataset):
    def __init__(self, df, resize, train_mode=True):
        self.audio_paths = df['path'].values
        # 将标签转换为数值
        self.label_to_idx = {label: idx for idx, label in enumerate(df['label'].unique())}
        self.labels = [self.label_to_idx[label] for label in df['label'].values]
        self.resize = resize
        self.train_mode = train_mode  # 添加训练模式标志
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        # 加载音频文件
        waveform, sample_rate = torchaudio.load(self.audio_paths[idx])
        
        # 将音频转换为梅尔频谱图
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=640,
            n_mels=128
        )
        mel_spectrogram = transform(waveform)
        
        # 仅在训练模式下进行数据增强
        if self.train_mode:
            # 1. 时间遮蔽 (Time Masking)：通过随机选择一个时间步，然后遮蔽掉20个时间步
            time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)
            mel_spectrogram = time_mask(mel_spectrogram)
            
            # 2. 频率遮蔽 (Frequency Masking)：通过随机选择一个频率步，然后遮蔽掉20个频率步
            freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
            mel_spectrogram = freq_mask(mel_spectrogram)
            
            # 3. 随机增加高斯噪声
            if random.random() < 0.5:
                noise = torch.randn_like(mel_spectrogram) * 0.01
                mel_spectrogram = mel_spectrogram + noise
            
            # 4. 随机调整响度
            if random.random() < 0.5:
                gain = random.uniform(0.8, 1.2)
                mel_spectrogram = mel_spectrogram * gain

        # 确保数值在合理范围内
        mel_spectrogram = torch.clamp(mel_spectrogram, min=0)
        
        # 转换为3通道图像格式 (为了适配ResNet)
        mel_spectrogram = mel_spectrogram.repeat(3, 1, 1)
        
        # 确保尺寸一致
        resize = torch.nn.AdaptiveAvgPool2d((self.resize, self.resize))
        mel_spectrogram = resize(mel_spectrogram)
        
        return mel_spectrogram, self.labels[idx]

# 修改ResNet模型
class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()
        # 加载预训练的ResNet
        self.resnet = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
        # 修改最后的全连接层
        self.resnet.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, run):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 前5个epoch进行warmup
        if epoch < 5:
            warmup_factor = (epoch + 1) / 5
            for param_group in optimizer.param_groups:
                param_group['lr'] = run.config.learning_rate * warmup_factor
        
        # optimizer.zero_grad()  # 移到循环外部
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss
        train_acc = 100.*correct/total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss/len(val_loader)
        val_acc = 100.*correct/total
        
        # 只在warmup结束后使用学习率调度器
        if epoch >= 5:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录训练和验证指标
        swanlab.log({
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "train/epoch": epoch,
            "train/lr": current_lr
        })
            
        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')

# 主函数
def main():
    # 设置随机种子
    set_seed(42)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    run = swanlab.init(
        project="PyTorch_Audio_Classification-simple",
        experiment_name="😄resnext101_32x8d",
        config={
            "batch_size": 16,
            "learning_rate": 1e-4,
            "num_epochs": 30,
            "resize": 512,
            "weight_decay": 0  # 添加到配置中
        },
    )
    
    # 生成或加载数据集CSV文件
    if not os.path.exists('audio_dataset.csv'):
        df = create_dataset_csv()
    else:
        df = pd.read_csv('audio_dataset.csv')
    
    # 划分训练集和验证集
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        label_train, label_val = train_test_split(label_df, test_size=0.2, random_state=42)
        train_df = pd.concat([train_df, label_train])
        val_df = pd.concat([val_df, label_val])
    
    # 创建数据集和数据加载器 
    train_dataset = AudioDataset(train_df, resize=run.config.resize, train_mode=True)
    val_dataset = AudioDataset(val_df, resize=run.config.resize, train_mode=False)
    
    train_loader = DataLoader(train_dataset, batch_size=run.config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # 创建模型
    num_classes = len(df['label'].unique())  # 根据实际分类数量设置
    model = AudioClassifier(num_classes).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=run.config.learning_rate,
        weight_decay=run.config.weight_decay
    )  # Adam优化器
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,  # 在第10个epoch衰减
        gamma=0.1,     # 衰减率为0.1
        verbose=True
    )
    
    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=run.config.num_epochs, device=device, run=run)
    

if __name__ == "__main__":
    main()

```

![alt text](/assets/examples/audio_classification/example-audio-classification-9.png)

可以看到提升的非常明显

> 期待有训练师能把eval acc刷上90！



## 8. 相关链接

* Github：[https://github.com/Zeyi-Lin/PyTorch-Audio-Classification](https://github.com/Zeyi-Lin/PyTorch-Audio-Classification)
* 数据集：[https://pan.baidu.com/s/14CTI_9MD1vXCqyVxmAbeMw?pwd=1a9e](https://pan.baidu.com/s/14CTI_9MD1vXCqyVxmAbeMw?pwd=1a9e) 提取码: 1a9e
* SwanLab实验日志：[https://swanlab.cn/@ZeyiLin/PyTorch\_Audio\_Classification-simple/charts](https://swanlab.cn/@ZeyiLin/PyTorch\_Audio\_Classification-simple/charts)
* 更多实验日志：[https://swanlab.cn/@ZeyiLin/PyTorch\_Audio\_Classification/charts](https://swanlab.cn/@ZeyiLin/PyTorch\_Audio\_Classification/charts)
* SwanLab官网：[https://swanlab.cn](https://swanlab.cn)