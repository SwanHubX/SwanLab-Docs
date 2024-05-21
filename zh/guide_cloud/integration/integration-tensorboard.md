# Tensorboard

[TensorBoard](https://github.com/tensorflow/tensorboard) 是 Google TensorFlow 提供的一个可视化工具，用于帮助理解、调试和优化机器学习模型。它通过图形界面展示训练过程中的各种指标和数据，让开发者更直观地了解模型的性能和行为。

![TensorBoard](/assets/ig-tensorboard.png)

你可以使用`swanlab convert`将Tensorboard生成的Tfevent文件转换成SwanLab实验。

## 方式一：命令行转换

```bash
swanlab convert -t tensorboard -tb_logdir [TFEVENT_LOGDIR]
```

这里的`[TFEVENT_LOGDIR]`是指你先前用Tensorboard记录实验时，生成的日志文件路径。

SwanLab Converter将会自动检测文件路径及其子目录下的`tfevent`文件（默认子目录深度为3），并为每个`tfevent`文件创建一个SwanLab实验。

## 方式二：代码内转换

```python
from swanlab.converter import TFBConverter

tfb_converter = TFBConverter(convert_dir="[TFEVENT_LOGDIR]")
tfb_converter.run()
```

效果与命令行转换一致。

## 参数列表

| 参数 | 对应CLI参数       | 描述                  | 
| ---- | ---------- | --------------------- | 
| convert_dir    | -      | Tfevent文件路径       | 
| project    | -p, --project      | SwanLab项目名       |
| workspace  | -w, --workspace      | SwanLab工作空间名 |
| cloud    | --cloud      | 是否使用云端版，默认为True       | 
| logdir    | -l, --logdir      | SwanLab日志文件保存路径       | 

例子：

```python
from swanlab.converter import TFBConverter

tfb_converter = TFBConverter(
    convert_dir="./runs",
    project="Tensorboard-Converter",
    workspace="SwanLab",
    logdir="./logs",
    )
tfb_converter.run()
```

与之作用相同的CLI：
```bash
swanlab convert -t tensorboard --tb_logdir ./runs -p Tensorboard-Converter -w SwanLab -l ./logs
```

执行上面的脚本，将会在`SwanLab`空间下，创建一个名为`Tensorboard-Converter`的项目，将`./runs`目录下tfevent文件创建为一个个swanlab实验，并将swanlab运行时产生的日志保存在`./logs`目录下。