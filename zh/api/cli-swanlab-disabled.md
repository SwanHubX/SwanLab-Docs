# swanlab disabled

``` bash
swanlab disabled
```

## 介绍

在命令行中将SwanLab快速设置为`disabled`模式。

## 优先级

SwanLab提供了三种设置`disabled`模式的方式：

1. 命令行

```bash
swanlab disabled
```

2. 环境变量

```bash
export SWANLAB_MODE=disabled
```

3. 代码

```python
swanlab.init(mode="disabled")
```

其中优先级排序：代码 > 环境变量 > 命令行。

## 设置为其他模式

离线模式：

```bash
swanlab offline
```

云端模式：

```bash
swanlab online
```

离线看板模式：

```bash
swanlab local
```