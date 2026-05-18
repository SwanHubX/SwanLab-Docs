# swanlab local

``` bash
swanlab local
```

## 介绍

在命令行中将SwanLab快速设置为`local`模式。

## 优先级

SwanLab提供了三种设置`local`模式的方式：

1. 命令行

```bash
swanlab local
```

2. 环境变量

```bash
export SWANLAB_MODE=local
```

3. 代码

```python
swanlab.init(mode="local")
```

其中优先级排序：代码 > 环境变量 > 命令行。

## 设置为其他模式

云端模式：

```bash
swanlab online
```

禁用模式：

```bash
swanlab disabled
```

离线模式：

```bash
swanlab offline
```