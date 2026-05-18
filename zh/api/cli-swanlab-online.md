# swanlab online

``` bash
swanlab online
```

## 介绍

在命令行中将SwanLab快速设置为`cloud`模式。

## 优先级

SwanLab提供了三种设置`cloud`模式的方式：

1. 命令行

```bash
swanlab online
```

2. 环境变量

```bash
export SWANLAB_MODE=cloud
```

3. 代码

```python
swanlab.init(mode="cloud")
```

其中优先级排序：代码 > 环境变量 > 命令行。

## 设置为其他模式

离线模式：

```bash
swanlab offline
```

禁用模式：

```bash
swanlab disabled
```

离线看板模式：

```bash
swanlab local
```