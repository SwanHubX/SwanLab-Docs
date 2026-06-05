# save

::: warning 仅私有化版本可见
此 API 仅在 SwanLab **私有化部署版本**中可用，公有云版本不支持。
:::

```python
save(
    glob_str: Union[str, bytes, Path],
    base_path: Optional[Union[str, Path]] = None,
    policy: Literal["now", "end", "live"] = "live",
) -> List[str]
```

| 参数 | 描述 |
|------|------|
| glob_str | (Union[str, bytes, Path]) 要保存文件的 glob 匹配模式，例如 ``"checkpoints/*.pt"``。 |
| base_path | (Optional[Union[str, Path]]) 用于解析相对路径的基准目录。默认为当前工作目录。 |
| policy | (Literal["now", "end", "live"]) 保存策略：``"now"`` 立即上传；``"end"`` 延迟到实验结束后上传；``"live"`` 监听文件变化并自动重新上传。默认为 ``"live"``。 |

## 简介

`swanlab.save` 允许通过 glob 模式匹配文件并保存到当前实验中。适用于在训练过程中保存模型检查点、日志或其他产物。

该函数返回匹配到的文件相对路径列表。

## 使用方法

```python
import swanlab

swanlab.init(project="my-project", mode="local")

# 保存 checkpoints 目录下所有 .pt 文件
saved = swanlab.save("checkpoints/*.pt")
print(saved)  # ['checkpoints/epoch_1.pt', 'checkpoints/epoch_2.pt']

swanlab.finish()
```

## 保存策略

| 策略 | 行为 |
|------|------|
| `now` | 立即上传匹配的文件。 |
| `end` | 延迟到实验结束后上传。 |
| `live` | 监听文件变化并自动重新上传。（默认） |

### 示例：训练结束时保存

```python
import swanlab

swanlab.init(project="my-project")

# 训练模型...
# 训练结束时保存最终检查点
swanlab.save("output/model_final.pt", policy="end")

swanlab.finish()
```

### 示例：实时监听检查点

```python
import swanlab

swanlab.init(project="my-project")

# 保存并监听文件变化 — 适用于长时间训练
swanlab.save("checkpoints/*.pt", policy="live")

# ... 训练循环 ...
swanlab.finish()
```

## 注意事项

1. `glob_str` 模式相对于 `base_path`（或当前工作目录）解析。
2. 仅保存普通文件 — 目录会被自动过滤。
3. 单次调用保存的文件数量有上限（由 `save_batch` 配置项控制）。
