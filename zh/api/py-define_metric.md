# define_metric

```python
define_metric(
    key: str,
    *,
    x_axis: Optional[str] = None,
    section_name: Optional[str] = None,
    hidden: bool = False,
    step_sync: Optional[bool] = None,
    overwrite: bool = False,
    **kwargs: Any,
) -> None
```

除 `key` 外的参数均为关键字参数；`**kwargs` 仅用于兼容 `step_metric` 别名（其余未知参数会被静默忽略）。

| 参数         | 描述                                                                                                                                                                                                                  |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| key          | 指标名，支持精确 key 或末尾带单个 `*` 的 glob 匹配（如 `"train/*"`，单独的 `"*"` 匹配所有自定义指标）。`"*loss"`、`"train/*/x"`、`"train/**"` 等写法会被拒绝（仅给出告警，该条定义不生效）；系统指标 key 不参与匹配。 |
| x_axis       | 自定义 X 轴的指标 key。为 `None` 时使用系统 step 作为 X 轴。兼容 `step_metric` 参数别名。                                                                                                                             |
| section_name | 图表分组名。为 `None` 时使用默认分组（按指标名前缀分组）。                                                                                                                                                            |
| hidden       | 是否隐藏。为 `True` 时，匹配指标的图表会被放入 HIDDEN 分组。默认为 `False`。                                                                                                                                          |
| step_sync    | X 轴与 Y 轴指标分开 log 时，自动为 Y 值补上最近一次的 X 值，传入 `x_axis`/`step_metric` 时默认开启；当前版本强制开启，显式传 `False` 会被忽略并给出告警。                                                             |
| overwrite    | `False` 时与已有定义合并；为 `True` 时将未指定的字段重置为默认值。默认为 `False`。仅影响**未被 log 过**的 key。                                                                                                       |

## 简介

`swanlab.define_metric` 用于在 `swanlab.log` 之前定义指标的图表行为，包括：

- **自定义 X 轴**：用另一个指标（如 epoch）作为 X 轴，而不是默认的 step
- **图表分组**：将指标图表归入自定义分组
- **隐藏图表**：将不常用的图表放入 HIDDEN 分组

## 自定义 X 轴

将 `train/loss` 的 X 轴设置为 `train/epoch`：

```python
import swanlab

swanlab.init(project="my-project")

swanlab.define_metric("train/loss", x_axis="train/epoch")

for epoch in range(num_epochs):
    swanlab.log({"train/epoch": epoch})
    # ... 训练 ...
    swanlab.log({"train/loss": loss})  # 自动同步 train/epoch 的值
```

X 轴指标和 Y 轴指标可以分开记录，`step_sync` 会自动给 Y 值补上最近一次的 X 值。建议每轮**先 log X 轴指标、再 log Y 轴指标**；如果 X 值比 Y 值晚记录，图表会沿用之前自动补上的 X 值，并给出告警提示。

`x_axis` 必须是合法的指标 key（或系统值 `"_step"`、`"_relative_time"`），且不能是系统指标 key；校验失败时整条定义不生效并给出报错。

自定义 X 轴假设**单调递增**：同一 X 值只保留首个 Y 点（实现上仅抑制连续重复的 X 值）。如果 X 非单调（如 5→6→5），回退的 X 值会被当作新值接受，同一 X 值上可能出现多个 Y 点。

## glob 批量定义

`key` 支持末尾带单个 `*` 的 glob 匹配，可批量定义一组指标：

```python
# 将 val/ 下的所有指标放入 "Validation" 分组
swanlab.define_metric("val/*", section_name="Validation")
```

当一个指标同时命中多条规则时：精确匹配的 key 优先，其次是前缀更长的 glob，最后才是默认行为。例如：

```python
swanlab.define_metric("train/stage/acc", x_axis="x1")
swanlab.define_metric("train/*", x_axis="x2")
swanlab.define_metric("train/stage/*", x_axis="x3")
```

- `train/stage/acc` 的 X 轴为 `x1`（精确匹配，最具体）
- `train/stage/` 下的其他指标 X 轴为 `x3`（`train/stage/*` 比 `train/*` 更具体）
- `train/` 下的其余指标 X 轴为 `x2`

## 隐藏图表

```python
# 图表放入 HIDDEN 分组，默认不在看板中显示
swanlab.define_metric("debug/grad_norm", hidden=True)
```

被隐藏的指标数据仍会正常记录和上传，只是图表被折叠进 HIDDEN 分组，可随时在WebUI看板中取消。

需要注意：在默认的合并模式（`overwrite=False`）下，`hidden` 是「sticky」的——只要有一次设为 `True`，结果就是 `True`，之后传 `hidden=False` 与未提供等效，无法把它改回 `False`。如需清除之前设置的 `hidden=True`，请使用 `overwrite=True`（未指定的字段重置为默认值，`hidden` 回到 `False`）。

## 合并与覆盖

对同一个 key 多次调用 `define_metric` 时：

- `overwrite=False`（默认）：在已有定义的基础上合并，只更新这次指定的字段
- `overwrite=True`：覆盖，此次没指定的字段全部回到默认值

```python
swanlab.define_metric("train/loss", x_axis="train/epoch")
swanlab.define_metric("train/loss", section_name="Train")  # 合并：x_axis 保留，新增分组
swanlab.define_metric("train/loss", section_name="Train", overwrite=True)  # 覆盖：x_axis 重置为默认 step
```

合并和覆盖只对**还没有被 log 过**的 key 有效。一旦指标被 log、图表已经创建，再调用 `define_metric` 不会生效，只能通过 WebUI 调整。

## 注意事项

1. 同一项目下，同一个指标只对应一张图表。例如 run1 中首次 `define_metric("train/loss", x_axis="x1")` 生效后，同项目的 run2 再定义 `x_axis="x2"` 不会生效。
2. 图片、音频等多媒体指标会忽略 `x_axis`（仍按 step 展示），只应用 `section_name` 和 `hidden`。
3. 通过 `define_metric` 图表定义只在「当前默认视图」和「复制视图」中生效，执行「复制/移动实验」、「添加对比实验」、「新建默认视图」时，分组关系会保留，但自定义的 X 轴关联会丢失，X 轴回退为默认的 step。
