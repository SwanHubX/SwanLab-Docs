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

All parameters after `key` are keyword-only; `**kwargs` exists only for the `step_metric` compatibility alias (any other unknown parameters are silently ignored).

| Parameter    | Description                                                                                                                                                                                                                                                                                                    |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| key          | Metric key. Supports an exact key or a glob with a single trailing `*` (e.g. `"train/*"`; a bare `"*"` matches all custom metrics). Patterns like `"*loss"`, `"train/*/x"` and `"train/**"` are rejected (a warning is emitted and the definition does not take effect); system metric keys are never matched. |
| x_axis       | Metric key used as the custom X axis. `None` uses the system step as the X axis. Accepts `step_metric` as a parameter alias.                                                                                                                                                                                   |
| section_name | Chart section (group) name. `None` uses the default grouping derived from the key prefix.                                                                                                                                                                                                                      |
| hidden       | Whether to hide the chart. When `True`, charts of the matched metrics are placed in the HIDDEN section. Defaults to `False`.                                                                                                                                                                                   |
| step_sync    | When X-axis and Y-axis metrics are logged separately, automatically fills each Y value with the most recent X value. Enabled by default when `x_axis`/`step_metric` is set; set to `False` to disable filling for this metric.                                                                                 |
| overwrite    | `False` merges with the existing definition; `True` resets unspecified fields to their defaults. Defaults to `False`. Only affects keys that have **not been logged yet**.                                                                                                                                     |

## Introduction

`swanlab.define_metric` defines the chart behavior of metrics before `swanlab.log`, including:

- **Custom X axis**: use another metric (e.g. epoch) as the X axis instead of the default step
- **Chart grouping**: assign metric charts to a custom section
- **Hide charts**: put rarely used charts into the HIDDEN section

## Custom X axis

Set the X axis of `train/loss` to `train/epoch`:

```python
import swanlab

swanlab.init(project="my-project")

swanlab.define_metric("train/loss", x_axis="train/epoch")

for epoch in range(num_epochs):
    swanlab.log({"train/epoch": epoch})
    # ... training ...
    swanlab.log({"train/loss": loss})  # automatically syncs the value of train/epoch
```

X-axis and Y-axis metrics can be logged separately — `step_sync` automatically fills each Y value with the most recent X value. We recommend logging the X-axis metric **before** the Y-axis metric in each round. If the X value is logged after the Y value, the chart keeps the auto-filled X value and a warning is emitted.

`x_axis` must be a valid metric key (or the system values `"_step"` / `"_relative_time"`) and must not be a system metric key; if validation fails, the whole definition is aborted with an error.

A custom X axis is assumed to be **monotonically non-decreasing**: only the first Y point is kept for a given X value (in practice, only consecutively duplicated X values are dropped). With a non-monotonic X (e.g. 5→6→5), a rolled-back X value is accepted as new, so multiple Y points may appear at the same X value.

## Batch definition with glob

`key` supports a glob with a single trailing `*`, so you can define a group of metrics at once:

```python
# Put all metrics under val/ into the "Validation" section
swanlab.define_metric("val/*", section_name="Validation")
```

When a metric matches multiple rules: an exact key match wins first, then the glob with the longest prefix, and finally the default behavior. For example:

```python
swanlab.define_metric("train/stage/acc", x_axis="x1")
swanlab.define_metric("train/*", x_axis="x2")
swanlab.define_metric("train/stage/*", x_axis="x3")
```

- `train/stage/acc` uses `x1` as its X axis (exact match, most specific)
- Other metrics under `train/stage/` use `x3` (`train/stage/*` is more specific than `train/*`)
- The remaining metrics under `train/` use `x2`

## Hiding charts

```python
# The chart goes into the HIDDEN section and is not shown on the dashboard by default
swanlab.define_metric("debug/grad_norm", hidden=True)
```

Hidden metric data is still recorded and uploaded as usual — the chart is simply folded into the HIDDEN section, and you can unhide it anytime in the WebUI dashboard.

Note: in the default merge mode (`overwrite=False`), `hidden` is "sticky" — once set to `True` the result stays `True`, and passing `hidden=False` later is indistinguishable from not providing it. To clear a previously set `hidden=True`, use `overwrite=True` (unspecified fields reset to their defaults, so `hidden` returns to `False`).

## Merge vs. overwrite

When calling `define_metric` multiple times for the same key:

- `overwrite=False` (default): merge on top of the existing definition, updating only the fields specified this time
- `overwrite=True`: overwrite — every field not specified this time returns to its default value

```python
swanlab.define_metric("train/loss", x_axis="train/epoch")
swanlab.define_metric("train/loss", section_name="Train")  # merge: x_axis kept, section added
swanlab.define_metric("train/loss", section_name="Train", overwrite=True)  # overwrite: x_axis resets to the default step
```

Merging and overwriting only work for keys that have **not been logged yet**. Once a metric has been logged and its chart created, further `define_metric` calls take no effect — adjust the chart in the WebUI instead.

## Notes

1. Within the same project, each metric corresponds to only one chart. For example, once the first `define_metric("train/loss", x_axis="x1")` in run1 takes effect, defining `x_axis="x2"` in run2 of the same project will not take effect.
2. Media metrics such as images and audio ignore `x_axis` (they are still displayed by step); only `section_name` and `hidden` are applied to them.
3. Chart definitions made via `define_metric` only take effect in the **current default view** and **copied views**. When performing "copy/move experiment", "add comparison experiment", or "create default view", the section grouping is preserved, but custom X-axis associations are lost and the X axis falls back to the default step.
