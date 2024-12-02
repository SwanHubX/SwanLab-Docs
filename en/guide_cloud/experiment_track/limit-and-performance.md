# Limitations and Performance

## Optimize Metric Logging

Using `swanlab.log` to track and record experiment metrics, after recording, these metrics will generate charts and display in the table. When the amount of recorded data is too large, it may slow down web access.

### Suggestion 1: Keep the Total Number of Different Metrics Below 10,000

Recording more than 10k different metric names may slow down the rendering of your dashboard and table operations.

For media data, try to record related media data under the same metric name:

```python
# ❌ Not recommended
for i, img in enumerate(images):
    swanlab.log({f"pred_img_{i}": swanlab.Image(image)})

# ✅ Recommended
swanlab.log({"pred_imgs": [swanlab.Image(image) for image in images]})
```

<br>

### Suggestion 2: Keep the Metric Width Below 10 Million

The metric width, in line charts with step as the x-axis, refers to the range difference between the minimum and maximum values of step.

When the metric width is too large, it affects the drawing load time of all metrics in the experiment, leading to slow access.

<br>

### Suggestion 3: Limit the Submission Frequency of Metrics

Choose an appropriate logging frequency for the metrics you are recording. Empirically, the wider the metric, the lower the frequency of logging it.

Specifically, we recommend:

- Scalars: Less than `50k` data points per metric
- Media data: Less than `10k` data points per metric

If you exceed these guidelines, SwanLab will continue to accept your recorded data, but page loading speed may be slow.

The recommended logging method is shown in the following code:

```python
# For example, there are 1 million loops
for step in range(1000000):
    ....

    # Submit once every 1k loops, effectively reducing the submission frequency of metrics
    if step % 1000 == 0:
        swanlab.log({"scalar": step})
```