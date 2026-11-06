# swanlab.Histogram

[GitHub Source Code](https://github.com/SwanHubX/SwanLab/blob/feat/histogram/swanlab/sdk/internal/run/transforms/histogram/__init__.py)

```python
Histogram(
    sequence=None,
    bins=None,
    *,
    np_histogram=None,
    **kwargs,
)
```

Convert samples into bucket counts and use [`swanlab.log()`](./py-log.md) to record a distribution snapshot at each step. This is useful for tracking changes in model parameter or gradient distributions.

| Parameter      | Description                                                                                                                                                                                                                          |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `sequence`     | Raw samples as a one-dimensional sequence of real numbers, such as a `list`, `tuple`, `range`, or NumPy array. PyTorch tensors and existing `Histogram` instances are also accepted. Required when `np_histogram` is not provided.   |
| `bins`         | A bucket count or bucket edges. Counts accept Python / NumPy integers from 1 to 512; edges accept a one-dimensional sequence of real numbers. Defaults to `None`, which uses 64 equal-width buckets.                                 |
| `np_histogram` | Precomputed `(counts, bin_edges)`, including the return value of `numpy.histogram()`. Defaults to `None`. When provided, `sequence`, `bins`, and `num_bins` are ignored.                                                             |
| `num_bins`     | Bucket count parameter for W&B compatibility, passed as a keyword argument. Accepts Python / NumPy integers from 1 to 512 and takes effect only when `bins` is not given. When both are given, `bins` wins and a warning is emitted. |

`np_histogram` and `num_bins` are keyword-only parameters. Other unrecognized keyword arguments are ignored by the constructor.

## Create from Raw Samples

When given a bucket count, the SDK uses `numpy.histogram()` to generate equal-width buckets over the sample range:

```python
import numpy as np
import swanlab

swanlab.init(project="histogram-demo")

rng = np.random.default_rng(42)
for step in range(10):
    samples = rng.normal(loc=step * 0.1, size=1000)
    swanlab.log({"weights": swanlab.Histogram(samples, bins=64)}, step=step)

swanlab.finish()
```

Each `Histogram` computes its buckets independently, so edges may differ between steps. To keep edges fixed, specify them explicitly as shown below.

### Specify Bucket Edges

`bins` supports unequal-width buckets. N buckets require N+1 edges:

```python
import swanlab

hist = swanlab.Histogram(
    [0.0, 0.5, 1.0, 2.0, 3.0],
    bins=[0.0, 1.0, 3.0],
)

print(hist.values)  # [2, 3]
print(hist.bins)  # [0.0, 1.0, 3.0]
```

Each bucket covers `[bins[i], bins[i+1])`, except the last bucket, which includes its right edge. Samples outside the explicit edge range are discarded with a warning.

## Create from Precomputed Buckets

Pass an existing `numpy.histogram()` result through `np_histogram`. The SDK validates and uses the result directly:

```python
import numpy as np
import swanlab

swanlab.init(project="histogram-demo")

samples = [0.0, 0.5, 1.0, 2.0, 3.0]
counts, bin_edges = np.histogram(samples, bins=[0.0, 1.0, 3.0])
hist = swanlab.Histogram(np_histogram=(counts, bin_edges))
swanlab.log({"weights": hist}, step=0)

swanlab.finish()
```

You can also provide count and edge sequences directly, for example `swanlab.Histogram(np_histogram=([2, 3], [0.0, 1.0, 3.0]))`.

## Log PyTorch Tensors

PyTorch tensors can be passed directly. The SDK converts them automatically with `detach().cpu()`, so tensors on GPU, tensors with gradients, and `bfloat16` tensors are all supported. Flatten multidimensional parameters to one dimension first:

```python
import torch
import swanlab

swanlab.init(project="histogram-demo")

layer = torch.nn.Linear(8, 4)
swanlab.log({"weights": swanlab.Histogram(layer.weight.flatten())}, step=0)

swanlab.finish()
```

## Data Rules and Exceptions

- The bucket count must be between 1 and 512. This limit also applies to explicit edges and precomputed input.
- `NaN`, `+Inf`, and `-Inf` in raw samples are discarded with a warning. With an integer bucket count, an empty sequence or a sequence with no finite values raises `ValueError`. Valid explicit edges allow a histogram with all counts equal to 0.
- Edges must be a one-dimensional sequence of finite real numbers in nondecreasing order. Duplicate edges are accepted with a warning about zero-width buckets.
- Precomputed counts must be one-dimensional, finite, nonnegative integer values no greater than `2**64 - 1`. Integer-valued floats such as `1.0` are converted to integers. Probability densities and fractional weighted counts are not supported.
- Precomputed input must satisfy `len(bin_edges) == len(counts) + 1`.
- `swanlab.Histogram(hist)` reuses an existing instance's buckets. In this case, do not specify `bins`, `num_bins`, or `np_histogram`.

Inputs with an unsupported type or shape do not raise an exception: the instance is marked invalid, the error details are written to the experiment's debug log, and `swanlab.log()` skips that histogram, with the terminal only reporting that the metric produced no data. Passing an unsupported object, non-numeric data, a multidimensional array, or a boolean bucket count all follow this path.

Violating the numeric and length rules listed above raises `ValueError`. Calling without both `sequence` and `np_histogram`, or specifying bucket parameters when nesting a `Histogram` instance, also raises `ValueError`.

## Access Bucket Results

A constructed `Histogram` exposes the following attributes:

| Attribute  | Type          | Description                                                                                    |
| ---------- | ------------- | ---------------------------------------------------------------------------------------------- |
| `values`   | `list[int]`   | Sample counts for each bucket, with length N.                                                  |
| `bins`     | `list[float]` | Bucket edges, with length N+1.                                                                 |
| `is_valid` | `bool`        | Whether the input passed type and shape validation. Invalid instances are skipped when logged. |

Logged data preserves the counts and edges at each step; the original samples cannot be recovered. Use a separate metric name for histograms and avoid mixing scalars, histograms, or other data types under the same name.
