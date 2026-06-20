# save

::: warning Private Deployment Only
This API is only available in the **self-hosted (private deployment)** version of SwanLab. It is not supported on the public cloud version.
:::

```python
save(
    glob_str: Union[str, bytes, Path],
    base_path: Optional[Union[str, Path]] = None,
    policy: Literal["now", "end", "live"] = "live",
) -> List[str]
```

| Parameter | Description                                                                                                                                                               |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| glob_str  | (Union[str, bytes, Path]) Glob pattern for files to save, e.g. `"checkpoints/*.pt"`.                                                                                      |
| base_path | (Optional[Union[str, Path]]) Base directory for resolving relative paths. Defaults to current working directory.                                                          |
| policy    | (Literal["now", "end", "live"]) Save policy: `"now"` upload immediately; `"end"` defer until run finishes; `"live"` watch and re-upload on changes. Defaults to `"live"`. |

## Introduction

`swanlab.save` allows you to save files matched by a glob pattern into the current run. This is useful for persisting model checkpoints, logs, or any other artifacts during training.

The function returns a list of relative file paths that were matched and saved.

## Usage

```python
import swanlab

swanlab.init(project="my-project", mode="local")

# Save all .pt files in the checkpoints directory
saved = swanlab.save("checkpoints/*.pt")
print(saved)  # ['checkpoints/epoch_1.pt', 'checkpoints/epoch_2.pt']

swanlab.finish()
```

## Save Policies

| Policy | Behavior                                                      |
| ------ | ------------------------------------------------------------- |
| `now`  | Upload matched files immediately.                             |
| `end`  | Defer upload until the run finishes.                          |
| `live` | Watch for file changes and re-upload automatically. (default) |

### Example: Save at the End of Training

```python
import swanlab

swanlab.init(project="my-project")

# Train your model...
# Save the final checkpoint when training ends
swanlab.save("output/model_final.pt", policy="end")

swanlab.finish()
```

### Example: Live Watch for Checkpoints

```python
import swanlab

swanlab.init(project="my-project")

# Save and watch for changes — useful during long training runs
swanlab.save("checkpoints/*.pt", policy="live")

# ... training loop ...
swanlab.finish()
```

## Notes

1. The `glob_str` pattern is resolved relative to `base_path` (or current working directory if not specified).
2. Only regular files are saved — directories are automatically filtered out.
3. There is a limit on the number of files that can be saved in a single call (controlled by `save_batch` setting).
