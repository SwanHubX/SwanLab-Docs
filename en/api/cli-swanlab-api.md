# swanlab api

```bash
swanlab api [OPTIONS] COMMAND ARGS [ARGS]
```

`swanlab api` is the CLI implementation of the [OpenAPI](./py-api.md), allowing direct manipulation of cloud **workspace / project / experiment / user / self-hosted** resources from the command line.

> Authentication follows the same logic as the Python API: pass `--api-key` / `--host` explicitly, or use the `swanlab login` local session. See [Authentication](./py-api.md#authentication).

## Subcommand Overview

### workspace — Workspace

| Subcommand | Description |
|------------|-------------|
| `swanlab api workspace info <username>` | Get workspace info |

### project — Project

| Subcommand | Description |
|------------|-------------|
| `swanlab api project info <path>` | Get project info |
| `swanlab api project list` | List projects under a workspace |
| `swanlab api project create` | Create a project |

### run — Experiment

| Subcommand | Description |
|------------|-------------|
| `swanlab api run info <path>` | Get experiment info |
| `swanlab api run list` | List experiments under a project |
| `swanlab api run filter` | Filter experiments by query |
| `swanlab api run metrics` | Get experiment scalar metrics |
| `swanlab api run summary` | Get experiment metric summaries |
| `swanlab api run column` | Get a single experiment column |
| `swanlab api run columns` | List experiment columns |
| `swanlab api run medias` | Get experiment media metrics |
| `swanlab api run logs` | Get experiment console logs |
| `swanlab api run export-logs` | Export experiment logs as a file |

### user — User

| Subcommand | Description |
|------------|-------------|
| `swanlab api user info` | Get current user info |

### self-hosted — Self-Hosted Management

> Requires super admin privileges.

| Subcommand | Description |
|------------|-------------|
| `swanlab api self-hosted info` | Get instance info |
| `swanlab api self-hosted create-user` | Create a user |
| `swanlab api self-hosted list-users` | List all users |
| `swanlab api self-hosted list-projects` | List all projects |
| `swanlab api self-hosted summary` | System usage summary |
| `swanlab api self-hosted list-workspaces` | List all workspaces |

## Common Options

All `swanlab api` subcommands support:

| Option | Description |
|--------|-------------|
| `-h`, `--host` | SwanLab server host address |
| `-k`, `--api-key` | API key; takes precedence over local login |
| `--save` | Save output as JSON file; pass a filename or omit for auto-generated name |


## Subcommand Reference

### workspace info

Get detailed information about a workspace.

```bash
swanlab api workspace info <username> [OPTIONS]
```

| Argument/Option | Type | Description |
|-----------------|------|-------------|
| `username` | Positional | Workspace username (unique ID) |
| `--save` | Option | Save output as JSON file; pass a filename or omit for auto-generated name |

```bash
# View workspace info
swanlab api workspace info my-team

# Save to file
swanlab api workspace info my-team --save workspace.json
```


### project info

Get detailed information about a project.

```bash
swanlab api project info <path> [OPTIONS]
```

| Argument/Option | Type | Description |
|-----------------|------|-------------|
| `path` | Positional | Project path, format: `username/project-name` |
| `--save` | Option | Save output as JSON file |

```bash
swanlab api project info my-team/image-classification
```


### project list

List all projects under a workspace.

```bash
swanlab api project list [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--workspace` | `str` | Current logged-in user | Workspace username |
| `--page_num` / `-n` | `int` | `1` | Page number |
| `--page_size` / `-s` | `str` | `"20"` | Page size; valid values: `"10"`, `"20"`, `"50"`, `"100"` |
| `--all` | Flag | `False` | Auto-paginate, fetch all projects |
| `--save` | Option | — | Save output as JSON file |

```bash
# List projects for current workspace
swanlab api project list

# List projects for a specific workspace, 50 per page
swanlab api project list --workspace my-team --page_size 50

# Fetch all projects
swanlab api project list --workspace my-team --all
```


### project create

Create a new project in a workspace.

```bash
swanlab api project create [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-n` / `--name` | `str` | Required | Project name (1–100 chars, `0-9a-zA-Z-_.+` only) |
| `-v` / `--visibility` | `str` | `"PRIVATE"` | Visibility: `PUBLIC` or `PRIVATE` |
| `-d` / `--description` | `str` | `None` | Project description |
| `-w` / `--workspace` | `str` | Current logged-in user | Workspace username |
| `--save` | Option | — | Save output as JSON file |

```bash
# Create a private project
swanlab api project create -n my-project -v PRIVATE

# Create a public project in a specific workspace
swanlab api project create -n my-project -v PUBLIC -w my-team -d "Image classification experiments"
```


### run info

Get detailed information about an experiment.

```bash
swanlab api run info <path> [OPTIONS]
```

| Argument/Option | Type | Description |
|-----------------|------|-------------|
| `path` | Positional | Experiment path, format: `username/project-name/experiment-id` |
| `--save` | Option | Save output as JSON file |

```bash
swanlab api run info my-team/image-classification/abc123
```


### run list

List all experiments under a project.

```bash
swanlab api run list [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--project_path` / `-p` | `str` | Required | Project path, format: `username/project-name` |
| `--page_num` / `-n` | `int` | `1` | Page number |
| `--page_size` / `-s` | `str` | `"20"` | Page size |
| `--all` | Flag | `False` | Auto-paginate, fetch all experiments |
| `--save` | Option | — | Save output as JSON file |

```bash
# List experiments in a project
swanlab api run list -p my-team/image-classification

# Fetch all experiments
swanlab api run list -p my-team/image-classification --all
```


### run filter

Query experiments by filter conditions. Use a JSON array to specify one or more filter rules.

```bash
swanlab api run filter --project_path <path> --filter_query <json> [OPTIONS]
```

| Option | Type | Description |
|--------|------|-------------|
| `--project_path` / `-p` | `str` | Required, project path |
| `--filter_query` / `-f` | `str` | Required, filter conditions (JSON array or path to JSON file) |
| `--save` | Option | Save output as JSON file |

Each filter rule is a JSON object:

```json
{
  "key": "<field name>",
  "type": "STABLE | CONFIG | SCALAR",
  "op": "EQ | NEQ | GT | LT | GTE | LTE | CONTAIN | NOT_CONTAIN",
  "value": ["<value>"]
}
```

`type` value reference:

| type | key values | value description |
|------|-----------|-------------------|
| `STABLE` | `state`, `name`, `description`, `show`, `pin`, `labels`, `createdAt`, `updatedAt`, `finishedAt`, etc. | The corresponding field value |
| `CONFIG` | Config parameter name (e.g. `param_2`, **no** `config.` prefix) | Config parameter value |
| `SCALAR` | Scalar metric name (e.g. `loss`) | Metric value |

```bash
# Query finished experiments
swanlab api run filter \
  -p my-team/image-classification \
  -f '[{"key": "state", "type": "STABLE", "op": "EQ", "value": ["FINISHED"]}]'

# Read filter conditions from file
swanlab api run filter -p my-team/image-classification -f ./filter.json
```


### run metrics

Get scalar metrics for an experiment, returned as JSON.

```bash
swanlab api run metrics <path> --keys <keys> [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `path` | Positional | Required | Experiment path |
| `--keys` | `str` | Required | Comma-separated metric keys, e.g. `"loss,acc"` |
| `--sample` / `-s` | `int` | `1500` | Sample size; auto-capped if exceeded |
| `--ignore-timestamp` | Flag | `False` | Remove timestamp field from metric data |
| `--all` | Flag | `False` | Fetch full data (CSV export for scalars) |
| `--range-type` | `str` | `None` | Range query type: `step` or `timestamp` |
| `--range-start` | `int` | `None` | Range start (inclusive), step number or unix timestamp in ms |
| `--range-end` | `int` | `None` | Range end (inclusive), step number or unix timestamp in ms |
| `--range-head` | `int` | `None` | Return first N data points |
| `--range-tail` | `int` | `None` | Return last N data points |
| `--save` | Option | — | Save output as JSON file |

### RangeQuery

`--range-head` and `--range-tail` are mutually exclusive. `--range-start` and `--range-end` work with `--range-type` (`step` or `timestamp`); timestamps are in milliseconds.

```bash
# Get loss metric (default 1500 samples)
swanlab api run metrics my-team/image-classification/abc123 --keys loss

# Get multiple metrics, 500 samples
swanlab api run metrics my-team/image-classification/abc123 --keys loss,acc -s 500

# Range query by step
swanlab api run metrics my-team/image-classification/abc123 \
  --keys loss --range-type step --range-start 100 --range-end 500

# Get last 200 data points
swanlab api run metrics my-team/image-classification/abc123 \
  --keys loss --range-tail 200
```


### run summary

Get scalar metric summaries for an experiment (final value, min, max, etc.).

```bash
swanlab api run summary <path> [OPTIONS]
```

| Option | Type | Description |
|--------|------|-------------|
| `path` | Positional | Experiment path |
| `--keys` | `str` | Comma-separated metric keys; omit for all keys |
| `--save` | Option | Save output as JSON file |

```bash
# Get summary for all metrics
swanlab api run summary my-team/image-classification/abc123

# Get summary for specific metrics
swanlab api run summary my-team/image-classification/abc123 --keys loss,acc
```


### run column

Get a single metric column for an experiment.

```bash
swanlab api run column <path> --key <key> [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `path` | Positional | Required | Experiment path |
| `--key` | `str` | Required | Column key name |
| `--class` | `str` | `"CUSTOM"` | Column class: `CUSTOM` or `SYSTEM` |
| `--type` | `str` | `None` | Column data type: `FLOAT`, `STRING`, `IMAGE`, etc. |
| `--save` | Option | — | Save output as JSON file |

```bash
swanlab api run column my-team/image-classification/abc123 --key loss
```


### run columns

List all metric columns for an experiment.

```bash
swanlab api run columns <path> [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `path` | Positional | Required | Experiment path |
| `--page_num` / `-n` | `int` | `1` | Page number |
| `--page_size` / `-s` | `str` | `"20"` | Page size |
| `--class` | `str` | `"CUSTOM"` | Column class filter |
| `--type` | `str` | `None` | Column data type filter |
| `--all` | Flag | `False` | Auto-paginate, fetch all columns |
| `--save` | Option | — | Save output as JSON file |

```bash
# List all columns
swanlab api run columns my-team/image-classification/abc123

# List SYSTEM FLOAT columns only
swanlab api run columns my-team/image-classification/abc123 \
  --class SYSTEM --type FLOAT --all
```


### run medias

Get media metrics (images, audio, etc.) for an experiment. Returns presigned URLs.

```bash
swanlab api run medias <path> --keys <keys> [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `path` | Positional | Required | Experiment path |
| `--keys` | `str` | Required | Comma-separated media keys, e.g. `"image,audio"` |
| `--step` / `-s` | `int` | `0` | Step number |
| `--all` | Flag | `False` | Fetch all steps |
| `--save` | Option | — | Save output as JSON file |

```bash
# Get image metric at step 0
swanlab api run medias my-team/image-classification/abc123 --keys generated_image

# Get all audio steps
swanlab api run medias my-team/image-classification/abc123 --keys audio --all
```


### run logs

Get console logs for an experiment.

```bash
swanlab api run logs <path> [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `path` | Positional | Required | Experiment path |
| `--offset` / `-o` | `int` | `0` | Log shard index |
| `--level` / `-l` | `str` | `"INFO"` | Log level: `DEBUG`, `INFO`, `WARN`, `ERROR` |
| `--ignore-timestamp` | Flag | `False` | Remove timestamp field |
| `--save` | Option | — | Save output as JSON file |

```bash
# Get INFO level logs
swanlab api run logs my-team/image-classification/abc123

# Get WARN and above
swanlab api run logs my-team/image-classification/abc123 --level WARN

# Get specific shard
swanlab api run logs my-team/image-classification/abc123 --offset 1
```


### run export-logs

Export experiment logs as a downloadable `.log` file.

```bash
swanlab api run export-logs <path> [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `path` | Positional | Required | Experiment path |
| `--start` | `int` | `0` | Start row index (0-based) |
| `--rows` / `-r` | `int` | `500000` | Number of rows to export, max 500000 |
| `--save` | Option | — | Save output as JSON file (includes download URL) |

```bash
# Export first 10000 rows
swanlab api run export-logs my-team/image-classification/abc123 --rows 10000
```


### user info

Get current logged-in user information.

```bash
swanlab api user info [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--save` | Save output as JSON file |

```bash
swanlab api user info
```


### self-hosted info

Get self-hosted instance information.

```bash
swanlab api self-hosted info [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--save` | Save output as JSON file |

```bash
swanlab api self-hosted info
```


### self-hosted create-user

Create a new user in the self-hosted instance (super admin only).

```bash
swanlab api self-hosted create-user [OPTIONS]
```

| Option | Type | Description |
|--------|------|-------------|
| `-u` / `--username` | `str` | Required, new username |
| `-p` / `--password` | `str` | Required, new user password |
| `--save` | Option | Save output as JSON file |

```bash
swanlab api self-hosted create-user -u testuser -p test123456
```


### self-hosted list-users

List all users in the self-hosted instance (super admin only).

```bash
swanlab api self-hosted list-users [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--page_num` / `-n` | `int` | `1` | Page number |
| `--page_size` / `-s` | `int` | `20` | Page size |
| `--all` | Flag | `False` | Auto-paginate, fetch all users |
| `--save` | Option | — | Save output as JSON file |

```bash
# List all users
swanlab api self-hosted list-users --all
```


### self-hosted list-projects

List all projects in the self-hosted instance (super admin only).

```bash
swanlab api self-hosted list-projects [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--page_num` / `-n` | `int` | `1` | Page number |
| `--page_size` / `-s` | `int` | `20` | Page size |
| `--all` | Flag | `False` | Auto-paginate, fetch all projects |
| `--search` | `str` | `None` | Search keyword |
| `--creator` | `str` | `None` | Filter by creator username |
| `--workspace` | `str` | `None` | Filter by workspace username |
| `--save` | Option | — | Save output as JSON file |

```bash
# List all projects
swanlab api self-hosted list-projects --all

# Search for projects
swanlab api self-hosted list-projects --search image --all
```


### self-hosted summary

Get system usage summary for the self-hosted instance (super admin only).

```bash
swanlab api self-hosted summary [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--save` | Save output as JSON file |

```bash
swanlab api self-hosted summary
```


### self-hosted list-workspaces

List all workspaces in the self-hosted instance (super admin only).

```bash
swanlab api self-hosted list-workspaces [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--page_num` / `-n` | `int` | `1` | Page number |
| `--page_size` / `-s` | `int` | `20` | Page size |
| `--all` | Flag | `False` | Auto-paginate, fetch all workspaces |
| `--search` | `str` | `None` | Search keyword |
| `--save` | Option | — | Save output as JSON file |

```bash
# List all workspaces
swanlab api self-hosted list-workspaces --all
```
