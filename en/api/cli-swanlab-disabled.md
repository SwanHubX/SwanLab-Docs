# swanlab disabled

```bash
swanlab disabled
```

## Introduction

Quickly set SwanLab to `disabled` mode in the command line.

## Priority

SwanLab provides three ways to set the `disabled` mode:

1. Command line

```bash
swanlab disabled
```

2. Environment variable

```bash
export SWANLAB_MODE=disabled
```

3. Code

```python
swanlab.init(mode="disabled")
```

The priority order is: Code > Environment variable > Command line.

## Set to other modes

Offline mode:

```bash
swanlab offline
```

Cloud mode:

```bash
swanlab online
```

Local dashboard mode:

```bash
swanlab local
```