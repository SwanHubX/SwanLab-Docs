# swanlab offline

```bash
swanlab offline
```

## Introduction

Quickly set SwanLab to `offline` mode in the command line.

## Priority

SwanLab provides three ways to set the `offline` mode:

1. Command line

```bash
swanlab offline
```

2. Environment variable

```bash
export SWANLAB_MODE=offline
```

3. Code

```python
swanlab.init(mode="offline")
```

The priority order is: Code > Environment variable > Command line.

## Set to other modes

Cloud mode:

```bash
swanlab online
```

Disabled mode:

```bash
swanlab disabled
```

Local dashboard mode:

```bash
swanlab local
```