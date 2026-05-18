# swanlab local

```bash
swanlab local
```

## Introduction

Quickly set SwanLab to `local` mode in the command line.

## Priority

SwanLab provides three ways to set the `local` mode:

1. Command line

```bash
swanlab local
```

2. Environment variable

```bash
export SWANLAB_MODE=local
```

3. Code

```python
swanlab.init(mode="local")
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

Offline mode:

```bash
swanlab offline
```