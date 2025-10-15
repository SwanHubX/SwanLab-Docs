# swanlab online

```bash
swanlab online
```

## Introduction

Quickly set SwanLab to `cloud` mode in the command line.

## Priority

SwanLab provides three ways to set the `cloud` mode:

1. Command line

```bash
swanlab online
```

2. Environment variable

```bash
export SWANLAB_MODE=cloud
```

3. Code

```python
swanlab.init(mode="cloud")
```

The priority order is: Code > Environment variable > Command line.

## Set to other modes

Offline mode:

```bash
swanlab offline
```

Disabled mode:

```bash
swanlab disabled
```

Local dashboard mode:

```bash
swanlab local
```