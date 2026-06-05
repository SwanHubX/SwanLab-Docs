# swanlab verify

```bash
swanlab verify [OPTIONS]
```

| Option | Description |
| --- | --- |
| `--local` | Verify local login status only (checks for `.swanlab` in the current directory). |

## Introduction

Verify the current login status. If logged in, it displays the server address and current username; if not logged in, it exits with an error message.

```bash
swanlab verify
```

Successful output example:

```
You are logged into https://api.swanlab.cn as my_username
```

If not logged in:

```
You are not logged in. Use `swanlab login` to login.
```

## Verify Local Login

Use the `--local` option to verify the local login status in the current directory:

```bash
swanlab verify --local
```
