# swanlab logout

```bash
swanlab logout [OPTIONS]
```

| Option          | Description                                                                         |
| --------------- | ----------------------------------------------------------------------------------- |
| `-f`, `--force` | Force logout without confirmation prompt.                                           |
| `--local`       | Logout from local login, removing `.swanlab/` credentials in the current directory. |

## Introduction

Logout from the currently logged-in SwanLab account:

```bash
swanlab logout
```

A confirmation prompt will appear. Enter `y` to confirm logout.

## Force Logout

In non-interactive environments (e.g., CI/CD), use `--force` to skip confirmation:

```bash
swanlab logout --force
```

## Local Logout

If you logged in locally using `swanlab login --local`, you also need to specify `--local` when logging out:

```bash
swanlab logout --local
```
