# swanlab login

```bash
swanlab login [OPTIONS]
```

| Option | Description |
| --- | --- |
| `-r`, `--relogin` | Re-login. |
| `-h`, `--host` | Specify the host where the SwanLab service is located. For example, `http://localhost:8000`. |
| `-k`, `--api-key` | Specify the API Key. This allows automatic login if you prefer not to enter the API key via the command line. |
| `-w`, `--web-host` | Specify the web host where the SwanLab frontend is located. |

## Introduction

Log in to your SwanLab account to synchronize experiments to the cloud.

After executing the following command, if it's your first time logging in, you will be prompted to enter your [API_KEY](https://swanlab.cn/settings):

```bash
swanlab login
```

After logging in once, the credentials will be saved locally, overwriting any previously saved credentials, eliminating the need to log in again via `swanlab.login` or `swanlab login`.

> If you do not want the credentials to be saved locally, use [swanlab.login()](./py-login.md) in a Python script to log in.

If your computer is not suitable for entering the API Key via the command line (e.g., some Windows CMD), you can use:

```bash
swanlab login -k <api-key>
```

## Re-login

If you need to log in with a different account, use the following command:

```bash
swanlab login --relogin
```

This will prompt you to enter a new API Key to log in again.

## Logout

```bash
swanlab logout
```

## Logging in to a Private Service

```bash
swanlab login --host <host>
```