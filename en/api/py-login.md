# swanlab.login

```bash
login(
    api_key: str = None,
    host: str = None,
    web_host: str = None,
    save: bool = False
):
```

| Parameter | Description |
| --- | --- |
| `api_key` | (str) Authentication key. If not provided, the key will be read from the key file. |
| `host` | (str) The API host where the SwanLab service is located. If not provided, the default host (i.e., the cloud version) will be used. |
| `web_host` | (str) The web host where the SwanLab service is located. If not provided, the default host (i.e., the cloud version) will be used. |
| `save` | (bool) Whether to save the API key to the key file. The default value is False. |

## Introduction

Log in to your SwanLab account in Python code to upload experiments to the specified cloud server. The API Key can be obtained from the "Settings" - "General" page of your SwanLab account.

## Logging in to the Public Cloud

```python
import swanlab

swanlab.login(api_key='your-api-key', save=True)
```

By default, this will log in to `swanlab.cn`, the public cloud service of SwanLab.

If you need to log in to another host, you can specify the `host` parameter, such as `http://localhost:8000`.

Setting the `save` parameter to `True` will save the login credentials locally (overwriting any previously saved credentials), eliminating the need to log in again via `swanlab.login` or `swanlab login`.

**If you are using a public machine, set the `save` parameter to `False`** to avoid leaking your API Key and prevent others from accidentally uploading data to your space.

## Logging in to a Private Service

```python
swanlab.login(api_key='your-api-key', host='your-private-host')
```