# swanlab.login

``` bash
login(
    api_key: str
    ):
```

| Parameter | Description |
| --- | --- |
| `api_key` | If already logged in, force a re-login. |

## Introduction

Log in to your SwanLab account to synchronize experiments to the cloud. [API_KEY Retrieval Address](#)

```python
import swanlab

swanlab.login(api_key='your-api-key')

```

After logging in once, the credentials are saved locally, so you do not need to log in again via `swanlab.login` or `swanlab login`.