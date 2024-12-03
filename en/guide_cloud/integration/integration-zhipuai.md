# ZhipuAI

[zhipuai](https://github.com/MetaGLM/zhipuai-sdk-python-v4) is the Python SDK for the [Zhipu Open Platform](https://open.bigmodel.cn/dev/api) large model interface, making it easier for developers to call the Zhipu Open API.

![](/assets/integration-zhipu.jpg)

You can use zhipuai to get responses from ChatGLM and automatically log the process using SwanLab.

## 1. Introducing autolog

```python
from swanlab.integration.zhipuai import autolog
```

`autolog` is a process logging class adapted for zhipuai, which can automatically log your zhipuai interaction process.

## 2. Passing Parameters

```python
autolog(init=dict(project="zhipuai_logging"))
```

The parameters passed to `init` here are exactly the same as those for `swanlab.init`.

## 3. Automatic Logging

```python
from swanlab.integration.zhipuai import autolog

autolog(init=dict(project="zhipuai_logging"))
client = autolog.client

response = client.chat.completions.create(
    model="glm-4",
    messages=[
        {"role": "user", "content": "作为一名营销专家，请为我的产品创作一个吸引人的slogan"},
        {"role": "assistant", "content": "当然，为了创作一个吸引人的slogan，请告诉我一些关于您产品的信息"},
        {"role": "user", "content": "智谱AI开放平台"},
        {"role": "assistant", "content": "智启未来，谱绘无限一智谱AI，让创新触手可及!"},
        {"role": "user", "content": "创造一个更精准、吸引人的slogan"},
    ],
)

response2 = client.chat.completions.create(
    model="glm-4",
    messages=[
        {"role": "user", "content": "谁获得了NBA2015年的总冠军"},
    ],
)
```