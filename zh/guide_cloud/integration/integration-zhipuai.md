# ZhipuAI

[zhipuai](https://github.com/MetaGLM/zhipuai-sdk-python-v4)是[智谱开放平台](https://open.bigmodel.cn/dev/api) 大模型接口的Python SDK，让开发者更便捷的调用智谱开放API。

![](/assets/integration-zhipu.jpg)

你可以使用zhipuai获得ChatGLM的回复，同时使用SwanLab自动进行过程记录。

## 1. 引入autolog

```python
from swanlab.integration.zhipuai import autolog
```

autolog是一个为zhipuai适配的过程记录类，能够自动记录你的zhipuai交互的过程。

## 2. 传入参数

```python
autolog(init=dict(project="zhipuai_logging"))
```

这里给`init`传入的参数与`swanlab.init`的参数形式完全一致。


## 3. 自动记录

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