# OpenAI

[openai](https://github.com/openai/openai-python)是ChatGPT在Python环境下使用的核心库。

![openai](/assets/ig-openai.png)

你可以使用openai获得ChatGPT的回复，同时使用SwanLab自动进行过程记录。


## 1. 引入autolog

```python
from swanlab.integration.openai import autolog
```

`autolog`是一个为openai适配的过程记录类，能够自动记录你的openai交互的过程。

## 2. 传入参数

```python
autolog(init={"project":"openai_autologging", "experiment_name":"chatgpt4.0"})
```

这里给`init`传入的参数与`swanlab.init`的参数形式完全一致。

## 3. 自动记录

由于`openai`在1.0.0版本以后，采用了和先前不一样的API设计，所以下面分为两个版本：

### openai>=1.0.0

需要将`client=openai.OpenAI()`替换为`client=autolog.client`。

```python
from swanlab.integration.openai import autolog

autolog(init=dict(experiment_name="openai_autologging"))
client = autolog.client

# chat_completion
response = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[
        {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
        {"role": "user", "content": "Who won the world series in 2015?"},
    ],
)

# text_completion
response2 = client.completions.create(model="gpt-3.5-turbo-instruct", prompt="Write a song for jesus.")
```

### openai<=0.28.0

```python
import openai
from swanlab.integration.openai import autolog

autolog(init=dict(experiment_name="openai_logging123"))

chat_request_kwargs = dict(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers"},
        {"role": "user", "content": "Where was it played?"},
    ],
)
response = openai.ChatCompletion.create(**chat_request_kwargs)
```