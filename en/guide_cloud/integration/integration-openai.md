# OpenAI

[openai](https://github.com/openai/openai-python) is the core library for using ChatGPT in the Python environment.

![openai](/assets/ig-openai.png)

You can use openai to get responses from ChatGPT while using SwanLab to automatically log the process.

## 1. Import autolog

```python
from swanlab.integration.openai import autolog
```

`autolog` is a process logging class adapted for openai, which can automatically log your openai interaction process.

## 2. Pass Parameters

```python
autolog(init={"project":"openai_autologging", "experiment_name":"chatgpt4.0"})
```

The parameters passed to `init` here are exactly the same as those for `swanlab.init`.

## 3. Automatic Logging

Since `openai` adopted a different API design after version 1.0.0, the following is divided into two versions:

### openai>=1.0.0

You need to replace `client=openai.OpenAI()` with `client=autolog.client`.

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