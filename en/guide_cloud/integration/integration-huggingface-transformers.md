# 🤗HuggingFace Transformers

Hugging Face's [Transformers](https://github.com/huggingface/transformers) is a very popular open-source library that provides a large number of pre-trained models, primarily for natural language processing (NLP) tasks. The goal of this library is to make the latest models easy to use and support multiple frameworks such as TensorFlow and PyTorch.

![hf-transformers-image](/assets/ig-huggingface-transformers.png)

You can use Transformers to quickly train models while using SwanLab for experiment tracking and visualization.

## 1. Import SwanLabCallback

```python
from swanlab.integration.huggingface import SwanLabCallback
```

**SwanLabCallback** is a logging class adapted for Transformers.

**SwanLabCallback** can define parameters such as:

- `project`, `experiment_name`, `description`, and other parameters consistent with `swanlab.init`, used for initializing the SwanLab project.
- You can also create the project externally via `swanlab.init`, and the integration will log the experiment to the project you created externally.

## 2. Pass to Trainer

```python (1,7,12)
from swanlab.integration.huggingface import SwanLabCallback
from transformers import Trainer, TrainingArguments

...

# Instantiate SwanLabCallback
swanlab_callback = SwanLabCallback(project="hf-visualization")

trainer = Trainer(
    ...
    # Pass the callbacks parameter
    callbacks=[swanlab_callback],
)

trainer.train()
```

## 3. Complete Example Code

```python (4,41,50)
import evaluate
import numpy as np
import swanlab
from swanlab.integration.huggingface import SwanLabCallback
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


dataset = load_dataset("yelp_review_full")

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

metric = evaluate.load("accuracy")

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

training_args = TrainingArguments(
    output_dir="test_trainer",
    # If you only want to use SwanLab to track experiments, set the report_to parameter to "none"
    report_to="none",
    num_train_epochs=3,
    logging_steps=50,
)

# Instantiate SwanLabCallback
swanlab_callback = SwanLabCallback(experiment_name="TransformersTest")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
    # Pass the callbacks parameter
    callbacks=[swanlab_callback],
)

trainer.train()
```

## 4. GUI Effect Display

Hyperparameters automatically recorded:

![ig-hf-transformers-gui-1](/assets/ig-hf-transformers-gui-1.png)

Metrics recorded:

![ig-hf-transformers-gui-2](/assets/ig-hf-transformers-gui-2.png)

## 5. Extension: Add More Callbacks

Imagine a scenario where you want the model to infer test samples at the end of each epoch and log the inference results with SwanLab. You can create a new class that inherits from `SwanLabCallback` and add or reconstruct lifecycle functions. For example:

```python
class NLPSwanLabCallback(SwanLabCallback):    
    def on_epoch_end(self, args, state, control, **kwargs):
        test_text_list = ["example1", "example2"]
        log_text_list = []
        for text in test_text_list:
            result = model(text)
            log_text_list.append(swanlab.Text(result))
            
        swanlab.log({"Prediction": test_text_list}, step=state.global_step)
```

The above is a new callback class for NLP tasks, which adds the `on_epoch_end` function. It will be executed at the end of each epoch during the Transformers training.

View all Transformers lifecycle callback functions: [Link](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py#L311)