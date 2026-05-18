# CatBoost

[CatBoost (Categorical Boosting)](https://catboost.ai)  is an open-source machine learning algorithm framework based on Gradient Boosting Decision Trees (GBDT), proposed by the Russian telecommunications company Yandex in 2017. Compared with XGBoost and LightGBM, it demonstrates better performance in terms of algorithm accuracy.

## 1. Import SwanLabCallback
```python
from swanlab.integration.catboost import SwanLabCallback
```

## 2. Initialzie SwanLab Project
``` python
    swanlab.init(
        project="gbdt-demo",
        experiment_name="catboost-demo-full",
        description="A demo of catboost integration."
    )
```

## 3. Define CatBoost model instance && pass callback

```python
import catboost

...

model = catboost.CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    eval_metric="Accuracy",
    random_seed=42,
    logging_level="Silent",
)
...
model.fit(
    ...
    callbacks=[SwanLabCallback(model.get_params())], # !NOTE: pass model_params by hand
)

```

## Full Test Code

```python
import catboost
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from loguru import logger
import swanlab
from swanlab.integration.catboost import SwanLabCallback


if __name__ == "__main__":
    # Step 1: Initialize swanlab project
    swanlab.init(
        project="catboost-demo-project",
        experiment_name="catboost-demo-experiment",
        description="A demo of catboost integration.",
        config={
            "model": "CatBoostClassifier",
            "dataset": "breast_cancer",
        },
    )

    # Step 2: load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Step 3: split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: create catboost classifier
    model = catboost.CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        eval_metric="Accuracy",
        random_seed=42,
        logging_level="Silent",
    )
    logger.info(f"Model Parameters: {model.get_params()}")
    # Step 5:  train with SwanLabCallback
    model.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        use_best_model=True,
        callbacks=[SwanLabCallback(model.get_params())], # !NOTE: pass model params
    )

    # Step 6: predict testset
    y_pred = model.predict(X_test)
    # Step 7: evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info("Classification Report:")
    logger.info(
        classification_report(
            y_test,
            y_pred,
            target_names=data.target_names,  # pyright: ignore[reportAttributeAccessIssue]
        )
    )
    # Optional: finish swanlab process
    swanlab.finish()


```
