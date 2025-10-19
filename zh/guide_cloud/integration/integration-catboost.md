# CatBoost

[CatBoost (Categorical Boosting)](https://catboost.ai) 是由俄罗斯通信公司 Yandex 在 2017 年提出的一种基于梯度提升决策树的开源机器学习算法框架，相比 XGBoost 和 LightGBM 在算法准确率方面具有更好的表现。

## 1. 引入 SwanLabCallback
```python
from swanlab.integration.catboost import SwanLabCallback
```

## 2. 初始化 SwanLab 项目
``` python
    swanlab.init(
        project="gbdt-demo",
        experiment_name="catboost-demo-full",
        description="A demo of catboost integration."
    )
```

## 3. 定义 CatBoost 模型实例 && 传递 callback 参数

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
    callbacks=[SwanLabCallback(model.get_params())], # catboost 的 callback 触发点支持有限，建议手动传递模型参数
)

```

## 完整测试代码

```python
import catboost
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from loguru import logger
import swanlab
from swanlab.integration.catboost import SwanLabCallback


if __name__ == "__main__":
    # Step 1: 初始化 swanlab 项目
    swanlab.init(
        project="catboost-demo-project",
        experiment_name="catboost-demo-experiment",
        description="A demo of catboost integration.",
        config={
            "model": "CatBoostClassifier",
            "dataset": "breast_cancer",
        },
    )

    # Step 2: 加载数据集
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Step 3: 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: 创建CatBoost分类器
    model = catboost.CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        eval_metric="Accuracy",
        random_seed=42,
        logging_level="Silent",
    )
    logger.info(f"Model Parameters: {model.get_params()}")
    # Step 5: 训练模型，使用SwanLabCallback
    model.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        use_best_model=True,
        callbacks=[SwanLabCallback(model.get_params())], # ！注意： 建议手动传递超参数
    )

    # Step 6: 进行预测
    y_pred = model.predict(X_test)
    # Step 7: 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"模型准确率: {accuracy:.4f}")
    logger.info("分类报告:")
    logger.info(
        classification_report(
            y_test,
            y_pred,
            target_names=data.target_names,  # pyright: ignore[reportAttributeAccessIssue]
        )
    )
    # Optional: 手动结束 swanlab 进程
    swanlab.finish()


```
