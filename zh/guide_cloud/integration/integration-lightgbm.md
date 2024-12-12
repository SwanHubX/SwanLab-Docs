# LightGBM

LightGBM（Light Gradient Boosting Machine）是一种基于决策树算法的分布式梯度提升框架，由微软公司在2017年发布。它以高效、快速和准确著称，广泛应用于分类、回归和排序等机器学习任务。

![lightgbm](/zh/guide_cloud/integration/lightgbm/logo.png)

你可以使用LightGBM快速进行模型训练，同时使用SwanLab进行实验跟踪与可视化。

## 1. 引入SwanLabCallback

```python
from swanlab.integration.lightgbm import SwanLabCallback
```

SwanLabCallback是适配于LightGBM的日志记录类。


## 2. 初始化SwanLab

```python
swanlab.init(
    project="lightgbm-example", 
    experiment_name="breast-cancer-classification"
)
```

## 3. 传入`lgb.train`

```python
import lightgbm as lgb

gbm = lgb.train(
    ...
    callbacks=[SwanLabCallback()]
)
```


## 4. 完整测试代码

```python
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import swanlab
from swanlab.integration.lightgbm import SwanLabCallback

# Step 1: 初始化swanlab
swanlab.init(project="lightgbm-example", experiment_name="breast-cancer-classification")

# Step 2: 加载数据集
data = load_breast_cancer()
X = data.data
y = data.target

# Step 3: 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: 创建LightGBM数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Step 5: 设置参数
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Step 6: 使用swanlab callback训练模型
num_round = 100
gbm = lgb.train(
    params,
    train_data,
    num_round,
    valid_sets=[test_data],
    callbacks=[SwanLabCallback()]
)

# Step 7: 预测
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]

# Step 8: 评估模型
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"模型准确率: {accuracy:.4f}")
swanlab.log({"accuracy": accuracy})

# Step 9: 保存模型
gbm.save_model('lightgbm_model.txt')

# Step 10: 加载模型并预测
bst_loaded = lgb.Booster(model_file='lightgbm_model.txt')
y_pred_loaded = bst_loaded.predict(X_test)
y_pred_binary_loaded = [1 if p >= 0.5 else 0 for p in y_pred_loaded]

# Step 11: 评估加载模型
accuracy_loaded = accuracy_score(y_test, y_pred_binary_loaded)
print(f"加载模型后的准确率: {accuracy_loaded:.4f}")
swanlab.log({"accuracy_loaded": accuracy_loaded})

# Step 12: 结束swanlab实验
swanlab.finish()
```