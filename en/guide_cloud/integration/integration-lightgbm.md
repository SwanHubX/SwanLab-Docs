# LightGBM

LightGBM (Light Gradient Boosting Machine) is a distributed gradient boosting framework based on decision tree algorithms, released by Microsoft in 2017. It is renowned for its efficiency, speed, and accuracy, and is widely used in machine learning tasks such as classification, regression, and ranking.

![lightgbm](/zh/guide_cloud/integration/lightgbm/logo.png)

You can use LightGBM for rapid model training while leveraging SwanLab for experiment tracking and visualization.

## 1. Import SwanLabCallback

```python
from swanlab.integration.lightgbm import SwanLabCallback
```

`SwanLabCallback` is a logging class designed for LightGBM.

## 2. Initialize SwanLab

```python
swanlab.init(
    project="lightgbm-example", 
    experiment_name="breast-cancer-classification"
)
```

## 3. Pass to `lgb.train`

```python
import lightgbm as lgb

gbm = lgb.train(
    ...
    callbacks=[SwanLabCallback()]
)
```

## 4. Complete Test Code

```python
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import swanlab
from swanlab.integration.lightgbm import SwanLabCallback

# Step 1: Initialize swanlab
swanlab.init(project="lightgbm-example", experiment_name="breast-cancer-classification")

# Step 2: Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Step 5: Set parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Step 6: Train the model with swanlab callback
num_round = 100
gbm = lgb.train(
    params,
    train_data,
    num_round,
    valid_sets=[test_data],
    callbacks=[SwanLabCallback()]
)

# Step 7: Predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Model Accuracy: {accuracy:.4f}")
swanlab.log({"accuracy": accuracy})

# Step 9: Save the model
gbm.save_model('lightgbm_model.txt')

# Step 10: Load the model and predict
bst_loaded = lgb.Booster(model_file='lightgbm_model.txt')
y_pred_loaded = bst_loaded.predict(X_test)
y_pred_binary_loaded = [1 if p >= 0.5 else 0 for p in y_pred_loaded]

# Step 11: Evaluate the loaded model
accuracy_loaded = accuracy_score(y_test, y_pred_binary_loaded)
print(f"Accuracy after loading the model: {accuracy_loaded:.4f}")
swanlab.log({"accuracy_loaded": accuracy_loaded})

# Step 12: Finish the swanlab experiment
swanlab.finish()
```