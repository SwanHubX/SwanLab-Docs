# XGBoost

XGBoost (eXtreme Gradient Boosting) is an efficient, flexible, and widely-used gradient boosting framework introduced by Tianqi Chen in 2014. It is based on decision tree algorithms and builds a powerful predictive model by integrating multiple weak learners (typically decision trees). XGBoost has demonstrated outstanding performance in various machine learning competitions and practical applications, especially in classification, regression, and ranking tasks.

![xgboost](/zh/guide_cloud/integration/xgboost/logo.png)

You can use XGBoost for rapid model training while leveraging SwanLab for experiment tracking and visualization.

## 1. Import SwanLabCallback

```python
from swanlab.integration.xgboost import SwanLabCallback
```

`SwanLabCallback` is a logging class designed for XGBoost.

## 2. Initialize SwanLab

```python
swanlab.init(
    project="xgboost-example", 
)
```

## 3. Pass to `xgb.train`

```python
import xgboost as xgb

bst = xgb.train(
    ...
    callbacks=[SwanLabCallback()]
)
```

## 4. Complete Test Code

```python
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import swanlab
from swanlab.integration.xgboost import SwanLabCallback

# Initialize swanlab
swanlab.init(
    project="xgboost-breast-cancer",
    config={
        "learning_rate": 0.1,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "num_round": 100
    }
)

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to DMatrix format, which is the internal data format of XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters
params = {
    'objective': 'binary:logistic',  # Binary classification task
    'max_depth': 3,                  # Maximum depth of a tree
    'eta': 0.1,                      # Learning rate
    'subsample': 0.8,                # Subsample ratio
    'colsample_bytree': 0.8,         # Feature subsample ratio
    'eval_metric': 'logloss'         # Evaluation metric
}

# Train the model
num_round = 100  # Number of iterations
bst = xgb.train(
    params, 
    dtrain, 
    num_round,
    evals=[(dtrain, 'train'), (dtest, 'test')], 
    callbacks=[SwanLabCallback()]
)

# Make predictions
y_pred = bst.predict(dtest)
y_pred_binary = [round(value) for value in y_pred]  # Convert probabilities to binary classification results

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy:.4f}")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_binary, target_names=data.target_names))

# Save the model
bst.save_model('xgboost_model.model')

# End the swanlab session
swanlab.finish()
```