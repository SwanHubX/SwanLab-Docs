# swanlab.roc_curve

[Github Source Code](https://github.com/SwanHubX/SwanLab/blob/main/swanlab/data/modules/custom_charts/metrics.py)

```python
roc_curve(
    y_true: Union[List, np.ndarray],
    y_pred_proba: Union[List, np.ndarray],
    title: Optional[str, bool] = None,
) -> None
```

| Parameter      | Description                                                                                                                           |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------|
| y_true       | (Union[List, np.ndarray]) True labels, the actual class labels (0 or 1) in binary classification problems                                                                        |
| y_pred_proba | (Union[List, np.ndarray]) Prediction probabilities, the model's predicted probability values for the positive class (range 0-1)                                                                        |
| title        | (Optional[str, bool]) Whether to display chart title, defaults to None                                                                                                           |

## Introduction

Draw an ROC (Receiver Operating Characteristic) curve to evaluate the performance of binary classification models. The ROC curve shows the relationship between True Positive Rate and False Positive Rate at different thresholds.

ROC curves are an important tool for evaluating classification model performance and can intuitively display model performance at different decision thresholds.

### Basic Usage

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import xgboost as xgb
import swanlab

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Get prediction probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Initialize SwanLab
swanlab.init(project="ROC-Curve-Demo", experiment_name="ROC-Curve-Example")

# Log ROC curve
swanlab.log({
    "roc_curve": swanlab.roc_curve(y_test, y_pred_proba, title=True)
})

swanlab.finish()
```

![](./pr-roc_curve/demo.png)

### Custom Title

```python
# Don't show title (default)
roc_curve = swanlab.roc_curve(y_test, y_pred_proba, title=False)
swanlab.log({"roc_curve_no_title": roc_curve})

# Show title
roc_curve = swanlab.roc_curve(y_test, y_pred_proba, title=True)
swanlab.log({"roc_curve_with_title": roc_curve})

# Custom title
roc_curve = swanlab.roc_curve(y_test, y_pred_proba, title="demo")
swanlab.log({"roc_curve_with_custom_title": roc_curve})
```

### Using with Other Metrics

```python
import swanlab

# Log multiple ML metrics
swanlab.log({
    "roc_curve": swanlab.roc_curve(y_test, y_pred_proba),
    "pr_curve": swanlab.pr_curve(y_test, y_pred_proba),
    "accuracy": accuracy_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred)
})
```

### Notes

1. **Data Format**: `y_true` and `y_pred_proba` can be lists or numpy arrays
2. **Binary Classification**: This function is specifically for binary classification problems
3. **Probability Values**: `y_pred_proba` should be the model's predicted probability for the positive class, ranging from 0-1
4. **Dependencies**: Requires installation of `scikit-learn` and `pyecharts` packages
5. **AUC Calculation**: The function automatically calculates the area under the ROC curve (AUC), but does not display it in the title
6. **Curve Characteristics**: ROC curves start from (0,0) and end at (1,1), with the diagonal line representing random classifier performance
