# 记录PR曲线

PR曲线展示了在不同阈值下精确率（Precision）和召回率（Recall）的关系。绘制PR（Precision-Recall）曲线，在评估二分类模型的性能时很有用。

PR曲线也特别适用于处理不平衡数据集，能够更好地评估模型在少数类上的表现。

你可以使用`swanlab.pr_curve`来记录PR曲线。

[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](https://swanlab.cn/@ZeyiLin/ComputeMetrics/runs/35snhyn3wndz58r4j8d4h/chart#ZTIwZm1s-aVI2S1ZCQl8=)

![](./py-pr_curve/demo.png)

### 基本用法

```python {22}
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import xgboost as xgb
import swanlab

# 生成示例数据
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练模型
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 获取预测概率
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 初始化SwanLab
swanlab.init(project="PR-Curve-Demo", experiment_name="PR-Curve-Example")

# 记录PR曲线
swanlab.log({
    "pr_curve": swanlab.pr_curve(y_test, y_pred_proba, title=True)
})

swanlab.finish()
```

### 自定义标题

```python
# 不显示标题(默认)
pr_curve = swanlab.pr_curve(y_test, y_pred_proba, title=False)
swanlab.log({"pr_curve_no_title": pr_curve})

# 显示标题
pr_curve = swanlab.pr_curve(y_test, y_pred_proba, title=True)
swanlab.log({"pr_curve_with_title": pr_curve})

# 自定义标题
pr_curve = swanlab.pr_curve(y_test, y_pred_proba, title="demo")
swanlab.log({"pr_curve_with_custom_title": pr_curve})
```

### 注意事项

1. **数据格式**: `y_true`和`y_pred_proba`可以是列表或numpy数组
2. **二分类**: 此函数专用于二分类问题
3. **概率值**: `y_pred_proba`应该是模型对正类的预测概率，范围在0-1之间
4. **依赖包**: 需要安装`scikit-learn`和`pyecharts`包
5. **AUC计算**: 函数会自动计算PR曲线下的面积（AUC），但不会默认在标题中显示
