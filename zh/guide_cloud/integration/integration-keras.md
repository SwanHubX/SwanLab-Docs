# Keras

[Keras](https://keras.io/) 是一个用 Python 编写的高级神经网络 API，最初由 François Chollet 创建，并于 2017 年合并到 TensorFlow 中，但依然可以作为一个独立的框架使用。它是一个开源的深度学习框架，运行在 TensorFlow、Theano 或 Microsoft Cognitive Toolkit (CNTK) 等深度学习后端之上。

![keras-image](/assets/ig-keras-1.png)

你可以使用Keras快速进行模型训练，同时使用SwanLab进行实验跟踪与可视化。

[在线演示](https://swanlab.cn/@ZeyiLin/keras_mnist/runs/9gzx3m1ga2q2xb6t6ekxb/chart)

## 1. 引入SwanLabLogger

```python
from swanlab.integration.keras import SwanLabLogger
```

## 2. 与model.fit配合

首先初始化SwanLab：

```python
swanlab.init(
    project="keras_mnist",
    experiment_name="mnist_example",
    description="Keras MNIST Example"
    )
```

然后，在`model.fit`的`callbacks`参数中添加`SwanLabLogger`，即可完成集成：

```python
model.fit(..., callbacks=[SwanLabLogger()])
```

## 3. 案例-MNIST

```python
from swanlab.integration.keras import SwanLabLogger
import tensorflow as tf
import swanlab

# Initialize SwanLab
swanlab.init(
    project="keras_mnist",
    experiment_name="mnist_example",
    description="Keras MNIST Example"
    )

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Build a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model with SwanLabLogger
model.fit(
    x_train, 
    y_train,
    epochs=5,
    validation_data=(x_test, y_test),
    callbacks=[SwanLabLogger()]
)
```

效果演示：

![keras-image](/assets/ig-keras-2.png)

[在线演示](https://swanlab.cn/@ZeyiLin/keras_mnist/runs/9gzx3m1ga2q2xb6t6ekxb/chart)