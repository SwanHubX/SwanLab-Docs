# FAQ

## Why can't I input the API Key during login?

Refer to this answer: [Link](https://www.zhihu.com/question/720308649/answer/25076837539)

## How to start multiple experiments from a single script?

Add `swanlab.finish()` between multiple experiment creations.

After executing `swanlab.finish()`, executing `swanlab.init()` again will create a new experiment.  
If `swanlab.finish()` is not executed, subsequent `swanlab.init()` calls will be ignored.

## How to disable SwanLab logging during training (for debugging)?

Set the `mode` parameter of `swanlab.init` to 'disabled' to prevent experiment creation and data logging.

```python
swanlab.init(mode='disabled')
```

## The local training has ended, but the experiment is still shown as running on the SwanLab UI. How to change the status?

Click the stop button next to the experiment name to change the status from "Running" to "Interrupted" and stop receiving data uploads.

![stop](/assets/stop.png)

## How to view local details of the line chart?

Zoom in on the line chart by holding down the mouse and dragging over the target area to magnify that region.

![details](/assets/faq-chart-details.png)

Double-click the area to restore the original view.

## How to remove the suffix from the experiment name?

```python
swanlab.init(suffix=None)
```

Note: Starting from version 0.3.22, the suffix is no longer automatically added to the experiment name.