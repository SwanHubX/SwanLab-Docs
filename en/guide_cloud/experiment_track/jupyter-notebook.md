# Track Experiments with Jupyter Notebook

Combine SwanLab with Jupyter to get interactive visualizations without leaving the Notebook.

![swanlab-love-jupyter](./jupyter-notebook/swanlab-love-jupyter.jpg)

## Install SwanLab in Notebook

```bash
!pip install swanlab -qqq
```

ps: `-qqq` is used to control the amount of output information during command execution and is optional.

## Interact with SwanLab in Notebook

```python
import swanlab

swanlab.init()
...
# In the Notebook, you need to explicitly close the experiment
swanlab.finish()
```

When initializing the experiment with `swanlab.init`, a "Display SwanLab Dashboard" button will appear at the end of the printed information:

![](/assets/jupyter-notebook-1.jpg)

Clicking this button will embed the SwanLab web page for the experiment in the Notebook:

![](/assets/jupyter-notebook-2.jpg)

Now, you can directly see the training process and interact with it in this embedded web page.
