# Finish an Experiment

In a typical Python runtime environment, when the script finishes running, SwanLab will automatically call `swanlab.finish` to close the experiment and set the run status to "Completed". This step does not require an explicit call.

However, in some special cases, such as **Jupyter Notebook**, you need to explicitly close the experiment using `swanlab.finish`.

The usage is simple; execute `finish` after `init`:

```python (5)
import swanlab

swanlab.init()
...
swanlab.finish()
```

## FAQ

### Can I initialize multiple experiments in one Python script run?

Yes, but you need to add `finish` between multiple `init` calls, like this:

```python
swanlab.init()
···
swanlab.finish()
···
swanlab.init()
```