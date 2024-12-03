# Argparse

`argparse` is a module in the Python standard library used for parsing command-line arguments and options. With argparse, developers can easily write user-friendly command-line interfaces, defining the names, types, default values, help information, and more of command-line arguments.

Integrating `argparse` with swanlab is very simple; just pass the created argparse object to `swanlab.config` to record it as hyperparameters:

```python
import argparse
import swanlab

# Initialize Argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=20)
parser.add_argument('--lr', default=0.001)
args = parser.parse_args()

swanlab.init(config=args)
```

Running example:
```bash
python main.py --epochs 100 --lr 1e-4
```

![alt text](/assets/ig-argparse.png)