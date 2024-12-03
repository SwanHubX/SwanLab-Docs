# PaddleDetection

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) is an end-to-end object detection development toolkit developed by Baidu based on its deep learning framework PaddlePaddle. It supports object detection, instance segmentation, multi-object tracking, and real-time multi-person keypoint detection, aiming to help developers more efficiently develop and train object detection models.

![PaddleDetection](/assets/ig-paddledetection-1.png)

You can use PaddleDetection to quickly train object detection models while using SwanLab for experiment tracking and visualization.

## 1. Import SwanLabCallback

First, in your cloned PaddleDetection project, find the `ppdet/engine/callbacks.py` file and add the following code at the bottom:

```python
class SwanLabCallback(Callback):
    def __init__(self, model):
        super(SwanLabCallback, self).__init__(model)

        try:
            import swanlab
            self.swanlab = swanlab
        except Exception as e:
            logger.error('swanlab not found, please install swanlab. '
                         'Use: `pip install swanlab`.')
            raise e

        self.swanlab_params = {k[8:]: v for k, v in model.cfg.items() if k.startswith("swanlab_")}

        self._run = None
        if dist.get_world_size() < 2 or dist.get_rank() == 0:
            _ = self.run
            self.run.config.update(self.model.cfg)

        self.best_ap = -1000.
        self.fps = []

    @property
    def run(self):
        if self._run is None:
            self._run = self.swanlab.get_run() or self.swanlab.init(**self.swanlab_params)
        return self._run

    def on_step_end(self, status):
        if dist.get_world_size() < 2 or dist.get_rank() == 0 and status['mode'] == 'train':
            training_status = status['training_staus'].get()
            batch_time = status['batch_time']
            data_time = status['data_time']
            batch_size = self.model.cfg['{}Reader'.format(status['mode'].capitalize())]['batch_size']

            ips = float(batch_size) / float(batch_time.avg)
            metrics = {
                "train/" + k: float(v) for k, v in training_status.items()
            }
            metrics.update({
                "train/ips": ips,
                "train/data_cost": float(data_time.avg),
                "train/batch_cost": float(batch_time.avg)
            })

            self.fps.append(ips)
            self.run.log(metrics)

    def on_epoch_end(self, status):
        if dist.get_world_size() < 2 or dist.get_rank() == 0:
            mode = status['mode']
            epoch_id = status['epoch_id']
            
            if mode == 'train':
                fps = sum(self.fps) / len(self.fps)
                self.fps = []

                end_epoch = self.model.cfg.epoch
                if (epoch_id + 1) % self.model.cfg.snapshot_epoch == 0 or epoch_id == end_epoch - 1:
                    save_name = str(epoch_id) if epoch_id != end_epoch - 1 else "model_final"
                    tags = ["latest", f"epoch_{epoch_id}"]
            
            elif mode == 'eval':
                fps = status['sample_num'] / status['cost_time']

                merged_dict = {
                    f"eval/{key}-mAP": map_value[0]
                    for metric in self.model._metrics
                    for key, map_value in metric.get_results().items()
                }
                merged_dict.update({
                    "epoch": status["epoch_id"],
                    "eval/fps": fps
                })

                self.run.log(merged_dict)

                if status.get('save_best_model'):
                    for metric in self.model._metrics:
                        map_res = metric.get_results()
                        key = next((k for k in ['bbox', 'keypoint', 'mask'] if k in map_res), None)
                        
                        if not key:
                            logger.warning("Evaluation results empty, this may be due to "
                                           "training iterations being too few or not "
                                           "loading the correct weights.")
                            return
                        
                        if map_res[key][0] >= self.best_ap:
                            self.best_ap = map_res[key][0]
                            save_name = 'best_model'
                            tags = ["best", f"epoch_{epoch_id}"]

    def on_train_end(self, status):
        self.run.finish()
```

## 2. Modify the Trainer Code

In the `ppdet/engine/trainer.py` file, add `SwanLabCallback` to the line where `from .callbacks import` is:

```python
from .callbacks import Callback, ComposeCallback, LogPrinter, Checkpointer, WiferFaceEval, VisualDLWriter, SniperProposalsGenerator, WandbCallback, SemiCheckpointer, SemiLogPrinter, SwanLabCallback
```

Next, find the `__init_callbacks` method of the `Trainer` class and add the following code under `if self.mode == 'train':`:

```python
if self.cfg.get('use_swanlab', False) or 'swanlab' in self.cfg:
    self._callbacks.append(SwanLabCallback(self))
```

With this, you have completed the integration of SwanLab with PaddleYolo! Next, simply add `use_swanlab: True` to the training configuration file to start visualizing and tracking the training.

## 3. Modify the Configuration File

We will use `yolov3_mobilenet_v1_roadsign` as an example.

In the `configs/yolov3/yolov3_mobilenet_v1_roadsign.yml` file, add the following code:

```yaml
use_swanlab: true
swanlab_project: PaddleYOLO # Optional
swanlab_experiment_name: yolov3_mobilenet_v1_roadsign # Optional
swanlab_description: A training test for PaddleYOLO # Optional
# swanlab_workspace: swanhub # Organization name, optional
```

## 4. Start Training

```bash
python -u tools/train.py -c configs/yolov3/yolov3_mobilenet_v1_roadsign.yml --eval
```

During the training process, you can see the logs of the entire training process, as well as the automatically generated visual charts after the training is complete.