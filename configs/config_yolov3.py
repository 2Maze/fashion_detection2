from mmcv import Config
from mmdet.apis import set_random_seed

# Inherit from
cfg = Config.fromfile('mmdetection/configs/yolo/yolov3_d53_320_273e_coco.py')

# Dataset settings
cfg.dataset_type = 'DeepFashion2'
cfg.data_root = 'data'

cfg.data.test.type = 'DeepFashion2'
cfg.data.test.data_root = 'data'
cfg.data.test.ann_file = 'test.txt'
cfg.data.test.img_prefix = 'test/image'

cfg.data.train.type = 'DeepFashion2'
cfg.data.train.data_root = 'data'
cfg.data.train.ann_file = 'train.json'
cfg.data.train.img_prefix = 'train/image'

cfg.data.val.type = 'DeepFashion2'
cfg.data.val.data_root = 'data'
cfg.data.val.ann_file = 'validation.json'
cfg.data.val.img_prefix = 'validation/image'

# Number of classes
cfg.model.bbox_head.num_classes = 13

# Path to weights
cfg.load_from = 'checkpoints/yolov3_d53_320_273e_coco-421362b6.pth'

# Batch size
cfg.data.samples_per_gpu = 16

# Modify number of classes as per the model head.
cfg.model.bbox_head.num_classes = 13

cfg.optimizer.lr = 0.008 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 5

# The output directory for training. As per the model name.
cfg.work_dir = 'outputs/yolov3_d53_320_273e_coco'
# Evaluation Metric.
cfg.evaluation.metric = 'mAP'
# Evaluation times.
cfg.evaluation.interval = 1
# Checkpoint storage interval.
cfg.checkpoint_config.interval = 1

# Set random seed for reproducible results.
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = 'cuda'
cfg.runner.max_epochs = 10

# Tensorboard logs
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]

