from mmcv import Config
from mmdet.apis import set_random_seed

# Inherit from
cfg = Config.fromfile('mmdetection/configs/ssd/ssd300_coco.py')

# Dataset settings
cfg.dataset_type = 'DeepFashion2'
cfg.data_root = 'data'

cfg.data.test.type = 'DeepFashion2'
cfg.data.test.data_root = 'data'
cfg.data.test.ann_file = 'test.txt'
cfg.data.test.img_prefix = 'test/image'

cfg.data.train.dataset.type = 'DeepFashion2'
cfg.data.train.dataset.data_root = 'data'
cfg.data.train.dataset.ann_file = 'train.json'
cfg.data.train.dataset.img_prefix = 'train/image'

cfg.data.val.type = 'DeepFashion2'
cfg.data.val.data_root = 'data'
cfg.data.val.ann_file = 'validation.json'
cfg.data.val.img_prefix = 'validation/image'

# Number of classes
cfg.model.bbox_head.num_classes = 13

# Path to weights
cfg.load_from = 'checkpoints/ssd300_coco_20210803_015428-d231a06e.pth'

# Batch size
cfg.data.samples_per_gpu = 16

cfg.work_dir = 'outputs/ssd300_coco'

cfg.optimizer.lr = 0.02 / 8.  # 3e-3
cfg.lr_config.policy = 'step'
cfg.lr_config.warmup = 'linear'
cfg.lr_config.warmup_ratio = 1.0 / 1e10

cfg.log_config.interval = 5

cfg.evaluation.metric = 'mAP'
cfg.evaluation.interval = 1
cfg.checkpoint_config.interval = 1

# Set random seed for reproducible results.
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = 'cuda'
cfg.runner.max_epochs = 1
    
# Use tensorboard
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]
