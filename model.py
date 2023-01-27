# This file will be use for demo in web
from mmdet.apis import init_detector
from importlib.machinery import SourceFileLoader

CONFIG_PATH = 'configs/config_yolov3.py'
WEIGHT_PATH = 'outputs/yolov3_d53_320_273e_coco/latest.pth' 
DEVICE = 'cuda'

config_module = SourceFileLoader('cfg', CONFIG_PATH).load_module()
cfg = config_module.cfg
model = init_detector(cfg, WEIGHT_PATH, device=DEVICE)
print(f'Model has been loaded successfully on {DEVICE}!')