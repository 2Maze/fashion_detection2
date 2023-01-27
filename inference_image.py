import os
import argparse

from os import listdir
from os.path import join
from importlib.machinery import SourceFileLoader
from mmdet.apis import inference_detector, init_detector


def parse_args():
    """Parse arguments

    Returns:
        argparse.Namespace: object with arguments from user
    """
    parser = argparse.ArgumentParser(description='Inference model on video.')
    
    parser.add_argument(
        '-c',
        '--config',
        help='Path to config file of model. Where config must be cfg variable.'
    )

    parser.add_argument(
        '-w',
        '--weights',
        type=str,
        help='Path to weights of model.'
    )

    parser.add_argument(
        '-r',
        '--root',
        default='demo',
        type=str,
        help='Path to root folder. Root folder must contain demo_images folder with images.'
    )

    parser.add_argument(
        '-t',
        '--threshold',
        default=0.3,
        type=float,
        help='Threshold for model output.'
    )

    parser.add_argument(
        '-d',
        '--device',
        default='cuda',
        type=str,
        help='Device that will do the computing. May be cpu or cuda. If you have several video cards, you can specify which one.'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    os.makedirs(join(args.root, 'inference_images'), exist_ok=True)

    config_module = SourceFileLoader('cfg', args.config).load_module()
    
    model = init_detector(config_module.cfg, args.weights, device=args.device)

    for img in listdir(join(args.root, 'demo_images')):
        if img.split('.')[1] in ['jpg', 'jpeg', 'png']:
            result = inference_detector(model, join(args.root, 'demo_images', img))
            model.show_result(join(args.root, 'demo_images', img), result, score_thr=args.threshold, out_file=join(args.root, 'inference_images', config_module.cfg.model.type + '_' + f'{args.threshold}_' + img.split('.')[0] + '_inference.jpg'))