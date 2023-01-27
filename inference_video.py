import argparse
import os
import mmcv
import cv2 as cv

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
        help='Path to root folder. Root folder must contain demo_videos folder with videos.'
    )

    parser.add_argument(
        '-t',
        '--threshold',
        default=0.3,
        type=float,
        help='Threshold for model outputs.'
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

    os.makedirs(join(args.root, 'inference_videos'), exist_ok=True)

    config_module = SourceFileLoader('cfg', args.config).load_module()
    
    model = init_detector(config_module.cfg, args.weights, device=args.device)

    for video in listdir(join(args.root, 'demo_videos')):
        if video.split('.')[1] in ['mp4']:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            video_reader = mmcv.VideoReader(join(args.root, 'demo_videos', video))
            video_writer = cv.VideoWriter(
                                          join(args.root, 'inference_videos', config_module.cfg.model.type + '_' + f'{args.threshold}_' + video.split('.')[0] + '_inference.mp4'),
                                          fourcc,
                                          video_reader.fps,
                                          (video_reader.width, video_reader.height))
    
            for frame in mmcv.track_iter_progress(video_reader):
                result = inference_detector(model, frame)
                frame = model.show_result(frame, result, score_thr=args.threshold)
                video_writer.write(frame)
            video_writer.release()