import argparse
import json
import mmcv

from os import listdir
from os.path import join
from tqdm import tqdm


def parse_args():
    """Parse arguments

    Returns:
        argparse.Namespace: object with arguments from user
    """
    parser = argparse.ArgumentParser(description='Prepare annotations for mmdetection class dataset.')
    
    parser.add_argument(
        '-r',
        '--root-folder',
        default='.',
        type=str,
        help='Root of dataset'
    )

    parser.add_argument(
        '-t',
        '--train-folder',
        default='train',
        help='Train folder of dataset'
    )

    parser.add_argument(
        '-v',
        '--validation-folder',
        default='validation',
        help='Validation folder of dataset'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_ann = dict()
    validation_ann = dict()

    # Train annotations
    with open(f'{args.root_folder}/train.json', 'w') as file:
        for name in tqdm(listdir(join(args.train_folder, 'annos')), desc='Train set'):
            name = name.split('.')[0]
            train_ann[name] = dict()
            
            # load image
            image = mmcv.imread(join(args.train_folder, 'image', name.split('.')[0] + '.jpg'))
            height, width = image.shape[:2]
            train_ann[name]['height'] = height
            train_ann[name]['width'] = width
            
        json.dump(train_ann, file)

    # Validation annotations
    with open(f'{args.root_folder}/validation.json', 'w') as file:
        for name in tqdm(listdir(join(args.validation_folder, 'annos')), desc='Validation set'):
            name = name.split('.')[0]
            validation_ann[name] = dict()
            
            # load image
            image = mmcv.imread(join(args.validation_folder, 'image', name.split('.')[0] + '.jpg'))
            height, width = image.shape[:2]
            validation_ann[name]['height'] = height
            validation_ann[name]['width'] = width
            
        json.dump(validation_ann, file)