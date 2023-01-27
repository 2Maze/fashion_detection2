import os
import requests
import argparse
import glob
import yaml

from tqdm import tqdm

CHECKPOINTS_FOLDER_NAME = "checkpoints"

def parse_args() -> argparse.Namespace:
    """Parse arguments

    Returns:
        argparse.Namespace: object with arguments from user
    """

    parser = argparse.ArgumentParser(description='Download weights of chosen model by url. When activate script, weights.txt updating.')
    
    parser.add_argument(
        '-w',
        '--weights',
        type=str,
        help='Weights name'
    )

    return parser.parse_args()


def download_weights(url, file_save_name) -> None:
    """Download weights for any model.
    
    Args:
        url: Download URL for the weihgt file.
        file_save_name: String name to save the file on to disk.
    """

    # Download the file if not present.
    if not os.path.exists(os.path.join(CHECKPOINTS_FOLDER_NAME, file_save_name)):
        print(f"Downloading {file_save_name}")
        file = requests.get(url, stream=True)
        total_size = int(file.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(
            total=total_size, 
            unit='iB', 
            unit_scale=True
        )

        with open(os.path.join(CHECKPOINTS_FOLDER_NAME, file_save_name), 'wb') as f:
            for data in file.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()
    else:
        print('File already present')

def parse_meta_file() -> list:
    """
    Function to parse all the model meta files inside `mmdetection/configs`
    and return the download URLs for all available models.
    Returns:
        weights_list: List containing URLs for all the downloadable models.
    """
    root_meta_file_path = 'mmdetection/configs'
    all_metal_file_paths = glob.glob(os.path.join(root_meta_file_path, '*', 'metafile.yml'), recursive=True)
    weights_list = []
    for meta_file_path in all_metal_file_paths:
        with open(meta_file_path,'r') as f:
            yaml_file = yaml.safe_load(f)
            
        for i in range(len(yaml_file['Models'])):
            try:
                weights_list.append(yaml_file['Models'][i]['Weights'])
            except:
                for k, v in yaml_file['Models'][i]['Results'][0]['Metrics'].items():
                    if k == 'Weights':
                        weights_list.append(yaml_file['Models'][i]['Results'][0]['Metrics']['Weights'])
    return weights_list


def get_model(model_name: str) -> None:
    """
    Either downloads a model or loads one from local path if already 
    downloaded using the weight file name (`model_name`) provided.
    
    Args
        model_name: Name of the weight file. Most likely in the format
            retinanet_ghm_r50_fpn_1x_coco. SEE `weights.txt` to know weight file
            name formats and downloadable URL formats.
    Returns:
        None: Download model to specified folder
    """

    # Get the list containing all the weight file download URLs.
    weights_list = parse_meta_file()
    download_url = None
    
    for weights in weights_list:
        if model_name == weights.split('/')[-2]:
            print(f"Founds weights: {weights}\n")
            download_url = weights
            break
    assert download_url != None, f"{model_name} weight file not found!!!"
    # Download the checkpoint file.
    download_weights(
        url=download_url,
        file_save_name=download_url.split('/')[-1]
    )

if __name__ == '__main__':

    # check folder of checkpoints
    os.makedirs(CHECKPOINTS_FOLDER_NAME, exist_ok=True)

    # update weights.txt
    weights_list = parse_meta_file()

    with open('weights.txt', 'w') as f:
        for weights in weights_list:
            f.writelines(f"{weights}\n")
    f.close()

    args = parse_args()

    # if args.url is not empty, then download model
    if args.weights:
        get_model(args.weights)
    