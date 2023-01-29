# Object detection with YOLOV3 and SSD
![](doc/ssd_inference.gif)
<p align="center">Example of working SSD</p>

Train SSD and YOLOv3 on [DeepFashion2 dataset](https://github.com/switchablenorms/DeepFashion2) with [mmdetection](https://github.com/open-mmlab/mmdetection).

## Installation

```Shell
git clone https://github.com/2Maze/fashion_detection2.git
cd fashion_detection2
pip install -r requirements.txt
```

## Logs
All logs can be viewed in tensorboard by entering this command:
```
tensorboard --logdir .
```

## Streamlit demo
Model will be loaded from model.py file. If you want change model, edit config path, weights path in model.py.
```Shell
streamlit run demo.py
```
## Scipt inference
You may inference model with scripts in repository. If you have images then copy them in demo/demo_images, because  inference_images.py will be search it there. Scipt contain flags which must be specified. For help start this script with -h flag.
```Shell
python inference_image.py
```
For inference video copy them into demo/demo_videos. All videos must .mp4 format. For helps with flags of script start it with -h flag.
```Shell
python inference_video.py
```

## Instruction for fit your model
If you want to train your model:
1. Download [dataset](https://github.com/switchablenorms/DeepFashion2)
2. Choose model.
    
    If you don't know what models are in mmdetection:
    ```Shell
    python download_weights.py
    ```
    You will see weights.txt file with all models in mmdetection.
3. Download weights your models.

    When you have chosen a model, you must download its weights using the same script with the flag -w and model's name. For example download ssd weights:

    ```Shell
    python download_weights.py -w ssd300_coco
    ```
    After download appear folder with name configs with your weights.
4. Create and configure config.

Next step is create and configure .py file with config of your model. All configs are in the configs folder. Examples you can see in the same folder.

5. Train model

At this step you need build dataset and train detector. All examples in repository. Good luck!