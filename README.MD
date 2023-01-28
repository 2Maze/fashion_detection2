# Object detection with YOLOV3 and SSD
![](doc/ssd_inference.gif)
<p align="center">Example of working SSD</p>

Train SSD and YOLOv3 on DeepFashion2 with [mmdetection](https://github.com/open-mmlab/mmdetection)

## Instalation

```Shell
git clone https://github.com/2Maze/fashion_detection2.git
pip install -r requirements.txt
```

## Dataset
If you want to fit your model, then download [DeepFashion2 dataset](https://github.com/switchablenorms/DeepFashion2).


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