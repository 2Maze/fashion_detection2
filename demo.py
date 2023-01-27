import streamlit as st
import numpy as np
import io
import PIL.Image as Image

from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from mmdet.apis import inference_detector
from model import model, cfg


def webcam_object_detection():
    """
    Real time object detection
    """
    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.threshold = 0.3
            
        def transform(self, frame):
            img = frame.to_ndarray(format='bgr24')
            result = inference_detector(model, img)
            frame = model.show_result(img, result, score_thr=self.threshold)
            return frame


    ctx = webrtc_streamer(key='example', video_transformer_factory=VideoTransformer)

    if ctx.video_transformer:
        ctx.video_transformer.threshold = st.slider('Threshold', min_value=0., max_value=1., value=0.3)
    
def img_object_detection():
    """
    Image object detection
    """
    image_file = st.file_uploader(label='Load', type=['png', 'jpeg', 'jpg'])
    threshold = st.slider('Threshold', min_value=0., max_value=1., value=0.3)

    if image_file is not None:
        bytes_data = image_file.read()
        image = np.array(Image.open(io.BytesIO(bytes_data)))
        result = inference_detector(model, image)
        inference_image = model.show_result(image, result, score_thr=threshold)
        st.image(inference_image)
        

if __name__ == '__main__':
    st.header('Demo of model which trained on DeepFashion2')
    st.info(f'Next model working for you: {cfg.model.type}', icon='🤖')

    pages = {
        'Real time object detection': webcam_object_detection,
        'Image object detection': img_object_detection,
    }
    
    page_titles = pages.keys()

    page_title = st.sidebar.selectbox(
        'Choose method',
        page_titles,
    )
    
    st.subheader(page_title)

    page_func = pages[page_title]
    page_func()