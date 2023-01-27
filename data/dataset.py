import json
import numpy as np

from os.path import join
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset


@DATASETS.register_module()
class DeepFashion2(CustomDataset):
    
    CLASSES = ('short sleeve top', 'long sleeve top', 'short sleeve outwear', 'long sleeve outwear', 'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short sleeve dress', 'long sleeve dress', 'vest dress', 'sling dress')

    def load_annotations(self, ann_file):
        # category to label
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        # load image list from file
        j = json.load(open(join(ann_file)))
        image_list = [image for image in j]
        
        data_infos = []
        for image_name in image_list:
            height, width = j[image_name]['height'], j[image_name]['width']

            with open(join(*(self.img_prefix.split('/')[:-1]), 'annos', image_name + '.json')) as f:
                annt = json.load(f)
            

            data_info = dict(filename=image_name + '.jpg', width=width, height=height)

            for key in list(annt.keys()):            
                gt_bboxes = []
                gt_labels = []

                if 'item' in key:
                    box = annt[key]['bounding_box']
                    gt_bboxes.append(box)
                    gt_labels.append(cat2label[annt[key]['category_name']])
            
            data_anno = dict(
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels=np.array(gt_labels, dtype=np.long))

            data_info.update(ann=data_anno)
            data_infos.append(data_info)

        return data_infos