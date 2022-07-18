import cv2
import numpy as np
import os
import json
from tqdm import tqdm

from leveraging_geometry_for_shape_estimation.segmentation.meshrcnn_vis_tools import draw_segmentation_prediction

target_folder = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val/'

target_size = (480,360)

for img_name in tqdm(sorted(os.listdir(target_folder + 'images'))):
# for img_name in tqdm(sorted(['scene0000_00-000000.jpg'])):

    img = cv2.imread(target_folder + 'images/' + img_name)
    orig_img_size = img.shape[:2][::-1]
    img = cv2.resize(img,target_size)

    with open(target_folder + 'gt_infos/' + img_name.split('.')[0] + '.json','r') as f:
        infos = json.load(f)

    for i in range(len(infos['objects'])):

        out_path = target_folder + 'masks_vis/' + img_name.split('.')[0] + '_' + str(i).zfill(2) + '.png'
        if os.path.exists(out_path):
            continue

        mask_path = target_folder + 'masks/' + img_name.split('.')[0] + '_' + str(i).zfill(2) + '.png'
        assert os.path.exists(mask_path),mask_path
        mask = cv2.imread(mask_path)
        boxes = np.array(infos['objects'][i]['bbox'])

        mask = cv2.resize(mask,target_size)
        ratio = np.array(target_size) / np.array(orig_img_size)
        boxes = boxes * np.concatenate([ratio,ratio])

        vis = draw_segmentation_prediction(img,mask,np.array([0,255,0]),np.array([boxes]),'',font_scale=0.6)

        vis = cv2.resize(vis,(480,360))

        cv2.imwrite(out_path,vis)