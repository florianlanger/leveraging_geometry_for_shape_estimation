import argparse
from copy import copy
import os
import cv2
from glob import glob
from detectron2.structures import Boxes, BoxMode, pairwise_iou
import json
import torch
from shutil import copyfile
import sys
from tqdm import tqdm

from numpy.core.numeric import full_like

def calculate_box_overlap(gt_bbox,pred_bbox):
    gt_bbox = Boxes(torch.tensor([gt_bbox], dtype=torch.float32))
    pred_bbox = Boxes(torch.tensor([pred_bbox], dtype=torch.float32))

    boxiou = pairwise_iou(gt_bbox, pred_bbox)

    return boxiou

def get_file(expression,files):
    for file in files:
        if expression in file:
            return file



def main():
    print('Check iou')
    global_info = os.path.dirname(os.path.abspath(__file__)) + '/../global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    target_folder = global_config["general"]["target_folder"]
    boxiou_threshold = global_config["segmentation"]["boxiou_threshold"]


    for name in tqdm(os.listdir(target_folder + '/segmentation_masks')):
        out_path = target_folder + '/bbox_overlap/' + name.split('.')[0] + '.json'
        if os.path.exists(out_path):
            continue

        valid_predictions = {}

        # with open(target_folder + '/segmentation_infos/' + name.replace('.png','.json'),'r') as f:
        with open(target_folder + '/segmentation_infos/' + name.split('.')[0] + '.json','r') as f:
            seg_info = json.load(f)

        

        with open(target_folder + '/gt_infos/' + seg_info['img'].split('.')[0] + '.json','r') as f:
            gt_info = json.load(f)
        
        gt_bbox = gt_info['bbox']
        pred_bbox = seg_info['predictions']['bbox']

        boxiou = calculate_box_overlap(gt_bbox,pred_bbox)
        valid_predictions['box_iou'] = boxiou.item()
        valid_predictions['valid'] = boxiou.item() > boxiou_threshold


        with open(out_path,'w') as bbox_file:
            json.dump(valid_predictions, bbox_file)

if __name__ == '__main__':
    main()





