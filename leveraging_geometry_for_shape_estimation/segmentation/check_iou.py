import argparse
from copy import copy
import os
from re import S
import cv2
from glob import glob
from detectron2.structures import Boxes, BoxMode, pairwise_iou
import json
import torch
from shutil import copyfile
import pycocotools.mask as mask_util
import sys
from tqdm import tqdm
import numpy as np
from PIL import Image

from meshrcnn_vis_tools import draw_segmentation_prediction,draw_mask,draw_boxes,draw_text

def calculate_box_overlap(gt_bbox,pred_bbox):
    gt_bbox = Boxes(torch.tensor([gt_bbox], dtype=torch.float32))
    pred_bbox = Boxes(torch.tensor([pred_bbox], dtype=torch.float32))

    boxiou = pairwise_iou(gt_bbox, pred_bbox)

    return boxiou

def get_file(expression,files):
    for file in files:
        if expression in file:
            return file

def load_gt_mask(path):
    with open(path, "rb") as f:
        gt_mask = torch.tensor(np.asarray(Image.open(f), dtype=np.float32) / 255.0)
    gt_mask = (gt_mask > 0).to(dtype=torch.uint8)  # binarize mask
    gt_mask_rle = [mask_util.encode(np.array(gt_mask[:, :, None], order="F"))[0]]
    return gt_mask_rle

def calculate_mask_overlap(gt_mask_rle,predicted_mask_path):

    # load predicted
    with open(predicted_mask_path, "rb") as f:
        pred_mask = torch.tensor(np.asarray(Image.open(f), dtype=np.float32) / 255.0)
    pred_mask = (pred_mask > 0).to(dtype=torch.uint8)  # binarize mask
    pred_rles = [mask_util.encode(np.array(pred_mask[:, :, None], order="F", dtype="uint8"))[0]]

    miou = mask_util.iou(pred_rles, gt_mask_rle, [0])
    return miou[0][0]



def main():
    print('Check iou')
    # global_info = os.path.dirname(os.path.abspath(__file__)) + '/../global_information.json'
    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    target_folder = global_config["general"]["target_folder"]
    with open(target_folder + '/global_stats/visualisation_images.json','r') as f:
        visualisation_list = json.load(f)


    target_folder = global_config["general"]["target_folder"]
    boxiou_threshold = global_config["segmentation"]["boxiou_threshold"]

    for name in tqdm(os.listdir(target_folder + '/segmentation_masks')):
        out_path = target_folder + '/bbox_overlap/' + name.split('.')[0] + '.json'
        # if os.path.exists(out_path):
        #     continue

        valid_predictions = {}

        # with open(target_folder + '/segmentation_infos/' + name.replace('.png','.json'),'r') as f:
        with open(target_folder + '/segmentation_infos/' + name.split('.')[0] + '.json','r') as f:
            seg_info = json.load(f)

        with open(target_folder + '/gt_infos/' + seg_info['img'].split('.')[0] + '.json','r') as f:
            gt_info = json.load(f)
        
        gt_bbox = gt_info['bbox']
        pred_bbox = seg_info['predictions']['bbox']
        pred_category = seg_info['predictions']['category']
        

        boxiou = calculate_box_overlap(gt_bbox,pred_bbox)

        gt_mask_path = target_folder + '/masks/' + gt_info["img"].split('.')[0] + '.png'
        predicted_mask_path = target_folder + '/segmentation_masks/' + name
        gt_mask_rle = load_gt_mask(gt_mask_path)
        mask_iou = calculate_mask_overlap(gt_mask_rle,predicted_mask_path)

        valid_predictions['box_iou'] = boxiou.item()
        valid_predictions['valid'] = boxiou.item() > boxiou_threshold
        valid_predictions['mask_iou'] = mask_iou
        valid_predictions['correct_category'] = pred_category == gt_info["category"]

        with open(out_path,'w') as bbox_file:
            json.dump(valid_predictions, bbox_file)

        if gt_info["img"] in visualisation_list:

            img = cv2.imread(target_folder + '/images/' + gt_info["img"])
            pred_mask = cv2.imread(target_folder + '/segmentation_masks/' + name)
            gt_mask = cv2.imread(target_folder + '/masks/' + gt_info["img"].split('.')[0] + '.png')
            pred_text = "score " + str(np.round(seg_info["predictions"]["score"],4)) + '  ' + seg_info["predictions"]["category"] + ' mask_iou: ' + str(np.round(mask_iou,3)) + ' box_iou: ' + str(np.round(boxiou.item(),3))
            gt_text = "GT: {}".format(gt_info["category"])

            img = draw_segmentation_prediction(img, gt_mask, np.array([0,200,0]), np.array(gt_bbox)[np.newaxis,...],gt_text)
            img = draw_segmentation_prediction(img, pred_mask,  np.array([0,0,200]),np.array(pred_bbox)[np.newaxis,...],pred_text)
        

            # intersect = np.logical_and(gt_mask,pred_mask)
            # union = np.logical_or(gt_mask,pred_mask)
            # diff = np.logical_and(np.invert(intersect),union)

            # img = draw_mask(img,intersect,np.array([0,200,0]))
            # img = draw_mask(img,diff,np.array([0,0,200]))
            # img = draw_boxes(img,np.array(gt_bbox)[np.newaxis,...])
            # img = draw_boxes(img,np.array(pred_bbox)[np.newaxis,...])

            cv2.imwrite(target_folder + '/segmentation_vis/' + name,img)
            

if __name__ == '__main__':
    main()





