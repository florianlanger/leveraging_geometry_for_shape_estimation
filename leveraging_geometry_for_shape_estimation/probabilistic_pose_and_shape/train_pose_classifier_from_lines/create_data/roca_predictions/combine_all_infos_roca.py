import argparse
from copy import copy
import os
from re import S
from selectors import EpollSelector
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

def calculate_box_overlap(gt_bbox,pred_bbox):
    gt_bbox = Boxes(torch.tensor([gt_bbox], dtype=torch.float32))
    pred_bbox = Boxes(torch.tensor([pred_bbox], dtype=torch.float32))

    boxiou = pairwise_iou(gt_bbox, pred_bbox)

    return boxiou


def add_R_from_lines(detection_name):

    path_precomputed_infos = '/scratch2/fml35/datasets/own_data/data_leveraging_geometry_for_shape/data_01/'
    path_selected = path_precomputed_infos + 'poses_R_selected/' + detection_name + '_000.json'

    if os.path.exists(path_selected):

        with open(path_selected,'r') as f:
            selected_R = json.load(f)

        with open(path_precomputed_infos + 'poses_R/' + detection_name + '_000_' + str(selected_R['R_index']).zfill(2) + '.json','r') as f:
            R_pred = json.load(f)["predicted_r"]

        return R_pred
    
    else:
        return None



def get_ious(bbox_pred,gt_infos,detection_infos):
    gt_img_size = gt_infos['img_size']
    rescale_array = np.array([gt_img_size[0],gt_img_size[1],gt_img_size[0],gt_img_size[1]]) / np.array([480,360,480,360])
    bbox_pred_rescaled = np.array(bbox_pred) * rescale_array

    ious = []
    indices_gt = []
    for i in range(len(gt_infos['objects'])):
        if gt_infos['objects'][i]['category'] == detection_infos['category']:
            bbox_gt = gt_infos['objects'][i]['bbox']
            iou = calculate_box_overlap(bbox_gt,bbox_pred_rescaled)
            ious.append(iou)
            indices_gt.append(i)
    
    return ious,indices_gt,bbox_pred_rescaled


def get_info_T_in_center_and_accept_reprojected_depth(gt_name,gt_index, accept_model_reprojected_depth,T_in_image):

    

    detection_gt = gt_name.split('.')[0] + '_' + str(gt_index).zfill(2) + '.json'
    accept_model_reprojected_depth_single = accept_model_reprojected_depth[detection_gt]
    T_in_image_single = T_in_image[detection_gt]

    quality_gt_object = {}
    quality_gt_object['ratio_reprojected_depth'] = accept_model_reprojected_depth_single['ratio']
    quality_gt_object['accept_depth'] = accept_model_reprojected_depth_single['accept']
    quality_gt_object['T_in_image'] = T_in_image_single['in_image']
    quality_gt_object['accept_overall'] = quality_gt_object['accept_depth'] and quality_gt_object['T_in_image']
    return quality_gt_object


def main():

    # roca_preds_path = '/scratch2/fml35/results/ROCA/per_frame_best_no_null.json'
    roca_preds_path = '/scratch2/fml35/results/ROCA/per_frame_best_no_null_correct_category_names.json'
    gt_detections_path = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/gt_infos.json'
    out_path = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/all_detection_infos_all_images.json'
    assert os.path.exists(out_path) == False, 'out_path already exists'

    dir_data = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val/'
    with open(dir_data + 'accept_model_reprojected_depth.json','r') as f:
        accept_model_reprojected_depth = json.load(f)
    with open(dir_data + 'T_in_image.json','r') as f:
        T_in_image = json.load(f)
   

    with open(roca_preds_path,'r') as f:
        roca_preds = json.load(f)
    with open(gt_detections_path,'r') as f:
        gt_detections = json.load(f)

    appended_infos = {}

    for frame in tqdm(sorted(roca_preds)):
        gt_name = frame.split('/')[0] + '-' + frame.split('/')[2].replace('.jpg','.json')
        appended_infos[gt_name] = []
        for k in range(len(roca_preds[frame])):

            detection_infos = roca_preds[frame][k]
            bbox_pred = detection_infos['bbox']
            detection_name = gt_name.split('.')[0] + '_' + str(k).zfill(2)

            if gt_name in gt_detections:
                ious,indices_gt,pred_bbox = get_ious(bbox_pred,gt_detections[gt_name],detection_infos)
                detection_infos['bbox'] = pred_bbox.tolist()
                associated_gt_infos = {key:gt_detections[gt_name][key] for key in ["img_size","focal_length","K"]}
                # print('ious',ious)
                # print('indices_gt',indices_gt)
                # print('pred_bbox',pred_bbox)
                if len(ious) > 0:
                    select_index = np.argmax(ious)
                    best_index = indices_gt[select_index]
                    for key in ["rot_mat","trans_mat","scaling","orig_q","orig_t","orig_s","model","catid","category","bbox"]:
                        associated_gt_infos[key] = gt_detections[gt_name]['objects'][best_index][key]
                    associated_gt_infos['matched_to_gt_object'] = True
                    associated_gt_infos['gt_index'] = best_index
                    associated_gt_infos['iou'] = ious[select_index].item()
                    associated_gt_infos['quality_gt_object'] = get_info_T_in_center_and_accept_reprojected_depth(gt_name,best_index,accept_model_reprojected_depth,T_in_image)

                else:
                    associated_gt_infos['matched_to_gt_object'] = False

            else:
                associated_gt_infos = {key:None for key in ["img_size","focal_length","K"]}
                associated_gt_infos['matched_to_gt_object'] = False
                detection_infos['bbox'] = None
            detection_infos['associated_gt_infos'] = associated_gt_infos
            detection_infos['detection'] = detection_name
            detection_infos['detection_id_roca'] = k

            detection_infos['r_from_lines'] = add_R_from_lines(detection_name)
            appended_infos[gt_name].append(detection_infos)
       

    with open(out_path,'w') as file:
        json.dump(appended_infos, file,indent=4)



if __name__ == '__main__':
    main()





