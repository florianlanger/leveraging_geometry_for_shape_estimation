

from tqdm import tqdm
import os
import json
import cv2
import numpy as np
from numpy.lib.utils import info
import torch
import os
import sys
import json
import pandas as pd


def combine_information(global_config):

    target_folder = global_config["general"]["target_folder"]

    counter = 0

    name_columns = ['img','gt_category','img_width','img_height','gt_bbox','gt_model_name','focal_length','gt_rotation','gt_translation']
    name_columns += ['pred_category','segmentation_score','pred_bbox']
    name_columns += ['box_iou','mask_iou']
    name_columns += ['pred_model_name','pred_elev','pred_azim']
    name_columns += ['matches_distances']
    name_columns += ['pred_rotation','pred_translation']
    metrics_list = ["Chamfer-L2","modified_Hausdorf","NormalConsistency","AbsNormalConsistency","Precision@0.300000","Recall@0.300000","F1@0.300000","Precision@0.500000","Recall@0.500000","F1@0.500000","Precision@0.700000","Recall@0.700000","F1@0.700000","F1","total_angle_diff","diff_tilt","diff_azim","diff_elev","diff_absolute_length","diff_absolute_distances","diff_normalised_length","diff_normalised_distances"]
    name_columns += metrics_list


    dir_list = os.listdir(target_folder + '/selected_nn')
    name_rows = [name.split('.')[0] for name in dir_list]
    df = pd.DataFrame(columns=name_columns, index=name_rows)

    for name in tqdm(os.listdir(target_folder + '/selected_nn')):
        with open(target_folder + '/selected_nn/' + name,'r') as f:
            selected = json.load(f)
        number_nn = selected["selected_nn"]
        
        with open(target_folder + '/gt_infos/' + name.rsplit('_',1)[0] + '.json','r') as f:
            gt_infos = json.load(f)

        name_pose = name.split('.')[0] + '_' + str(number_nn).zfill(3) + '_' + str(selected["selected_orientation"]).zfill(2) + '.json'

        with open(target_folder + '/poses/' + name_pose,'r') as f:
            poses = json.load(f)


        with open(target_folder + '/segmentation_infos/' + name,'r') as f:
            segmentation_infos = json.load(f)

        with open(target_folder + '/bbox_overlap/' + name,'r') as f:
            bbox_overlap = json.load(f)

        with open(target_folder + '/nn_infos/' + name,'r') as f:
            nn_infos = json.load(f)

        if os.path.exists(target_folder + '/matches_quality/' + name):
            with open(target_folder + '/matches_quality/' + name,'r') as f:
                matches_quality = json.load(f)
        else:
            matches_quality = {}
            matches_quality["distances"] = 1000

        with open(target_folder + '/metrics/' + name_pose,'r') as f:
            metrics = json.load(f)

    # dir_list = os.listdir(target_folder + '/metrics')
    # name_rows = [name.rsplit('_',2)[0] for name in dir_list]
    # df = pd.DataFrame(columns=name_columns, index=name_rows)

    # for name in tqdm(dir_list):
        
    #     with open(target_folder + '/gt_infos/' + name.rsplit('_',3)[0] + '.json','r') as f:
    #         gt_infos = json.load(f)

        # with open(target_folder + '/segmentation_infos/' + name.rsplit('_',2)[0] + '.json','r') as f:
        #     segmentation_infos = json.load(f)

        # with open(target_folder + '/bbox_overlap/' + name.rsplit('_',2)[0] + '.json','r') as f:
        #     bbox_overlap = json.load(f)

        # with open(target_folder + '/nn_infos/' + name.rsplit('_',2)[0] + '.json','r') as f:
        #     nn_infos = json.load(f)

        # if os.path.exists(target_folder + '/matches_quality/' + name.rsplit('_',2)[0] + '.json'):
        #     with open(target_folder + '/matches_quality/' + name.rsplit('_',2)[0] + '.json','r') as f:
        #         matches_quality = json.load(f)
        # else:
        #     matches_quality = {}
        #     matches_quality["distances"] = 1000

        # with open(target_folder + '/poses/' + name,'r') as f:
        #     poses = json.load(f)

        # with open(target_folder + '/metrics/' + name,'r') as f:
        #     metrics = json.load(f)


        infos_gt = [gt_infos["img"],gt_infos["category"],gt_infos["img_size"][0],gt_infos["img_size"][1],gt_infos["bbox"],gt_infos["model"],gt_infos["focal_length"],gt_infos["rot_mat"],gt_infos["trans_mat"]]
        infos_segmetation = [segmentation_infos["predictions"]["category"],segmentation_infos["predictions"]["score"],segmentation_infos["predictions"]["bbox"]]
        infos_segmentation_eval = [bbox_overlap["box_iou"],bbox_overlap["mask_iou"]]
        infos_nn = [nn_infos["nearest_neighbours"][0]["model"],float(nn_infos["nearest_neighbours"][0]["elev"]),float(nn_infos["nearest_neighbours"][0]["azim"])]
        infos_matches = [matches_quality["distances"]]
        infos_poses = [poses["predicted_r"],poses["predicted_t"]]
        infos_metrics = [metrics[metric] for metric in metrics_list]      
        df.iloc[counter] = infos_gt + infos_segmetation + infos_segmentation_eval + infos_nn + infos_matches + infos_poses + infos_metrics
    
        counter += 1

    df.to_csv(target_folder + '/global_stats/all_infos.csv')
        

            





def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    combine_information(global_config)

if __name__ == '__main__':
    main()