

from cmath import nan
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

from leveraging_geometry_for_shape_estimation.eval.scannet.metrics_roca import rotation_diff_considering_symmetry,load_cad_info
from leveraging_geometry_for_shape_estimation.eval.scannet import CSVHelper
from leveraging_geometry_for_shape_estimation.utilities.dicts import open_json_precomputed_or_current,determine_base_dir,load_json


def get_min_of_four_rotations(R_gt,sym,path_minus_rotation):
    all_angles = []
    for rot in range(4):
        path = path_minus_rotation + '_' + str(rot).zfill(2) + '.json'
        with open(path,'r') as f:
            rot_info = json.load(f)
        R_pred = rot_info["predicted_r"]
        rotation_error = rotation_diff_considering_symmetry(R_pred,R_gt,sym)
        all_angles.append(rotation_error)

    return all_angles,min(all_angles)


def get_n_lines(name,global_config):
    lines = np.load(determine_base_dir(global_config,'lines') + '/lines_2d_filtered/' + name.split('.')[0] + '.npy')
    unique_lines = np.unique(lines, axis=0)
    return unique_lines.shape[0]

def get_info_eval_3d(name,eval_info_3d):

    detection = name.split('.')[0]
    scene = detection.split('-')[0]

    detections_scan2cad_constraints = [alignment[14] for alignment in eval_info_3d['scan2cad_constraint_without_retrieval'][scene]]
    detections_after_filter = [alignment[14] for alignment in eval_info_3d['filtered'][scene]]

    if detection in detections_scan2cad_constraints:
        index = detections_scan2cad_constraints.index(detection)
        infos = eval_info_3d['scan2cad_constraint_without_retrieval'][scene][index]
        overall_with_retrieval = eval_info_3d['scan2cad_constraint_with_retrieval'][scene][index][15]
        overall_with_retrieval_tolerant = eval_info_3d['scan2cad_constraint_with_retrieval'][scene][index][23]
        infos_eval = [True,True,infos[15],overall_with_retrieval,infos[16],infos[17],infos[18],infos[19],infos[20],infos[21],infos[22],infos[23],overall_with_retrieval_tolerant,infos[24],infos[25]]
        

    elif detection in detections_after_filter:
        infos_eval = [True,False,"na","na","na","na","na","na","na","na","na","na","na","na","na"]

    else:
        infos_eval = [False,False,"na","na","na","na","na","na","na","na","na","na","na","na","na"]
    return infos_eval


def load_filtered_and_flags(target_folder):
    eval_info_3d = {}
    eval_info_3d['filtered'] = {}
    eval_info_3d['scan2cad_constraint_without_retrieval'] = {}
    eval_info_3d['scan2cad_constraint_with_retrieval'] = {}
    for file0 in os.listdir(target_folder + '/global_stats/eval_scannet/results_per_scene_filtered'):
        scene = file0.split('.')[0]
        eval_info_3d['filtered'][scene] = CSVHelper.read(target_folder + '/global_stats/eval_scannet/results_per_scene_filtered/' + file0)
        eval_info_3d['scan2cad_constraint_without_retrieval'][scene] = CSVHelper.read(target_folder + '/global_stats/eval_scannet/results_per_scene_flags_without_retrieval/' + file0)
        eval_info_3d['scan2cad_constraint_with_retrieval'][scene] = CSVHelper.read(target_folder + '/global_stats/eval_scannet/results_per_scene_flags_with_retrieval/' + file0)
        # eval_info_3d['scan2cad_constraint_without_retrieval'][scene] = CSVHelper.read(target_folder + '/global_stats/eval_scannet/results_per_scene_flags_tolerant_rotation_without_retrieval/' + file0)
        # eval_info_3d['scan2cad_constraint_with_retrieval'][scene] = CSVHelper.read(target_folder + '/global_stats/eval_scannet/results_per_scene_flags_tolerant_rotation_with_retrieval/' + file0)
    return eval_info_3d


def get_metrics_files(target_folder,names_correct_lines_metrics):
    metrics_files = []
    for name in names_correct_lines_metrics:
        with open(target_folder + '/global_stats/count_2d_lines_correct/' + name + '.json') as f:
            metrics_files.append(json.load(f))
    return metrics_files

def get_column_names_correct_lines(names_correct_lines_metrics):
    col_names = []
    for name in names_correct_lines_metrics:
        col_names += [name + '_correct',name + '_total']
    return col_names


def get_number_correct_lines(name,metrics_files):
    name_expanded = name.split('.')[0] + '_000_00.json'
    numbers = []
    for file in metrics_files:
        numbers += [file[name_expanded]["n_correct"],file[name_expanded]["n_total"]]
    return numbers


def combine_information(global_config):

    target_folder = global_config["general"]["target_folder"]

    counter = 0

    # name_columns = ['detections','img','gt_category']
    name_columns = ['img','gt_category']
    name_columns += ['pred_category','segmentation_score']
    metrics_list = ["F1@0.300000","F1@0.500000","F1@0.700000"]
    metrics_scannet_list = ["rotation_error","translation_error"]
    name_columns += metrics_list
    name_columns += metrics_scannet_list
    name_columns += ["lines_exist","min_angle_four_rotations","rotation_1","rotation_2","rotation_3","rotation_4","number_2d_filtered_lines"]

    # names_correct_lines_metrics = ['T_lines_vis_quality_2d_lines_2_overlap_04_n2','T_lines_vis_quality_2d_lines_3_overlap_04_n1','T_lines_vis_quality_2d_lines_4_overlap_03_both','T_lines_vis_quality_2d_lines_6_absolute_threshold','T_lines_vis_quality_2d_lines_7_absolute_threshold_smaller','T_lines_vis_quality_2d_lines_8_absolute_threshold_smaller_filtered','T_lines_vis_quality_2d_lines_9_absolute_threshold_filtered']
    # name_columns += get_column_names_correct_lines(names_correct_lines_metrics)

    name_columns += ['survived_filtering','used_for_evaluation','overall_without_retrieval','overall_with_retrieval','translation_correct','rotation_correct','scaling_correct','retrieval_correct','translation_error','rotation_error','scaling_error','overall_without_retrieval_tolerant_rotation','overall_with_retrieval_tolerant_rotation','rotation_correct_tolerant','rotation_error_tolerant']

    cad2info = load_cad_info(global_config["dataset"]["dir_path_images"] + '../scan2cad_annotations/full_annotations.json')
    eval_info_3d = load_filtered_and_flags(target_folder)


    dir_list = sorted(os.listdir(target_folder + '/selected_nn'))
    # dir_list = sorted(os.listdir('/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_178_gt_retrieval_matches_gt_z/selected_nn'))
    name_rows = [name.split('.')[0] for name in dir_list]
    df = pd.DataFrame(columns=name_columns, index=name_rows)
    # print(len(name_columns))
    # df = pd.DataFrame(columns=name_columns)

    # metrics_files = get_metrics_files(target_folder,names_correct_lines_metrics)

    for name in tqdm(dir_list):
    # print('only same objects as exp_178')
    # for name in tqdm(os.listdir('/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_178_gt_retrieval_matches_gt_z/selected_nn')):
        with open(target_folder + '/selected_nn/' + name,'r') as f:
            selected = json.load(f)
        number_nn = selected["selected_nn"]
        
        # with open(target_folder + '/gt_infos/' + name.rsplit('_',1)[0] + '.json','r') as f:
        #     gt_infos = json.load(f)
        gt_infos = open_json_precomputed_or_current('/gt_infos/' + name.rsplit('_',1)[0] + '.json',global_config,'segmentation')

        # with open(target_folder + '/bbox_overlap/' + name,'r') as f:
        #     bbox_overlap = json.load(f)
        bbox_overlap = open_json_precomputed_or_current('/bbox_overlap/' + name,global_config,'segmentation')

        gt_object = gt_infos["objects"][bbox_overlap['index_gt_objects']]

        name_pose = name.split('.')[0] + '_' + str(number_nn).zfill(3) + '_' + str(selected["selected_orientation"]).zfill(2) + '.json'

        # with open(target_folder + '/poses/' + name_pose,'r') as f:
        #     poses = json.load(f)
        # with open(target_folder + '/nn_infos/' + name ,'r') as f:
        #     nn_infos = json.load(f)
        nn_infos = open_json_precomputed_or_current('/nn_infos/' + name,global_config,'retrieval')
        segmentation_infos = open_json_precomputed_or_current('/segmentation_infos/' + name,global_config,'segmentation')

        # with open(target_folder + '/segmentation_infos/' + name,'r') as f:
        #     segmentation_infos = json.load(f)


        if os.path.exists(target_folder + '/metrics/' + name_pose):
            with open(target_folder + '/metrics/' + name_pose,'r') as f:
                metrics = json.load(f)
        else:
            metrics = {metric:None for metric in metrics_list}


        with open(target_folder + '/metrics_scannet/' + name_pose,'r') as f:
            metrics_scannet = json.load(f)


        model_name = nn_infos["nearest_neighbours"][number_nn]["name"]
        path = "/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_117_scannet_models/models/lines/" + model_name + '.npy'
        if os.path.exists(path):
            lines_exist = True
        else:
            lines_exist = False
        sym = cad2info[model_name]["sym"]
        path_minus_rotation = determine_base_dir(global_config,'R') + '/poses_R/' + name.split('.')[0] + '_' + str(number_nn).zfill(3)
        all_angles,min_four_rotations = get_min_of_four_rotations(gt_object["rot_mat"],sym,path_minus_rotation)
        number_2d_lines = get_n_lines(name,global_config)

        # number_correct_lines = get_number_correct_lines(name,metrics_files)




        info_eval_3d = get_info_eval_3d(name,eval_info_3d)

        # infos_gt = [name.split('.')[0],gt_infos["img"],gt_object["category"]]
        infos_gt = [gt_infos["img"],gt_object["category"]]
        infos_segmetation = [segmentation_infos["predictions"]["category"],segmentation_infos["predictions"]["score"]]
        infos_metrics = [metrics[metric] for metric in metrics_list] 
        infos_metrics_scannet = [metrics_scannet[metric] for metric in metrics_scannet_list]
        # infos_others = [lines_exist,min_four_rotations]  + all_angles  + [number_2d_lines] + number_correct_lines
        infos_others = [lines_exist,min_four_rotations]  + all_angles  + [number_2d_lines]


        df.iloc[counter] = infos_gt + infos_segmetation + infos_metrics + infos_metrics_scannet + infos_others + info_eval_3d
        # df.iloc[counter] = infos_gt + infos_segmetation + infos_metrics + infos_metrics_scannet + info_eval_3d
    
        counter += 1

    df.to_csv(target_folder + '/global_stats/all_infos_small_v5.csv')
        

            





def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    combine_information(global_config)

if __name__ == '__main__':
    main()