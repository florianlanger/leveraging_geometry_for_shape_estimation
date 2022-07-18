import cv2
import numpy as np
import sys
import os
import json
from torchvision import models
from tqdm import tqdm
from pytorch3d.renderer import look_at_view_transform
from math import ceil
import torch
from pytorch3d.io import load_obj
from scipy.spatial.transform import Rotation as scipy_rot

from leveraging_geometry_for_shape_estimation.keypoint_matching.get_matches_3d import load_information_depth_camera,create_pixel_bearing,pb_and_depth_to_wc
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.pose import init_Rs,init_Ts,get_pb_real_grid,get_R_limits,get_T_limits,get_nearest_pose_to_gt_all_R,check_gt_pose_in_limits,create_pose_info_dict
from leveraging_geometry_for_shape_estimation.utilities.dicts import load_json
from leveraging_geometry_for_shape_estimation.pose_and_shape_optimisation.select_best_v2 import get_angle
from leveraging_geometry_for_shape_estimation.utilities.dicts import open_json_precomputed_or_current

from probabilistic_formulation.utilities import create_all_possible_combinations,get_uvuv_p_from_superpoint,create_all_possible_combinations_uvuv_p_together
from probabilistic_formulation.factors.factor_R import get_factor_reproject_lines_single_R,get_factor_reproject_lines_multiple_R
from probabilistic_formulation.tests.test_reproject_lines import load_lines_2D,load_lines_3D,get_cuboid_line_dirs_3D,plot_vp_orig_size




def find_R_from_retrieval(R,nn_infos):

    transform_Rs  = [scipy_rot.from_euler('zyx',[0,0,0], degrees=True).as_matrix(),
                        scipy_rot.from_euler('zyx',[0,90,0], degrees=True).as_matrix(),
                        scipy_rot.from_euler('zyx',[0,180,0], degrees=True).as_matrix(),
                        scipy_rot.from_euler('zyx',[0,270,0], degrees=True).as_matrix()]

    elev = float(nn_infos["elev"])
    azim = float(nn_infos["azim"])

    retrieved_R = scipy_rot.from_euler('zyx',[0,180-azim,-elev], degrees=True).as_matrix()
    rotated_Rs = [np.matmul(R,transform_R) for transform_R in transform_Rs]
    angles = [get_angle(rotated_R,retrieved_R) for rotated_R in rotated_Rs]
    best_R = rotated_Rs[np.argmin(angles)]
    print(np.min(angles))

    return best_R


def find_closest_R_index(Rs,comparison_R):
    angles = [get_angle(R,comparison_R) for R in Rs]
    closest_index = int(np.argmin(angles))
    return closest_index,angles

def get_comparison_R(global_config,R_infos,retrieval_name,retrieval_index):
    if global_config["R"]["choose_R"] == "closest_retrieved":

        retrieval_list = open_json_precomputed_or_current(retrieval_name,global_config,'retrieval')["nearest_neighbours"]
        nn_infos = retrieval_list[retrieval_index]
        elev = float(nn_infos["elev"])
        azim = float(nn_infos["azim"])
        comparison_R = scipy_rot.from_euler('zyx',[0,180-azim,-elev], degrees=True).as_matrix()

    elif global_config["R"]["choose_R"] == "closest_gt":
        comparison_R = R_infos[0]["gt_R"]

    return comparison_R



def get_pose_for_folder(global_config):

    target_folder = global_config["general"]["target_folder"]

    for name in tqdm(sorted(os.listdir(target_folder + '/poses_R'))):
    # for name in tqdm(sorted(os.listdir('/scratch2/fml35/datasets/own_data/data_leveraging_geometry_for_shape/data_01/poses_R'))):
        
        if not "00.json" in name:
            continue
    
        R_infos = []
        for i in range(4):
            R_infos.append(load_json(target_folder + '/poses_R/' + name.replace('00.json',str(i).zfill(2) + '.json')))
            # R_infos.append(load_json('/scratch2/fml35/datasets/own_data/data_leveraging_geometry_for_shape/data_01/poses_R/' + name.replace('00.json',str(i).zfill(2) + '.json')))

        Rs = [R_info["predicted_r"] for R_info in R_infos]
    
        retrieval_name = '/nn_infos/' + name.split('.')[0] + '.json'
        retrieval_index = int(name.rsplit('_',2)[1].split('_')[0])

        assert retrieval_index == 0

        comparison_R = get_comparison_R(global_config,R_infos,retrieval_name,retrieval_index)

        R_index,angles = find_closest_R_index(Rs,comparison_R)
          
        output_path = target_folder + '/poses_R_selected/' + name.rsplit('_',1)[0] + '.json'
        out_info = {"R_index":R_index,"angles":angles}

        with open(output_path,'w') as open_f:
            json.dump(out_info,open_f,indent=4)

           



def main():


    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    if global_config["pose_and_shape_probabilistic"]["use_probabilistic"] == "True" and global_config["pose_and_shape_probabilistic"]["pose"]["gt_R"] == False:
        get_pose_for_folder(global_config)
    


if __name__ == '__main__':
    print('Select Rotation')
    main()

