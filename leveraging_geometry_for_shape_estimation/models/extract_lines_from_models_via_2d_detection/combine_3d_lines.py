from operator import mod
import cv2
import numpy as np
import sys
import os
import json
from torchvision import models
from tqdm import tqdm
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.ops import knn_gather, knn_points
import torch

from leveraging_geometry_for_shape_estimation.keypoint_matching.get_matches_3d import load_information_depth_camera,create_pixel_bearing,pb_and_depth_to_wc,get_world_coordinates
from leveraging_geometry_for_shape_estimation.keypoint_matching.detect_keypoints import make_empty_folder_structure
from leveraging_geometry_for_shape_estimation.utilities.folders import make_dir_save
from pytorch3d.io import save_ply,load_ply

def combine_3d_wc_for_folder(models_folder_read,individual_dir,combined_dir):

    for cat in tqdm(sorted(os.listdir(models_folder_read + '/models/extract_from_2d/exp_01_filter_5/lines_2d'))):
        for model in tqdm(sorted(os.listdir(models_folder_read + '/models/extract_from_2d/exp_01_filter_5/lines_2d/' + cat))):

            all_points = []
            all_lines = []
            for orientation in sorted(os.listdir(models_folder_read + '/models/extract_from_2d/exp_01_filter_5/lines_2d/' + cat + '/' + model)):

                inpath = individual_dir + '_vis/' + cat + '/' + model + '/' + orientation.replace('.npy','.ply')

                points = load_ply(inpath)[0]
                if not points.shape[0] == 0:
                    all_points.append(points)

                lines = np.load(individual_dir + '/' + cat + '/' + model + '/' + orientation)
                if not lines.shape[0] == 0:
                    all_lines.append(lines)

            out_path = combined_dir + '/' + cat + '/' + model + '.npy'
            out_path_vis = combined_dir + '_vis/' + cat + '/' + model + '.ply'

            if all_lines == []:
                assert all_points == []
                np.save(out_path,np.empty((0,6)))
                save_ply(out_path_vis,torch.empty((0,3)))

            else:        
                save_ply(out_path_vis,torch.cat(all_points))
                np.save(out_path,np.concatenate(all_lines))

def main():


    global_info = sys.argv[1] + '/global_information.json'
    line_exp = sys.argv[2]

    with open(global_info,'r') as f:
        global_config = json.load(f)

    models_folder_read = global_config["general"]["models_folder_read"]

    individual_dir = models_folder_read + '/models/extract_from_2d/{}/lines_3d'.format(line_exp)
    combined_dir = models_folder_read + '/models/extract_from_2d/{}/lines_3d_combined'.format(line_exp)
    
    # make_dir_save(combined_dir)
    # make_dir_save(combined_dir + '_vis')
            
    # for cat in sorted(os.listdir(individual_dir)):
    #     make_dir_save(combined_dir + '/' + cat)
    #     make_dir_save(combined_dir + '_vis' + '/' + cat)

    combine_3d_wc_for_folder(models_folder_read,individual_dir,combined_dir)

if __name__ == '__main__':
    main()