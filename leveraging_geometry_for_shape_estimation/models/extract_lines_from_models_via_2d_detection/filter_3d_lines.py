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
from leveraging_geometry_for_shape_estimation.models.extract_lines_from_models.extract_lines_from_surface_normal_faces_shapenet_v2 import sample_points_from_lines
from pytorch3d.io import save_ply,load_ply

def filter_3d_line_for_folder(input_dir,output_dir,threshold,min_support):
    for cat in tqdm(sorted(os.listdir(input_dir))):
        for model in tqdm(sorted(os.listdir(input_dir + '/' + cat))):
            
            in_path = input_dir + '/' + cat + '/' + model
            out_path = output_dir + '/' + cat + '/' + model
            out_path_vis = output_dir + '_vis/' + cat + '/' + model.replace('.npy','.ply')

            if os.path.exists(out_path_vis):
                continue

            all_lines = np.load(in_path)
            all_lines = torch.Tensor(all_lines)
            if all_lines.shape[0] == 0:
                np.save(out_path,np.empty((0,6)))
                save_ply(out_path_vis,torch.empty((0,3)))
                continue
                
            filtered_lines = []

            while True:
                line,all_lines = filter_points(all_lines,threshold,min_support)
                if line is None:
                    break
                filtered_lines.append(line)

            if filtered_lines == []:
                np.save(out_path,np.empty((0,6)))
                save_ply(out_path_vis,torch.empty((0,3)))
            
            else:
                filtered_lines = torch.cat(filtered_lines)
                np.save(out_path,filtered_lines.numpy())

                points = sample_points_from_lines(filtered_lines,100)
                save_ply(out_path_vis,points)


def filter_points(lines,threshold,min_support):

    n = lines.shape[0]


    A = torch.repeat_interleave(lines,repeats=n,dim=0)
    B = lines.repeat(n,1)
    dists_1 = torch.sum((A[:,:3]-B[:,:3])**2,dim=1)
    dists_1 = dists_1.view(n,n)

    dists_2 = torch.sum((A[:,3:6]-B[:,3:6])**2,dim=1)
    dists_2 = dists_2.view(n,n)

    dists_smaller_threshold = torch.bitwise_and(dists_1 <= threshold,dists_2 <= threshold)

    N_in_threshold = torch.sum(dists_smaller_threshold,dim=1)


    best_index = torch.argmax(N_in_threshold)

    if N_in_threshold[best_index] < min_support:
        return None,None

    else:
        # point = points[best_index:best_index+1]
        mask = torch.bitwise_and(dists_1[best_index] <= threshold,dists_2[best_index] <= threshold)

        line = torch.mean(lines[mask],dim = 0 ).unsqueeze(0)
        remaining_lines = lines[~mask]
        return line,remaining_lines

def main():


    global_info = sys.argv[1] + '/global_information.json'
    line_exp = sys.argv[2]
    min_support =int(sys.argv[3])
    with open(global_info,'r') as f:
        global_config = json.load(f)


    models_folder_read = global_config["general"]["models_folder_read"]

    input_dir = models_folder_read + '/models/extract_from_2d/{}/lines_3d_combined'.format(line_exp)
    output_dir = models_folder_read + '/models/extract_from_2d/{}/lines_3d_filtered'.format(line_exp)
    
    # make_dir_save(output_dir)
    # make_dir_save(output_dir + '_vis')
            
    # for cat in os.listdir(input_dir):
    #     make_dir_save(output_dir + '/' + cat)
    #     make_dir_save(output_dir + '_vis' + '/' + cat)

    threshold = 0.03**2

    filter_3d_line_for_folder(input_dir,output_dir,threshold,min_support)



if __name__ == '__main__':
    main()