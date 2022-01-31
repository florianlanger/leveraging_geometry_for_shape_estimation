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
from pytorch3d.io import save_ply,load_ply

def combine_3d_lines_and_points(dir_1,dir_2,dir_save):

    for cat_model in tqdm(sorted(os.listdir(dir_1))):
        
        lines,_ = load_ply(dir_1 + '/' + cat_model)
        points,_ = load_ply(dir_2 + '/' + cat_model.split('_')[0] + '/' + cat_model.replace(cat_model.split('_')[0] + '_','')) 

        n_points = points.shape[0]

        values = torch.linspace(-0.01,0.01,4)
        x,y,z = torch.meshgrid([values,values,values])

        offsets = torch.cat([x.unsqueeze(-1),y.unsqueeze(-1),z.unsqueeze(-1)],dim=-1)

        points_repeated = torch.repeat_interleave(points,4**3,dim=0)
        offsets_repeated = offsets.view(-1,3).repeat(n_points,1)

        new_points = points_repeated + offsets_repeated

        all_vis = torch.cat([new_points,lines])

        save_ply(dir_save + '/' + cat_model,all_vis)


def main():



    # combine_3d_wc_for_folder(models_folder_read)
    dir_1 = '/scratch/fml35/datasets/pix3d_new/own_data/rendered_models/3d_lines/exp_26_points_on_edges_angle_20_lines_one_and_three_face/edge_points'
    dir_2 = '/scratch/fml35/experiments/leveraging_geometry_for_shape/test_output_all_s2/models/3d_points/3d_points_filtered_02_mean'
    dir_save = '/scratch/fml35/datasets/pix3d_new/own_data/rendered_models/3d_lines/exp_26_points_on_edges_angle_20_lines_one_and_three_face/vis_with_points'

    # os.mkdir(dir_save)
    combine_3d_lines_and_points(dir_1,dir_2,dir_save)
    

    



if __name__ == '__main__':
    main()