from distutils import dep_util
from operator import mod
import cv2
import numpy as np
import sys
import os
import json
from torchvision import models
from tqdm import tqdm
from pytorch3d.renderer import look_at_view_transform

import torch

from leveraging_geometry_for_shape_estimation.keypoint_matching.get_matches_3d import load_information_depth_camera,create_pixel_bearing,pb_and_depth_to_wc,get_world_coordinates
from leveraging_geometry_for_shape_estimation.keypoint_matching.detect_keypoints import make_empty_folder_structure
from leveraging_geometry_for_shape_estimation.models.extract_lines_from_models.extract_lines_from_surface_normal_faces_shapenet_v2 import sample_points_from_lines
from pytorch3d.io import save_ply


def lines_to_pixel(lines,shape_depth):
    if len(lines.shape) == 1:
        return np.empty((0,2)),np.empty((0,2))

    else:
        assert shape_depth[0] == shape_depth[1]
        pixels = np.round(lines).astype(int)
        pixels = np.clip(pixels,a_min=0,a_max=shape_depth[1]-5)
        pixels_1 = pixels[:,:2]
        pixels_2 = pixels[:,2:4]

        pixels_1 = pixels_1[:,::-1]
        pixels_2 = pixels_2[:,::-1]

        return pixels_1,pixels_2


def get_3d_wc_for_folder(models_folder_read,output_dir,output_dir_vis,fov,W,device,line_exp):

    print('GETTING ALL LINES ? If not within 4 pixel, line removed')

    P_proj = load_information_depth_camera(fov)
    pb_x,pb_y,pb_z = create_pixel_bearing(W,W,P_proj,device)

    for cat in tqdm(sorted(os.listdir(models_folder_read + '/models/extract_from_2d/exp_01_filter_5/lines_2d'))):
        for model in tqdm(sorted(os.listdir(models_folder_read + '/models/extract_from_2d/exp_01_filter_5/lines_2d/' + cat))):
            for orientation in sorted(os.listdir(models_folder_read + '/models/extract_from_2d/exp_01_filter_5/lines_2d/' + cat + '/' + model)):

                out_path = output_dir + '/' + cat + '/' + model + '/' + orientation
                # print(out_path)
                out_path_vis = output_dir_vis + '/' + cat + '/' + model + '/' + orientation.replace('.npy','.ply')
                if os.path.exists(out_path_vis):
                    continue
        
                depth_path = models_folder_read + '/models/depth/' + cat + '/' + model + '/' + orientation
                depth = np.load(depth_path)
                depth = torch.Tensor(depth).to(device)

                # keypoints = np.load(models_folder_read + '/models/keypoints/' + cat + '/' + model + '/' + orientation.replace('.npy','.npz'))
                lines_2d_path = models_folder_read + '/models/extract_from_2d/'+ line_exp + '/lines_2d/' + cat + '/' + model + '/' + orientation

                elev = orientation.split('_')[1]
                azim = orientation.split('_')[3].replace('.npy','')
                mask = (depth > -1)

                lines_2d = np.load(lines_2d_path)
                if len(lines_2d.shape) == 1:
                    np.save(out_path,np.empty((0,6)))
                    save_ply(out_path_vis,torch.empty((0,3)))

                else:
                    pixel_1,pixel_2 = lines_to_pixel(lines_2d,depth.shape)
                    wc_grid = pb_and_depth_to_wc(pb_x,pb_y,pb_z,depth,elev,azim,mask,device)
                    wc_grid = wc_grid.cpu()


                    wc_1 = get_world_coordinates(wc_grid,pixel_1)
                    wc_2 = get_world_coordinates(wc_grid,pixel_2)
                    lines_3d = np.concatenate((wc_1,wc_2),axis=1)


                    mask = np.all(lines_3d > -1000,axis=1)
                    lines_3d = lines_3d[mask,:]
                    np.save(out_path,lines_3d)
                    points = sample_points_from_lines(torch.Tensor(lines_3d),100)
                    save_ply(out_path_vis,points)





def main():

    global_info = sys.argv[1] + '/global_information.json'
    line_exp = sys.argv[2]

    with open(global_info,'r') as f:
        global_config = json.load(f)

    models_folder_read = global_config["general"]["models_folder_read"]
    fov = global_config["models"]["fov"]
    W = global_config["models"]["img_size"]

    device = torch.device("cuda:{}".format(global_config["general"]["gpu"]))

    output_dir = models_folder_read + '/models/extract_from_2d/{}/lines_3d/'.format(line_exp)
    output_dir_vis = models_folder_read + '/models/extract_from_2d/{}/lines_3d_vis/'.format(line_exp)

    # make_empty_folder_structure(models_folder_read + '/models/extract_from_2d/{}/lines_2d/'.format(line_exp),output_dir)
    # make_empty_folder_structure(models_folder_read + '/models/extract_from_2d/{}/lines_2d/'.format(line_exp),output_dir_vis)


    torch.cuda.set_device(device)

    get_3d_wc_for_folder(models_folder_read,output_dir,output_dir_vis,fov,W,device,line_exp)
    

    



if __name__ == '__main__':
    main()
