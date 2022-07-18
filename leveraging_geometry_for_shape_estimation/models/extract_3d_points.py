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
from pytorch3d.io import save_ply

def get_3d_wc_for_folder(models_folder_read,fov,W,device):

    P_proj = load_information_depth_camera(fov)
    pb_x,pb_y,pb_z = create_pixel_bearing(W,W,P_proj,device)

    for cat in tqdm(os.listdir(models_folder_read + '/models/depth')):
        for model in tqdm(os.listdir(models_folder_read + '/models/depth/' + cat)):
            for orientation in os.listdir(models_folder_read + '/models/depth/' + cat + '/' + model):

                out_path = models_folder_read + '/models/3d_points_individual/' + cat + '/' + model + '/' + orientation.replace('.npy','.ply')
        
                depth_path = models_folder_read + '/models/depth/' + cat + '/' + model + '/' + orientation
                depth = np.load(depth_path)
                depth = torch.Tensor(depth).to(device)

                keypoints = np.load(models_folder_read + '/models/keypoints/' + cat + '/' + model + '/' + orientation.replace('.npy','.npz'))

                elev = orientation.split('_')[1]
                azim = orientation.split('_')[3].replace('.npy','')

                mask = (depth > -1)

                wc_grid = pb_and_depth_to_wc(pb_x,pb_y,pb_z,depth,elev,azim,mask,device)
                wc_grid = wc_grid.cpu()

                pixels_rendered = keypoints['pts']
                pixels_rendered = np.transpose(pixels_rendered)[:,:2]
                # print(pixels_rendered.shape)
                pixels_rendered = pixels_rendered[:,::-1]
                pixels_rendered = np.round(pixels_rendered).astype(int)
                # print(pixels_rendered)
                # print(pixels_rendered.shape)

                # print(pixels_rendered)

                wc = get_world_coordinates(wc_grid,pixels_rendered)
 


                mask = np.all(wc > -1000,axis=1)
                # print(mask.shape)
                wc = wc[mask,:]

                # print(wc.shape)
                # print(wc)
                # print(df)

                # print(out_path)
                save_ply(out_path,torch.from_numpy(wc))





def main():

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    models_folder_read = global_config["general"]["models_folder_read"]
    fov = global_config["models"]["fov"]
    W = global_config["models"]["img_size"]

    device = torch.device("cuda:{}".format(global_config["general"]["gpu"]))

    make_empty_folder_structure(models_folder_read + '/models/keypoints/',models_folder_read + '/models/3d_points_individual/')
    torch.cuda.set_device(device)

    get_3d_wc_for_folder(models_folder_read,fov,W,device)
    

    



if __name__ == '__main__':
    main()
