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

def combine_3d_wc_for_folder(models_folder_read):

    for cat in tqdm(os.listdir(models_folder_read + '/models/depth')):
        for model in tqdm(os.listdir(models_folder_read + '/models/depth/' + cat)):

            all_points = []
            for orientation in os.listdir(models_folder_read + '/models/depth/' + cat + '/' + model):

                inpath = models_folder_read + '/models/3d_points/3d_points_individual_orientation/' + cat + '/' + model + '/' + orientation.replace('.npy','.ply')
                all_points.append(load_ply(inpath)[0])

            out_path = models_folder_read + '/models/3d_points/3d_points_all_combined/' + cat + '/' + model + '.ply'

            save_ply(out_path,torch.cat(all_points))

    
def filter_3d_wc_for_folder(input_dir,output_dir,threshold,min_support):

    for cat in tqdm(os.listdir(input_dir)):
        for model in tqdm(os.listdir(input_dir + '/' + cat)):

            in_path = input_dir + '/' + cat + '/' + model
            out_path = output_dir + '/' + cat + '/' + model
            all_points,_ = load_ply(in_path)
            filtered_points = []

            while True:

                point,all_points = filter_points(all_points,threshold,min_support)
                if point is None:
                    break
                filtered_points.append(point)
            
            filtered_points = torch.cat(filtered_points)
            print(filtered_points.shape[0])
            save_ply(out_path,filtered_points)


def filter_points(points,threshold,min_support):

    n = points.shape[0]



    A = torch.repeat_interleave(points,repeats=n,dim=0)
    B = points.repeat(n,1)
    dists = torch.sum((A-B)**2,dim=1)
    dists = dists.view(n,n)

    # print('points',points.shape[0])
    # print(dists[:4,:4])

    dists_smaller_threshold = dists <= threshold

    N_in_threshold = torch.sum(dists_smaller_threshold,dim=1)

    # print('N_in_threshold',N_in_threshold[:6])

    best_index = torch.argmax(N_in_threshold)
    # print('best_index',best_index)
    # print('support ',torch.sum(dists[best_index] <= threshold))
    if N_in_threshold[best_index] < min_support:
        return None,None

    else:
        # point = points[best_index:best_index+1]
        point = torch.mean(points[dists[best_index] <= threshold],dim = 0 ).unsqueeze(0)
        remaining_points = points[dists[best_index] > threshold]
        return point,remaining_points




def main():


    device = torch.device("cuda:0")

    torch.cuda.set_device(device)

    # combine_3d_wc_for_folder(models_folder_read)
    input_dir = '/scratch/fml35/experiments/leveraging_geometry_for_shape/test_output_all_s2/models/3d_points/3d_points_all_combined'
    output_dir = '/scratch/fml35/experiments/leveraging_geometry_for_shape/test_output_all_s2/models/3d_points/3d_points_filtered_02_mean'
    # threshold = 0.05**2
    # min_support = 16

    threshold = 0.03**2
    min_support = 16


    filter_3d_wc_for_folder(input_dir,output_dir,threshold,min_support)
    

    



if __name__ == '__main__':
    main()