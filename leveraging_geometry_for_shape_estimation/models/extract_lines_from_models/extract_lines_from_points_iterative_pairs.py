import os
from leveraging_geometry_for_shape_estimation.pose_and_shape_optimisation.pose_selection import compute_selection_metric
import numpy as np
import sys
import socket
import pickle
import math
import os
import json
from numpy.core.defchararray import mod
from numpy.lib.function_base import append
from tqdm import tqdm
import torch
import k3d
from shutil import copytree
import time

from pytorch3d.io import load_obj,save_ply,load_ply
from pytorch3d.ops import sample_points_from_meshes,knn_points,sample_farthest_points
from pytorch3d.structures import Pointclouds,Meshes

def make_folder_check(path,path2=None):
    if not os.path.exists(path):
        os.mkdir(path)
    
    if path2 != None:
        if not os.path.exists(path2):
            os.mkdir(path2)

def sample_points_from_lines(lines,points_per_line):
    n_lines = lines.shape[0]
    lines = torch.repeat_interleave(lines,points_per_line,dim=0)
    interval = torch.linspace(0,1,points_per_line).repeat(n_lines)
    interval = interval.unsqueeze(1).repeat(1,3)
    points = lines[:,:3] + (lines[:,3:6] - lines[:,:3]) * interval
    return points

def dist_point_line(lp,p):
    # lp: N_lines x 2 (x1,y1,x2,y2) and p: N_points x 2
    N_l = lp.shape[0]
    N_p = p.shape[0]

    x1 = lp[:,0].repeat(N_p)
    y1 = lp[:,1].repeat(N_p)
    z1 = lp[:,2].repeat(N_p)
    x2 = lp[:,3].repeat(N_p)
    y2 = lp[:,4].repeat(N_p)
    z2 = lp[:,5].repeat(N_p)
    x0 = p[:,0].repeat_interleave(N_l)
    y0 = p[:,1].repeat_interleave(N_l)
    z0 = p[:,2].repeat_interleave(N_l)


    px = x2-x1
    py = y2-y1
    pz = z2-z1

    norm = px*px + py*py + pz*pz

    # print('norm',norm)

    u =  ((x0 - x1) * px + (y0 - y1) * py + (z0 - z1) * pz) / norm

    u = torch.clip(u,min=0,max=1)

    # print('u',u)

    x = x1 + u * px
    y = y1 + u * py
    z = z1 + u * pz

    dx = x - x0
    dy = y - y0
    dz = z - z0



    dist_non_flattened = (dx*dx + dy*dy + dz*dz)#**.5

    # print('dist_non_flatten',dist_non_flattened)
    # dist = torch.abs(dx) + torch.abs(dy) + torch.abs(dz)

    # dist = torch.abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / torch.sqrt((x2-x1)**2 + (y2 -y1)**2)
    dist_non_flattened = dist_non_flattened.reshape(N_p,N_l)
    dist,_ = torch.min(dist_non_flattened,dim=1)

    return dist,dist_non_flattened

def compute_support(lp,points,n_support_interval,dist_threshold_squared,device):
    N = points.shape[0]

    interval  = torch.linspace(0,1,n_support_interval).unsqueeze(0).unsqueeze(-1).tile(lp.shape[0],1,3).to(device)
    sampled_points_on_line = lp.unsqueeze(1).tile(1,n_support_interval,1)[:,:,:3] + interval * (lp.unsqueeze(1).tile(1,n_support_interval,1)[:,:,3:6] - lp.unsqueeze(1).tile(1,n_support_interval,1)[:,:,:3])

    sampled_points_flattened = sampled_points_on_line.view(-1,3)
    sampled_points_flattened_repeated = sampled_points_flattened.repeat_interleave(repeats=N,dim=0)
    points_repeated = points.repeat(N*n_support_interval,1)

    distances = torch.sum((sampled_points_flattened_repeated - points_repeated)**2,dim=1)
    distances = distances.view(N*n_support_interval,N)
    min_distances,_ = torch.min(distances,dim=1)
    min_distances = min_distances.view(N,n_support_interval)
    supported = min_distances < dist_threshold_squared

    percent_supported = torch.sum(supported,dim=1) / n_support_interval
    return percent_supported


def find_line(points,dist_threshold_squared,min_n_support,device,n_support_interval,min_percentage_line_support):
    N = points.shape[0]
    # print(N)
    if N == 0 or len(points.shape) == 1:
        return None,points

    # print(points.shape)
    start_id = torch.randint(low=0,high=N,size=(1,)).to(device)
    lp1 = points[start_id].tile(N,1)
    lp = torch.cat([lp1,points],dim=1)
    distances,non_flat = dist_point_line(lp,points)
    thresholded = non_flat < dist_threshold_squared
    n_support = torch.sum(thresholded,dim=0)
    percent_supported = compute_support(lp,points,n_support_interval,dist_threshold_squared,device)
    # print(percent_supported)
    supported_mask = percent_supported > min_percentage_line_support
    # n_support is how many points are within dist_threshold_squared of line and percent supported is if sample line evenly n_support_interval-times how many of those points within dist_threshold_squared of pointcloud,.i.e. if have 0.6 that 60% of line is supported
    # set those lines that have support over required length to have 0 n_support such that not selected
    # print(supported_mask[:20])
    # print(n_support[:20])
    n_support[~supported_mask] = n_support[~supported_mask] * 0
    # print(n_support[:20])
    best_index = torch.argmax(n_support)
    # print(d)
    # check support lp N x 4
    # want N x 10 x 2 where have


    # now repeat but
    lp1 = points[best_index].tile(N,1)
    lp = torch.cat([lp1,points],dim=1)
    distances,non_flat = dist_point_line(lp,points)
    thresholded_2 = non_flat < dist_threshold_squared
    n_support = torch.sum(thresholded_2,dim=0)

    percent_supported = compute_support(lp,points,n_support_interval,dist_threshold_squared,device)
    supported_mask = percent_supported > min_percentage_line_support
    n_support[~supported_mask] = n_support[~supported_mask] * 0

    sorted_support,indices = torch.sort(n_support,descending=True)

    # print('sorted',sorted[:10])
    # print('indices',indices[:10])

    found = False
    for i in range(N):
        test_index = indices[i]
        # print(test_index)
        # IDEA: use double distance or triple to allow for better lines, ie. 2x threshold
        if thresholded_2[start_id,test_index] == True and sorted_support[i] >= min_n_support:
            # print('N support',sorted_support[i])
            found = True
            break

    if found == False:
        return None,points
        
    else:
        # print('best_index',best_index)
        # print('test_index',test_index)
        line = torch.cat([points[best_index],points[test_index]])
        # print('line',line)
        remaining_points = points[thresholded[:,best_index].nonzero()]
        remaining_points_2 = points[(~thresholded_2[:,test_index]).nonzero()]

        return line,remaining_points_2.squeeze()

def main():

    target_folder = '/scratch/fml35/datasets/pix3d_new/own_data/rendered_models/3d_lines/exp_16_points_on_edges'
    
    device = torch.device("cuda:0")

    dist_threshold_squared = 0.0015**2
    min_n_support = 10
    min_percentage_line_support = 0.8
    n_support_interval = 10
    n_lines = 400
    # percent_point_farthest = 0.3
    torch.manual_seed(5)

    points_per_line_vis = 100
    
    with open("/data/cornucopia/fml35/experiments/exp_024_debug/models/model_list.json",'r') as f:
            model_list = json.load(f)["models"]

    make_folder_check(target_folder)
    make_folder_check(target_folder + '/vis_lines')
    make_folder_check(target_folder + '/lines')
    make_folder_check(target_folder + '/farthest_points')
    make_folder_check(target_folder + '/original_points')

    # copytree('/home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/models/extract_lines_from_models',target_folder + '/code')

    for j in tqdm(range(0,len(model_list))):
        # path = '/scratch/fml35/datasets/pix3d_new/own_data/rendered_models/3d_lines/exp_04/edge_points/' + model_list[j]["category"] + '_' + model_list[j]["model"].split('/')[2] + '.ply'
        path = '/scratch/fml35/datasets/pix3d_new/own_data/rendered_models/3d_lines/exp_16_points_on_edges/edge_points/' + model_list[j]["category"] + '_' + model_list[j]["model"].split('/')[2] + '.ply'
        remaining_points,_ = load_ply(path)
        remaining_points = remaining_points.to(device)
        save_ply(target_folder + '/original_points/' + model_list[j]["category"] + '_' + model_list[j]["model"].split('/')[2] + '.ply',remaining_points)

        remaining_points,_ = sample_farthest_points(remaining_points.unsqueeze(0),K = 1000)#int(remaining_points.shape[0]*percent_point_farthest))
        remaining_points = remaining_points.squeeze()
        save_ply(target_folder + '/farthest_points/' + model_list[j]["category"] + '_' + model_list[j]["model"].split('/')[2] + '.ply',remaining_points)

        lines = torch.zeros((n_lines,6)).to(device)

        counter = 0
        for i in range(n_lines):
            with torch.no_grad():
                line,remaining_points = find_line(remaining_points,dist_threshold_squared,min_n_support,device,n_support_interval,min_percentage_line_support)
                if line != None:
                    lines[counter] = line
                    counter += 1

        lines = lines[:counter]
        points_vis_line = sample_points_from_lines(lines.cpu(),points_per_line_vis)
        save_ply(target_folder + '/vis_lines/' + model_list[j]["category"] + '_' + model_list[j]["model"].split('/')[2] + '.ply',points_vis_line)
        np.save(target_folder + '/lines/' + model_list[j]["category"] + '_' + model_list[j]["model"].split('/')[2] + '.npy',lines.cpu().numpy())

if __name__ == '__main__':
    main()