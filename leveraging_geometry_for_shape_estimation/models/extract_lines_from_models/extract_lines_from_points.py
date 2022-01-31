import os
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
from pytorch3d.ops import sample_points_from_meshes,knn_points
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



def extract_lines_from_points(points_optimise,dist_threshold,n_lines,n_steps,min_n_support_points):

    start_ids = torch.randint(low=0,high=points_optimise.shape[0],size=(n_lines,2))
    start = points_optimise[start_ids]
    start[:,0,:] = torch.rand((start.shape[0],3)) - 0.5
    start = start.view(-1,6)

    # print(start.shape)
    line_points = start.clone()
    device = torch.device("cuda:0")
    points_optimise = points_optimise.to(device)
    line_points = line_points.to(device)
    line_points.requires_grad = True
    # optimizer = torch.optim.SGD([line_points], lr=1e2,momentum=0.0,dampening=0, weight_decay=0, nesterov=False)
    optimizer = torch.optim.Adam([line_points], lr=1e-1)
    for step in tqdm(range(n_steps)):
        optimizer.zero_grad()
        distances,non_flat = dist_point_line(line_points,points_optimise)
        
        loss = torch.mean(distances)
        loss.backward()
        optimizer.step()

    mask = distances.unsqueeze(1).tile(1,n_lines) == non_flat
    mask_all_points = torch.sum(mask,dim=0) >= min_n_support_points
    
    extracted_lines = line_points[mask_all_points]
    remaining_points = points_optimise[distances > dist_threshold]
    return extracted_lines.cpu(),remaining_points.cpu()


def main():

    target_folder = '/scratch/fml35/datasets/pix3d_new/own_data/rendered_models/3d_lines/exp_04'
    
    n_lines = 100
    n_steps = 1000
    min_n_support_points = 5
    dist_threshold_squared = 0.02**2

    points_per_line_vis = 100

    n_repeats = 5
    
    with open("/data/cornucopia/fml35/experiments/exp_024_debug/models/model_list.json",'r') as f:
            model_list = json.load(f)["models"]

    for i in range(n_repeats):
        make_folder_check(target_folder + '/remainig_points_{}'.format(i+1))
        make_folder_check(target_folder + '/vis_lines_{}'.format(i+1))
    make_folder_check(target_folder + '/vis_lines_combined')
    # copytree('/home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/models/extract_lines_from_models',target_folder + '/code')

    for j in tqdm(range(0,len(model_list))):
        j = np.random.randint(0,len(model_list))

        for k in range(n_repeats):
            if k == 0:
                path = target_folder + '/edge_points/' + model_list[j]["category"] + '_' + model_list[j]["model"].split('/')[2] + '.ply'
            else:
                path = target_folder + '/remainig_points_{}/'.format(k) + model_list[j]["category"] + '_' + model_list[j]["model"].split('/')[2] + '.ply'

            points_optimise,_ = load_ply(path)
            extracted_lines,remaining_points = extract_lines_from_points(points_optimise,dist_threshold_squared,n_lines,n_steps,min_n_support_points)
        
            points_vis_line = sample_points_from_lines(extracted_lines,points_per_line_vis)
            save_ply(target_folder + '/vis_lines_{}/'.format(k+1) + model_list[j]["category"] + '_' + model_list[j]["model"].split('/')[2] + '.ply',points_vis_line)
            if remaining_points.shape[0] != 0:
                save_ply(target_folder + '/remainig_points_{}/'.format(k+1) + model_list[j]["category"] + '_' + model_list[j]["model"].split('/')[2] + '.ply',remaining_points)
            else:
                break

        lines = []
        for i in range(k+1):
            points_vis,_ = load_ply(target_folder + '/vis_lines_{}/'.format(i+1) + model_list[j]["category"] + '_' + model_list[j]["model"].split('/')[2] + '.ply')
            lines.append(points_vis)
        
        all_points_vis = torch.cat(lines)
        save_ply(target_folder + '/vis_lines_combined/' + model_list[j]["category"] + '_' + model_list[j]["model"].split('/')[2] + '.ply',all_points_vis)


if __name__ == '__main__':
    main()