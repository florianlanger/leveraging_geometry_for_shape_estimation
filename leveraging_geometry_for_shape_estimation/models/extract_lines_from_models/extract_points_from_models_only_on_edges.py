import os
import numpy as np
import sys
import socket
import pickle
import math
import os
import json
from tqdm import tqdm
import torch
import k3d
from shutil import copytree

from pytorch3d.io import load_obj,save_ply
from pytorch3d.ops import sample_points_from_meshes,knn_points
from pytorch3d.structures import Pointclouds,Meshes

def make_folder_check(path,path2=None):
    if not os.path.exists(path):
        os.mkdir(path)
    
    if path2 != None:
        if not os.path.exists(path2):
            os.mkdir(path2)

def find_edge_points(sample_points,sample_normals,points_on_edges,n_nn,angle_threshold_degree):

    assert sample_normals.shape[0] == sample_points.shape[0]

    # print('sample_points,sample_normals,points_on_edges',sample_points.shape,sample_normals.shape,points_on_edges.shape)

    device = torch.device("cuda:0")
    knn_pred = knn_points(points_on_edges.unsqueeze(0).to(device),sample_points.unsqueeze(0).to(device), K=n_nn)

    # sample_normals_unsq = sample_normals.unsqueeze(1)
    # sample_normals_tiled = sample_normals_unsq.tile(1,n_nn,1)

    # nn_normals = torch.zeros(sample_points.shape[0],n_nn,3)
    nn_normals = torch.zeros(points_on_edges.shape[0],n_nn,3)
    for i in range(points_on_edges.shape[0]):
        n = sample_normals[knn_pred.idx[0,i].cpu()]
        nn_normals[i] = n

    # print(nn_normals.shape)
    sample_normals_tiled = nn_normals[:,:1,:].tile(1,n_nn,1)
        
    normal_dot = torch.sum(sample_normals_tiled*nn_normals,dim=2)
    angles = torch.arccos(normal_dot) * 180 / np.pi
    angles = torch.min(angles,torch.abs(180. - angles))
    # TODO: think if better 180 or 360
    # angles = torch.min(angles,torch.abs(360. - angles))
    max_angles,_ = torch.max(angles,dim=1)
    labels = (max_angles > angle_threshold_degree)
    return points_on_edges[labels]

def load_points_normals(path,n_points):
    verts_torch,faces_torch,_ = load_obj(path,load_textures=False)
    mesh = Meshes(verts=[verts_torch],faces=[faces_torch[0]])
    sample_points,sample_normals = sample_points_from_meshes(mesh,n_points,return_normals=True)
    sample_points = sample_points.squeeze()
    sample_normals = sample_normals.squeeze()
    return sample_points,sample_normals

def get_lines(verts,faces):
    line_segs = np.zeros((faces.shape[0]*3,6))
    print(line_segs.shape)
    counter = 0
    checked_pairs = []
    for i in range(faces.shape[0]):
        for pair in [[0,1],[1,2],[2,0]]:
            indices = [faces[i,pair[0]],faces[i,pair[1]]]
            if indices not in checked_pairs and indices[::-1] not in checked_pairs:
                line_segs[counter] = np.concatenate((verts[indices[0]],verts[indices[1]]))
                checked_pairs.append(indices)
                counter += 1

    return line_segs

def sample_points_from_lines(lines,points_per_line):
    n_lines = lines.shape[0]
    lines = torch.repeat_interleave(lines,points_per_line,dim=0)
    interval = torch.linspace(0,1,points_per_line).repeat(n_lines)
    interval = interval.unsqueeze(1).repeat(1,3)
    points = lines[:,:3] + (lines[:,3:6]-lines[:,:3]) * interval
    return points

def load_points_on_edges(path):
    verts_torch,faces_torch,_ = load_obj(path,load_textures=False)
    faces_torch = faces_torch[0]
    lines = get_lines(verts_torch,faces_torch)
    lines = torch.Tensor(lines)
    points_on_edges = sample_points_from_lines(lines,30)
    return points_on_edges


def main():

    n_points = 50000
    n_nn = 5
    angle_threshold_degree = 20
    pix_path = '/scratch/fml35/datasets/pix3d_new/'
    target_folder = '/scratch/fml35/datasets/pix3d_new/own_data/rendered_models/3d_lines/exp_16_points_on_edges'

    with open("/data/cornucopia/fml35/experiments/exp_024_debug/models/model_list.json",'r') as f:
            model_list = json.load(f)["models"]

    make_folder_check(target_folder)
    make_folder_check(target_folder + '/edge_points')
    # copytree('/home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/models/extract_lines_from_models',target_folder + '/code')

    for j in tqdm(range(0,len(model_list))):
        points_on_edges = load_points_on_edges(pix_path + model_list[j]["model"])
        save_ply(target_folder + '/edge_points/' + model_list[j]["category"] + '_' + model_list[j]["model"].split('/')[2] + '_all_lines.ply',points_on_edges)
        sample_points,sample_normals = load_points_normals(pix_path + model_list[j]["model"],n_points)
        edge_points = find_edge_points(sample_points,sample_normals,points_on_edges,n_nn,angle_threshold_degree)

        save_ply(target_folder + '/edge_points/' + model_list[j]["category"] + '_' + model_list[j]["model"].split('/')[2] + '.ply',edge_points)


if __name__ == '__main__':
    main()