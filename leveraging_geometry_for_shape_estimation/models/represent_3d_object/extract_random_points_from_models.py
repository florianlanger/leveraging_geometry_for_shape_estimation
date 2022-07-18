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
import random

from pytorch3d.io import load_obj,save_ply
from pytorch3d.ops import sample_points_from_meshes,knn_points,sample_farthest_points
from pytorch3d.structures import Pointclouds,Meshes

def make_folder_check(path,path2=None):
    if not os.path.exists(path):
        os.mkdir(path)
    
    if path2 != None:
        if not os.path.exists(path2):
            os.mkdir(path2)


def load_points_normals(path,n_points):
    verts_torch,faces_torch,_ = load_obj(path,load_textures=False)
    verts_fixed,faces_fixed = repair_mesh(verts_torch,faces_torch[0])
    mesh = Meshes(verts=[verts_fixed],faces=[faces_fixed])
    sample_points,sample_normals = sample_points_from_meshes(mesh,n_points,return_normals=True)
    sample_points = sample_points.squeeze()
    sample_normals = sample_normals.squeeze()
    return sample_points,sample_normals


def repair_mesh(verts,faces):
    # print(verts.shape)
    # print('faces',faces.shape)
    vertices = verts.numpy()
    faces = faces.numpy()
    # vertices, faces = pymeshfix.clean_from_arrays(verts.numpy(), faces.numpy())
    # print('vertices',vertices.shape)
    # print('faces',faces.shape)
    # Create object from vertex and face arrays
    # meshfix = pymeshfix.MeshFix(verts, faces)

    # # Repair input mesh
    # meshfix.repair()

    # # Or, access the resulting arrays directly from the object
    # vertices = meshfix.v # numpy np.float64 array
    # faves = meshfix.f # numpy np.int32 array



    # mesh = trimesh.Trimesh(vertices=verts.numpy(), faces=faces.numpy())
    # # mesh.remove_duplicate_faces()
    # broken = trimesh.repair.broken_faces(mesh)
    # print('broken',broken)
    # trimesh.repair.fix_normals(mesh)
    # trimesh.repair.fix_inversion(mesh)
    # trimesh.repair.fill_holes(mesh)
    # broken = trimesh.repair.broken_faces(mesh)
    # print('broken',broken)
    # mesh.remove_duplicate_faces()
    # faces = mesh.faces
    # vertices = mesh.vertices

    return torch.Tensor(vertices),torch.from_numpy(faces)

def plot_points(points,point_size=0.003):
    plot = k3d.plot()
    plot += k3d.points(points, point_size=point_size, shader="flat")
    plot.display()

def vis_representative_normals(points,normals,mesh=None,in_notebook=True):
    assert points.shape[0] == normals.shape[0]

    all_points = []
    for i in range(normals.shape[1]):
        lines = torch.cat([points,points+normals[:,i]*0.05],dim=1)
        # print(lines.shape)
        vis_points = sample_points_from_lines(lines,points_per_line=30)
        vis_points = vis_points.view(-1,3)
        all_points.append(vis_points)

    all_points = torch.cat(all_points,dim=0)
    if not in_notebook == True:
        return all_points
    
    else:
        plot = k3d.plot()
        plot += k3d.points(all_points, point_size=0.003,color=0xff0000, shader="flat")
        if mesh is not None:
            plot += mesh
        plot.display()

def different_normals_around_points(query_points,ref_points,ref_normals,n_nn=20,n_normals_per_point=3):

    knn_pred = knn_points(query_points.unsqueeze(0),ref_points.unsqueeze(0), K=n_nn)
    normals = ref_normals[knn_pred.idx[0]]

    K = n_normals_per_point*torch.ones(normals.shape[0]).long().to(get_device(query_points))
    representative_normals,_ = sample_farthest_points(normals,K=K)
    return representative_normals

def get_device(a):

    if a.is_cuda:
        gpu_n = a.get_device()
        device = torch.device("cuda:{}".format(gpu_n))
    else:
        device = torch.device("cpu")
    return device



def save_output(furthest_points,representative_normals,out_dir,model_name):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_path = out_dir + 'points_and_normals/' + model_name + '.npz'

    np.savez(out_path, points=furthest_points.cpu().numpy(), normals=representative_normals.cpu().numpy())


def add_random_points_to_edge_points(all_points,edge_points,N_random_points):
    device = get_device(edge_points)
    indices_random = np.random.choice(range(0,all_points.shape[0]),size=(N_random_points),replace=False)
    assert len(indices_random) == len(set(indices_random))
    random_points = all_points[indices_random].to(device)
    combined_points = torch.cat((edge_points,random_points),dim=0)
    return combined_points    



def main():

    # out_dir = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/3d_future/data_01/models/representation_points_and_normals/exp_01/'
    out_dir = '/scratch/fml35/datasets/shapenet_v2/ShapeNetRenamed/representation_points_and_normals/exp_12_3000_random_same_normals/'
    # shape_dir = '/scratch/fml35/datasets/3d_future/3D-FUTURE-model_reformatted/'
    shape_dir = '/scratch/fml35/datasets/shapenet_v2/ShapeNetRenamed/'
    # model_list_path = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/3d_future/data_01/models/model_list_full.json'
    model_list_path = '/scratches/octopus_2/fml35/experiments/leveraging_geometry_for_shape/exp_117_scannet_models/models/model_list.json'



    n_random_points = 3000


    
    os.mkdir(out_dir)
    os.mkdir(out_dir+'points_and_normals/')
    os.mkdir(out_dir+'vis')

    with open(model_list_path,'r') as f:
        model_list = json.load(f)["models"]
    
    device = torch.device("cuda:0")

    list_indices = list(range(len(model_list)))
    # shuffle
    random.shuffle(list_indices)

    for j in tqdm(list_indices):
    # for j in tqdm(range(len(model_list)-1,-1,-1)):

        if os.path.exists(out_dir + 'vis/' + model_list[j]["name"] + '.ply'):
            continue


        model_path = shape_dir + model_list[j]["model"].replace('model/','model_fixed/')
        if not os.path.exists(model_path):
            print('model path does not exist',model_path)
            continue

        sample_points,sample_normals = load_points_normals(model_path,n_random_points)
        sample_normals = sample_normals.unsqueeze(1).repeat(1,3,1)

        save_output(sample_points,sample_normals,out_dir,model_list[j]["name"])
    


if __name__ == '__main__':
    with torch.no_grad():
        main()