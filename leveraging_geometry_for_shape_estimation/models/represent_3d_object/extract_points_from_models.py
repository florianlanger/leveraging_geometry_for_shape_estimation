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

def find_edge_points(sample_points,sample_normals,n_nn,angle_threshold_degree,device):

    assert sample_normals.shape[0] == sample_points.shape[0]

    knn_pred = knn_points(sample_points.unsqueeze(0).to(device),sample_points.unsqueeze(0).to(device), K=n_nn)

    sample_normals_unsq = sample_normals.unsqueeze(1)
    sample_normals_tiled = sample_normals_unsq.tile(1,n_nn,1)

    nn_normals = torch.zeros(sample_points.shape[0],n_nn,3)
    for i in range(sample_points.shape[0]):
        n = sample_normals[knn_pred.idx[0,i].cpu()]
        nn_normals[i] = n
        
    normal_dot = torch.sum(sample_normals_tiled*nn_normals,dim=2)
    angles = torch.arccos(normal_dot) * 180 / np.pi
    angles = torch.min(angles,torch.abs(180. - angles))
    # TODO: think if better 180 or 360
    # angles = torch.min(angles,torch.abs(360. - angles))
    max_angles,_ = torch.max(angles,dim=1)
    labels = (max_angles > angle_threshold_degree)
    return sample_points[labels]

def load_points_normals(path,n_points):
    verts_torch,faces_torch,_ = load_obj(path,load_textures=False)
    verts_fixed,faces_fixed = repair_mesh(verts_torch,faces_torch[0])
    mesh = Meshes(verts=[verts_fixed],faces=[faces_fixed])
    sample_points,sample_normals = sample_points_from_meshes(mesh,n_points,return_normals=True)
    sample_points = sample_points.squeeze()
    sample_normals = sample_normals.squeeze()
    return sample_points,sample_normals

def sample_points_from_lines(lines,points_per_line):

    device = get_device(lines)

    n_lines = lines.shape[0]
    lines = torch.repeat_interleave(lines,points_per_line,dim=0)
    interval = torch.linspace(0,1,points_per_line).repeat(n_lines).to(device)
    interval = interval.unsqueeze(1).repeat(1,3)
    points = lines[:,:3] + (lines[:,3:6]-lines[:,:3]) * interval
    points = points.view(n_lines,points_per_line,3)
    return points

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



def save_output(furthest_points,representative_normals,out_dir,model_name,vis_points=None):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_path = out_dir + 'points_and_normals/' + model_name + '.npz'
    # print('furthest_points',furthest_points.shape)
    # print('representative_normals',representative_normals.shape)
    np.savez(out_path, points=furthest_points.cpu().numpy(), normals=representative_normals.cpu().numpy())

    if vis_points != None:
        vis_path = out_dir + 'vis/' + model_name + '.ply'
        save_ply(vis_path,vis_points.cpu())


def add_random_points_to_edge_points(all_points,edge_points,N_random_points):
    device = get_device(edge_points)
    indices_random = np.random.choice(range(0,all_points.shape[0]),size=(N_random_points),replace=False)
    assert len(indices_random) == len(set(indices_random))
    random_points = all_points[indices_random].to(device)
    combined_points = torch.cat((edge_points,random_points),dim=0)
    return combined_points    



def main():


    # out_dir = '/scratches/octopus_2/fml35/experiments/leveraging_geometry_for_shape/exp_117_scannet_models/models/representation_points_and_normals/exp_01/'
    # shape_dir = '/scratches/octopus_2/fml35/datasets/shapenet_v2/ShapeNetRenamed/'
    # model_list_path = '/scratches/octopus_2/fml35/experiments/leveraging_geometry_for_shape/exp_117_scannet_models/models/model_list.json'

    # out_dir = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/3d_future/data_01/models/representation_points_and_normals/exp_01/'
    out_dir = '/scratch/fml35/datasets/shapenet_v2/ShapeNetRenamed/representation_points_and_normals/exp_06_600_random/'
    # shape_dir = '/scratch/fml35/datasets/3d_future/3D-FUTURE-model_reformatted/'
    shape_dir = '/scratch/fml35/datasets/shapenet_v2/ShapeNetRenamed/'
    # model_list_path = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/3d_future/data_01/models/model_list_full.json'
    model_list_path = '/scratches/octopus_2/fml35/experiments/leveraging_geometry_for_shape/exp_117_scannet_models/models/model_list.json'


    # n_points_start = 50000
    # n_points_ref = 50000
    # n_points_end = 150

    n_points_start = 1000
    n_points_ref = 0
    n_points_end = 1

    n_random_points_add = 599

    n_nn_for_finding_edge_points = 5
    angle_threshold_degree = 20
    n_nn_different_normals = 20
    n_normals_per_point = 3


    
    # os.mkdir(out_dir)
    # os.mkdir(out_dir+'points_and_normals/')
    # os.mkdir(out_dir+'vis')

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

        # if model_list[j]["name"] == "table_7e101ef3-7722-4af8-90d5-7c562834fabd":
        #     continue

        model_path = shape_dir + model_list[j]["model"].replace('model/','model_fixed/')
        if not os.path.exists(model_path):
            print('model path does not exist',model_path)
            continue

        # print(model_path)

        sample_points,sample_normals = load_points_normals(model_path,n_points_start)
        edge_points = find_edge_points(sample_points,sample_normals,n_nn_for_finding_edge_points,angle_threshold_degree,device)


        if edge_points.shape[0] != 0:
            furthest_points,indices = sample_farthest_points(edge_points.unsqueeze(0).to(device),K=torch.Tensor([n_points_end]).to(device))
            furthest_points = furthest_points.squeeze(0)

            combined_points = add_random_points_to_edge_points(sample_points,furthest_points,n_random_points_add)
            # ref_points,ref_normals = load_points_normals(scannet_path + model_list[j]["model"],n_points_ref)
            representative_normals = different_normals_around_points(combined_points,sample_points.to(device),sample_normals.to(device),n_nn=n_nn_different_normals,n_normals_per_point=n_normals_per_point)
            vis_points = vis_representative_normals(combined_points,representative_normals,in_notebook=False)

        else:
            combined_points = torch.Tensor([])
            representative_normals = torch.Tensor([])
            vis_points = torch.Tensor([])


        save_output(combined_points,representative_normals,out_dir,model_list[j]["name"],vis_points)
    


if __name__ == '__main__':
    with torch.no_grad():
        main()