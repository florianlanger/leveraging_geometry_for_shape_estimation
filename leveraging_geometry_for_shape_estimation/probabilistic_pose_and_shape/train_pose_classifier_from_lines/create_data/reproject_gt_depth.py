import numpy as np
import cv2
from regex import R
import torch
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj
import pytorch3d
import trimesh

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (PerspectiveCameras,look_at_view_transform,FoVPerspectiveCameras,RasterizationSettings, MeshRenderer, MeshRasterizer,SoftPhongShader,Textures)
import sys
import os
import json 
from pytorch3d.io import save_ply
from tqdm import tqdm
import random


def convert_K(K,width,height):
    K[0,2] = - (K[0,2] - width/2)
    K[1,2] = - (K[1,2] - height/2)
    K = K/(width/2)
    K[2:4,2:4] = torch.Tensor([[0,1],[1,0]])
    return K

def depth_gt_object(data,device,path_to_remeshed,w,h,f,K):

    r_cam = torch.eye(3).unsqueeze(0).repeat(1,1,1)
    t_cam = torch.zeros((1,3))

    # cameras_pix = FoVPerspectiveCameras(device=device,fov = fov,degrees=False,R = r_cam, T = t_cam)
    cameras_pix = PerspectiveCameras(device=device,K = K.unsqueeze(0),T=t_cam,R=r_cam)
    raster_settings_soft = RasterizationSettings(image_size = max(w,h),blur_radius=0, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=cameras_pix,raster_settings=raster_settings_soft)


    # load gt mesh
    # vertices,faces,_ = load_obj(obj_full_path, device=device,create_texture_atlas=False, load_textures=False)
    gt_obj = load_obj(path_to_remeshed + data["model"].replace('model/',''), device=device,create_texture_atlas=False, load_textures=False)
    gt_vertices_origin,gt_faces,gt_properties = gt_obj
    R_gt = torch.Tensor(data["rot_mat"]).to(device) #.inverse().to(device)
    T_gt = torch.Tensor(data["trans_mat"]).to(device)
    S_gt = torch.Tensor(data['scaling']).to(device)
    gt_vertices_origin = gt_vertices_origin * S_gt
    gt_vertices = torch.transpose(torch.matmul(R_gt,torch.transpose(gt_vertices_origin,0,1)),0,1) + T_gt
    textures_gt = Textures(verts_rgb=torch.ones((1,gt_vertices.shape[0],3),device=device))
    mesh = Meshes(verts=[gt_vertices], faces=[gt_faces[0]],textures=textures_gt)


    # max_edge_length = 0.05
    # vertices_remeshed,faces_remeshed = trimesh.remesh.subdivide_to_size(mesh.verts_list()[0].cpu().numpy(), mesh.faces_list()[0].cpu().numpy(), max_edge=max_edge_length)
    # vertices = torch.from_numpy(vertices_remeshed).to(torch.float32).to(device)
    # faces = torch.from_numpy(faces_remeshed).to(device)
    # mesh = Meshes(verts=[vertices], faces=[faces])
    _,depth,_,_ = rasterizer(mesh,cameras=cameras_pix)
    depth = depth[0,:,:,0]



    if w >= h:
        depth = depth[int((w-h)/2):int((w+h)/2),:]
    elif w < h:
        depth = depth[:,int((h-w)/2):int((h+w)/2)]
    return depth

def save_depth(path,depth,target_size):

    depth = depth.cpu().numpy() 
    depth = cv2.resize(depth,(target_size[0],target_size[1]))

    depth = np.round(depth * 1000)
    depth[depth < 0] = depth[depth < 0] * 0 + 65535
    depth = depth.astype(np.uint16)
    # print(depth[300:305,600:605])
    # print(depth[1,1])
    cv2.imwrite(path,depth)


if __name__ == '__main__':


    path_to_remeshed = '/scratches/octopus_2/fml35/experiments/leveraging_geometry_for_shape/exp_117_scannet_models/models/remeshed/'
    # path_to_remeshed = '/scratches/octopus_2/fml35/datasets/shapenet_v2/ShapeNetRenamed/model/'
    gpu = 3
    target_folder = '/scratches/octopus/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/train'
    target_size = (640,480)

    device = torch.device("cuda:{}".format(gpu))

    # for img in tqdm(sorted(os.listdir(target_folder + '/gt_infos/'))):
    list_dir = os.listdir(target_folder + '/gt_infos/')
    random.shuffle(list_dir)

    for img in tqdm(list_dir):
        # for img in tqdm(os.listdir(target_folder + '/gt_infos/')[::-1]):
        with open(target_folder + '/gt_infos/' + img) as f:
            gt_info = json.load(f)
        
        

        w,h = gt_info["img_size"]
        f = gt_info["focal_length"]

        K = convert_K(torch.Tensor(gt_info["K"]),w,h)
        for i,object_info in enumerate(gt_info["objects"]):

            save_path = target_folder + '/reprojected_gt_depth/' + img.split('.')[0] + '_' + str(i).zfill(2) + '.png'
            if os.path.exists(save_path):
                continue

            depth = depth_gt_object(object_info,device,path_to_remeshed,w,h,f,K)

            save_depth(save_path,depth,target_size)
