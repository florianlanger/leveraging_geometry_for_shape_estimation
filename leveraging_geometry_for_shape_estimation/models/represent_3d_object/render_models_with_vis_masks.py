
print('start importing')
from typing import no_type_check_decorator
from yaml import load
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
from PIL import Image
import trimesh

import numpy as np
import random

from tqdm import tqdm
import cv2

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj
import pytorch3d

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (look_at_view_transform,FoVPerspectiveCameras,PerspectiveCameras,PointLights, DirectionalLights, Materials, 
    RasterizationSettings,MeshRenderer, MeshRasterizer,  SoftPhongShader,SoftSilhouetteShader,SoftPhongShader,TexturesVertex,Textures,HardPhongShader,)
# add path for demo utils functions 
import sys
import os
import torch
import json

from pytorch3d.utils import ico_sphere

from PIL import Image
from scipy.spatial.transform import Rotation as scipy_rot


def make_empty_folder_structure(inputpath,outputpath):
    for dirpath, dirnames, filenames in os.walk(inputpath):
        structure = os.path.join(outputpath, dirpath[len(inputpath):])
        if not os.path.isdir(structure):
            os.mkdir(structure)
        else:
            print("Folder {} already exists!".format(structure))

def create_setup(R,T,W,H,focal_length):
    sigma = 1e-4
    raster_settings = RasterizationSettings(image_size=W,blur_radius=0.0, faces_per_pixel=1)
    assert W==H,'same width and hieght'
    materials = Materials(device=device,specular_color=[[0.0, 0.0, 0.0]],shininess=1.0)
    image_size = ((W,H),)
    principal_point = ((W/2, H/2),)
    cameras = [PerspectiveCameras(device=device,T=torch.from_numpy(T).unsqueeze(0),R=torch.from_numpy(R).unsqueeze(0),focal_length=focal_length,image_size = image_size,principal_point=principal_point)]
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)


    raster_settings_soft = RasterizationSettings(image_size = max(W,H),blur_radius=0, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=cameras,raster_settings=raster_settings_soft)
    # renderer_textured = MeshRenderer(rasterizer=rasterizer,shader=SoftPhongShader(device=device))

    renderer_textured = MeshRenderer(rasterizer=rasterizer,shader=HardPhongShader(device=device, cameras=cameras[0]))
    P_full = cameras[0].get_full_projection_transform().get_matrix()
    P_proj = cameras[0].get_projection_transform().get_matrix()
    return cameras,rasterizer,renderer_textured,P_full,P_proj

def get_name(total_index,elev,azim):
    elev_index = total_index // len(azim)
    # NOTE: complicated formula because of old convention
    # azim_index = (len(azim) - total_index % len(azim)) % len(azim)
    azim_index = total_index % len(azim)
    elev_current = str(int(elev[elev_index])).zfill(3)
    azim_current = str(np.round(azim[azim_index],1)).zfill(3)

    name = 'elev_{}_azim_{}.npy'.format(elev_current,azim_current)
    return name

def load_points_and_masks(dir,name):

    # load points and masks
    points = np.load(dir+'/points_and_normals/'+name+'.npz')['points']
    masks = np.load(dir +'/masks/' + name + '.npz')
    return points,masks

def combine_vertices_and_faces_with_points(vertices,faces,points,device):
    points = torch.from_numpy(points).to(device)

    sphere = ico_sphere()

    verts_sphere = torch.Tensor(sphere.verts_list()[0]).to(device) * 0.01
    faces_sphere = sphere.faces_list()[0].to(device)
    n_verts_sphere = verts_sphere.shape[0]
    n_points = points.shape[0]

    verts_sphere = verts_sphere.unsqueeze(0).repeat(n_points,1,1) + points.unsqueeze(1).repeat(1,n_verts_sphere,1)
    faces_sphere = faces_sphere.unsqueeze(0).repeat(n_points,1,1) + torch.arange(0,n_points).to(device).unsqueeze(1).unsqueeze(2).repeat(1,faces_sphere.shape[0],3) * n_verts_sphere

    verts_sphere = verts_sphere.view(-1,3)
    faces_sphere = faces_sphere.view(-1,3)

    vertices_all = torch.cat([verts_sphere,vertices])

    faces_all = torch.cat([faces_sphere,faces + verts_sphere.shape[0]])

    texture_colors = torch.cat([verts_sphere*0 + 0.3,vertices*0 + 0.8])

    return vertices_all,faces_all,texture_colors

def get_vertices_and_faces_for_points(points,device):
    points = torch.from_numpy(points).to(device)

    sphere = ico_sphere()

    verts_sphere = torch.Tensor(sphere.verts_list()[0]).to(device) * 0.01
    faces_sphere = sphere.faces_list()[0].to(device)
    n_verts_sphere = verts_sphere.shape[0]
    n_points = points.shape[0]

    verts_sphere = verts_sphere.unsqueeze(0).repeat(n_points,1,1) + points.unsqueeze(1).repeat(1,n_verts_sphere,1)
    faces_sphere = faces_sphere.unsqueeze(0).repeat(n_points,1,1) + torch.arange(0,n_points).to(device).unsqueeze(1).unsqueeze(2).repeat(1,faces_sphere.shape[0],3) * n_verts_sphere

    verts_sphere = verts_sphere.view(-1,3)
    faces_sphere = faces_sphere.view(-1,3)

    # print('faces_sphere',faces_sphere.shape)

    return verts_sphere,faces_sphere,n_points,n_verts_sphere



def get_colors(total_index,elev,azim,masks,n_points,n_vertices_sphere):
    
    names = []
    for i in range(4):
        names.append(get_name(total_index + i,elev,azim))
    
    masks = [masks[name] for name in names]
    masks = np.stack(masks,axis=0)

    masks = torch.from_numpy(masks)

    masks = torch.repeat_interleave(masks,n_vertices_sphere,dim=1)

    texture_colors = torch.Tensor([[[0,0,1]]]).repeat(4,n_points * n_vertices_sphere,1)

    # texture_colors[masks,:] = texture_colors[masks,:] * 0 + torch.Tensor([[[0,1,0]]]).repeat(4,n_points * n_vertices_sphere,1)[~masks,:]

    texture_colors[masks,:] = torch.Tensor([[[0,1,0]]])

    return texture_colors





if __name__ == "__main__":

    print('THIS DOESNT WORK ON OCTOPUS / with oct_env')

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    global_config["general"]["target_folder"] = '/scratches/octopus_2/fml35/experiments/leveraging_geometry_for_shape/exp_117_scannet_models'

    fov = global_config["models"]["fov"]
    W = global_config["models"]["img_size"]
    H = global_config["models"]["img_size"]

    dir_path_3d_representation = '/scratches/octopus/fml35/datasets/shapenet_v2/ShapeNetRenamed/representation_points_and_normals/exp_03/'

    out_dir = '/scratches/octopus/fml35/datasets/shapenet_v2/ShapeNetRenamed/representation_points_and_normals/exp_03/mask_render_vis/'

    if torch.cuda.is_available():
        # device = torch.device("cuda:{}".format(global_config["general"]["gpu"]))
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    focal_length = (W /2) / np.tan(fov/2. * np.pi/180)
    # place camera at origin
    T = np.array([0,0,0])
    R = scipy_rot.from_euler('x', 0, degrees=True).as_matrix()
    cameras,rasterizer,renderer_textured,P_full,P_proj = create_setup(R,T,W,H,focal_length)

    # load rotations and translations

    R_and_T = np.load(global_config["general"]["target_folder"] + '/models/rotations/R_T_torch.npz')
    R_mesh = torch.from_numpy(R_and_T["R"]).to(device)
    T_mesh = torch.from_numpy(R_and_T["T"]).to(device)
    R_mesh = torch.inverse(R_mesh)

    elev = global_config["models"]["elev"]
    azim = global_config["models"]["azim"]

    # load model list
    with open(global_config["general"]["target_folder"] + "/models/model_list.json",'r') as f:
        model_list = json.load(f)["models"]

    # make_empty_folder_structure(global_config["general"]["target_folder"] + "/models/remeshed/",global_config["general"]["target_folder"] + "/models/depth/")

    with torch.no_grad():

        numbers = list(range(len(model_list)))
        # random.shuffle(numbers)

        # for k in tqdm(range(len(model_list))):
        for k in tqdm(numbers):

            model_name_combined = model_list[k]["category"] + '_' + model_list[k]["model"].split('/')[2]

            if not os.path.exists(dir_path_3d_representation+'/masks/' + model_name_combined + '.npz'):
                continue

            # if not model_name_combined == 'bathtub_4a6ed9c61b029d15904e03fa15bd73ec':
            #     continue

            points,masks = load_points_and_masks(dir_path_3d_representation,model_name_combined)
            # remesh model
            vertices,faces,textures = load_obj(global_config["general"]["target_folder"] + "/models/remeshed/" + model_list[k]["model"].replace('model/','') ,load_textures=False,device=device)
            # vertices,faces,textures = load_obj(global_config["general"]["target_folder"] + "/models/remeshed/" + model_list[k]["model"].replace('model_normalised/','') ,load_textures=False,device=device)
            # vertices,faces,textures = load_obj(global_config["dataset"]["dir_path"] + model_list[k]["model"],load_textures=False,device=device)
            if os.path.exists('{}{}_elev_045_azim_337.5.png'.format(out_dir,model_name_combined)):
                continue

            faces = faces[0]

            verst_sphere_single,faces_sphere_single,n_points,n_vertices_sphere = get_vertices_and_faces_for_points(points,device)

            # vertices,faces,texture_colors = combine_vertices_and_faces_with_points(vertices,faces,points,device)
            
            faces = faces.unsqueeze(0).repeat(4,1,1)

            T_mesh_this_object = T_mesh.unsqueeze(1).repeat(1,vertices.shape[0],1)

            counter = 0
            for i in range(16):
                # T_mesh.unsqueeze(1).repeat(1,vertices.shape[0],1)
                # T_mesh_this_object[4*i:4*(i+1),:,:]
                v_mesh = torch.transpose(torch.matmul(R_mesh[4*i:4*(i+1),:,:],torch.transpose(vertices,0,1)),1,2) + T_mesh.unsqueeze(1).repeat(1,vertices.shape[0],1)[4*i:4*(i+1),:,:]
                verts_sphere = torch.transpose(torch.matmul(R_mesh[4*i:4*(i+1),:,:],torch.transpose(verst_sphere_single,0,1)),1,2) + T_mesh.unsqueeze(1).repeat(1,verst_sphere_single.shape[0],1)[4*i:4*(i+1),:,:]

                
                texture_colors = get_colors(4 * i,elev,azim,masks,n_points,n_vertices_sphere)
                texture_colors = texture_colors.to(device)

                v_mesh = [v_mesh[i] for i in range(v_mesh.shape[0])]
                faces_mesh = [faces[i] for i in range(faces.shape[0])]
                textures_mesh = Textures(verts_rgb=(v_mesh[0] * 0 + 1).unsqueeze(0).repeat(4,1,1))


                textures_sphere = Textures(verts_rgb=texture_colors)


                meshes = Meshes(verts=v_mesh,faces=faces_mesh,textures=textures_mesh)
                meshes_sphere = Meshes(verts=verts_sphere,faces=faces_sphere_single.unsqueeze(0).repeat(4,1,1),textures=textures_sphere)

                output_mesh = renderer_textured(meshes,cameras=cameras[0],perspective_correct=True)
                output_spheres = renderer_textured(meshes_sphere,cameras=cameras[0],perspective_correct=True)

                output_mesh = output_mesh.detach().cpu().numpy()[:,:,:,:3]
                output_spheres = output_spheres.detach().cpu().numpy()[:,:,:,:3]
                # convert output to 0 to 255

                for j in range(4):
                    elev_index = counter // len(azim)
                    # NOTE: complicated formula because of old convention
                    # azim_index = (len(azim) - counter % len(azim)) % len(azim)
                    azim_index = counter % len(azim)
                    elev_current = str(int(elev[elev_index])).zfill(3)
                    azim_current = str(np.round(azim[azim_index],1)).zfill(3)

                    img = (output_mesh[j] + output_spheres[j]) / 2
                    img = np.round((img * 255)).astype(np.uint8)

                    # print('{}{}_elev_{}_azim_{}.png'.format(out_dir,model_name_combined,elev_current,azim_current))
                    cv2.imwrite('{}{}_elev_{}_azim_{}.png'.format(out_dir,model_name_combined,elev_current,azim_current),img)
                    counter += 1