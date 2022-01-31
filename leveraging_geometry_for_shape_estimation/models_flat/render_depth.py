

import torch
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
from PIL import Image
import trimesh

import numpy as np
print('import 1')
from tqdm import tqdm

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj
import pytorch3d
print('import 2')
# Data structures and functions for rendering
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (look_at_view_transform,FoVPerspectiveCameras,PerspectiveCameras,PointLights, DirectionalLights, Materials, 
    RasterizationSettings,MeshRenderer, MeshRasterizer,  SoftPhongShader,SoftSilhouetteShader,SoftPhongShader,TexturesVertex,Textures)
# add path for demo utils functions 
import sys
import os
import torch
import json

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
    renderer_textured = MeshRenderer(rasterizer=rasterizer,shader=SoftPhongShader(device=device, cameras=cameras[0]))
    P_full = cameras[0].get_full_projection_transform().get_matrix()
    P_proj = cameras[0].get_projection_transform().get_matrix()
    return cameras,rasterizer,renderer_textured,P_full,P_proj


if __name__ == "__main__":
    print('start')
    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    fov = global_config["models"]["fov"]
    W = global_config["models"]["img_size"]
    H = global_config["models"]["img_size"]

    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(global_config["general"]["gpu"]))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    focal_length = (W /2) / np.tan(fov/2. * np.pi/180)
    print('focal_length',focal_length)
    print(d)
    # place camera at origin
    T = np.array([0,0,0])
    R = scipy_rot.from_euler('x', 0, degrees=True).as_matrix()
    cameras,rasterizer,renderer_textured,P_full,P_proj = create_setup(R,T,W,H,focal_length)

    # load rotations and translations

    # R_and_T = np.load(global_config["general"]["target_folder"] + '/models/rotations/R_T_torch.npz')
    R_and_T = np.load('/scratch/fml35/experiments/leveraging_geometry_for_shape/test_output_all_s2/models/rotations/R_T_torch.npz')
    R_mesh = torch.from_numpy(R_and_T["R"]).to(device)
    T_mesh = torch.from_numpy(R_and_T["T"]).to(device)
    R_mesh = torch.inverse(R_mesh)

    print(R_mesh[0])
    print(T_mesh[0])

    elev = global_config["models"]["elev"]
    azim = global_config["models"]["azim"]

    # load model list
    with open(global_config["general"]["target_folder"] + "/models/model_list.json",'r') as f:
        model_list = json.load(f)

    make_empty_folder_structure(global_config["general"]["target_folder"] + "/models/remeshed/",global_config["general"]["target_folder"] + "/models/depth/")

    with torch.no_grad():
        for k in tqdm(range(len(model_list))):
            # remesh model
            vertices,faces,textures = load_obj(global_config["general"]["target_folder"] + "/models/remeshed/" + model_list[k]["name"] + '.obj' ,load_textures=False,device=device)
            # vertices,faces,textures = load_obj(global_config["dataset"]["pix3d_path"] + model_list[k]["model"],load_textures=False,device=device)
            if not os.path.exists(global_config["general"]["target_folder"] + "/models/depth/" + model_list[k]["name"]):
                os.mkdir(global_config["general"]["target_folder"] + "/models/depth/" + model_list[k]["name"])
            print(vertices.shape)

            
            T_mesh_this_object = T_mesh.unsqueeze(1).repeat(1,vertices.shape[0],1)

            faces = faces[0].unsqueeze(0).repeat(4,1,1)

            counter = 0
            for i in range(16):
                v_mesh = torch.transpose(torch.matmul(R_mesh[4*i:4*(i+1),:,:],torch.transpose(vertices,0,1)),1,2) + T_mesh_this_object[4*i:4*(i+1),:,:]
                
                # print(v_mesh.shape)
                # print(faces.shape)
                # print(dfd)
                print(v_mesh[:3,:])
                meshes = Meshes(verts=v_mesh,faces=faces)#,textures=textures)

                fragments = rasterizer(meshes,cameras=cameras[0],perspective_correct=True)

                depth = fragments.zbuf
                depth = depth.squeeze().cpu().numpy()

                for j in range(4):
                    elev_index = counter // len(azim)
                    # NOTE: complicated formula because of old convention
                    # azim_index = (len(azim) - counter % len(azim)) % len(azim)
                    azim_index = counter % len(azim)
                    elev_current = str(int(elev[elev_index])).zfill(3)
                    azim_current = str(np.round(azim[azim_index],1)).zfill(3)

                    out_path = global_config["general"]["target_folder"] + "/models/depth/" + model_list[k]["name"]
                    # print(depth[j][120:140,120:140])
                    np.save('{}/elev_{}_azim_{}.npy'.format(out_path,elev_current,azim_current),depth[j])
                    counter += 1