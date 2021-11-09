import numpy as np
import cv2
import torch
from pytorch3d.io import load_objs_as_meshes, save_obj, load_obj
import pytorch3d
import trimesh

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (look_at_view_transform,FoVPerspectiveCameras,RasterizationSettings, MeshRenderer, MeshRasterizer,SoftPhongShader,Textures)
import sys
import os
import json 
from pytorch3d.io import save_ply
from tqdm import tqdm


def create_pixel_bearing(W,H,P_proj,device):

    assert W == H , "This function does not work if W and H diff, think need to change W and H in line above .view(H,W,4) but not sure if this is all"
    x = -(2 * (torch.linspace(0,W-1,W,device=device)+0.5)/W - 1)
    y = - (2 * (torch.linspace(0,H-1,H,device=device)+0.5)/H - 1)

    ys,xs = torch.meshgrid(x,y)

    xyz_hom = torch.stack([xs,ys,xs*0+1,xs*0+1],axis=-1)

    P_proj_inv = torch.inverse(P_proj[0,:,:])

    # TODO: whcih order W,H
    xyz_proj = torch.matmul(P_proj_inv,xyz_hom.view(W*H,4).T).T.view(W,H,4)


    pb_x = xyz_proj[:,:,0]
    pb_y = xyz_proj[:,:,1]
    pb_z = xyz_proj[:,:,2]
    return pb_x,pb_y,pb_z

def wc_gt_object(data,device,path_to_remeshed):
    w,h = data["img_size"]

    f = data["focal_length"]
    if w >= h:
        fov = 2 * np.arctan((16.)/f)
    elif w < h:
        fov = 2 * np.arctan((16. * h/w)/f)

    r_cam = torch.eye(3).unsqueeze(0).repeat(1,1,1)
    t_cam = torch.zeros((1,3))
    cameras_pix = FoVPerspectiveCameras(device=device,fov = fov,degrees=False,R = r_cam, T = t_cam)
    raster_settings_soft = RasterizationSettings(image_size = max(w,h),blur_radius=0, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=cameras_pix,raster_settings=raster_settings_soft)
    renderer_textured = MeshRenderer(rasterizer=rasterizer,shader=SoftPhongShader(device=device))



    # load gt mesh
    # vertices,faces,_ = load_obj(obj_full_path, device=device,create_texture_atlas=False, load_textures=False)
    gt_obj = load_obj(path_to_remeshed + data["model"].replace('model/',''), device=device,create_texture_atlas=False, load_textures=False)
    gt_vertices_origin,gt_faces,gt_properties = gt_obj
    R_gt = torch.Tensor(data["rot_mat"]).to(device) #.inverse().to(device)
    T_gt = torch.Tensor(data["trans_mat"]).to(device)
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


    P_proj = torch.Tensor([[[1/np.tan(fov/2.), 0.0000, 0.0000, 0.0000],
                        [0.0000, 1/np.tan(fov/2.), 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.0000, 1.0000],
                        [0.0000, 0.0000, 1.0000, 0.0000]]]).to(device)


    pb_x, pb_y, pb_z = create_pixel_bearing(max(w,h),max(w,h),P_proj,device)


    cc_x = pb_x * depth
    cc_y = pb_y * depth
    cc_z = pb_z * depth

    cc = torch.stack([cc_x,cc_y,cc_z],dim=-1)
    wc = cc.cpu().numpy()

    # crop
    if w >= h:
        wc = wc[int((w-h)/2):int((w+h)/2),:,:]
    elif w < h:
        wc = wc[:,int((h-w)/2):int((h+w)/2),:]
    return wc


if __name__ == '__main__':


    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    path_to_remeshed = global_config["general"]["target_folder"] + '/models/remeshed/'
    
    device = torch.device("cuda:{}".format(global_config["general"]["gpu"]))

    for img in os.listdir(global_config["general"]["target_folder"] + '/gt_infos/'):
        with open(global_config["general"]["target_folder"] + '/gt_infos/' + img) as f:
            gt_info = json.load(f)

        wc = wc_gt_object(gt_info,device,path_to_remeshed)
        save_path = global_config["general"]["target_folder"] + '/wc_gt/' + img.replace('.json','.npy')
        np.save(save_path,wc)
