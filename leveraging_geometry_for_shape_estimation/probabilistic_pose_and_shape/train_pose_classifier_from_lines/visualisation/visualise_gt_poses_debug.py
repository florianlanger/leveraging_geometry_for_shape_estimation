import torch
from tqdm import tqdm
import os
import json
import cv2
import numpy as np
from numpy.lib.utils import info
import torch
import os
import sys
import json
# import quaternion
import shutil

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (look_at_view_transform,PerspectiveCameras,FoVPerspectiveCameras,RasterizationSettings, MeshRenderer, MeshRasterizer,SoftPhongShader,Textures)

from leveraging_geometry_for_shape_estimation.vis_pose.vis_pose import load_mesh,overlay_rendered_image,render_mesh


def convert_K(K,width,height):
    K[0,2] = -(K[0,2] - width/2)
    K[1,2] = -(K[1,2] - height/2)
    K = K/(width/2)
    K[2:4,2:4] = torch.Tensor([[0,1],[1,0]])
    return K

def render_mesh_calibration(w,h,K,mesh,device,flip=False):


    r_cam = torch.eye(3).unsqueeze(0).repeat(1,1,1)
    t_cam = torch.zeros((1,3))
    K = torch.Tensor(K).unsqueeze(0)
    # print('CHANGE BACK render_mesh in vis_pose')
    cameras_pix = PerspectiveCameras(device=device,K = K,T=t_cam,R=r_cam)

    raster_settings_soft = RasterizationSettings(image_size = max(w,h),blur_radius=0, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=cameras_pix,raster_settings=raster_settings_soft)
    renderer_textured = MeshRenderer(rasterizer=rasterizer,shader=SoftPhongShader(device=device))

    image = renderer_textured(mesh,cameras=cameras_pix).cpu().numpy()[0,:,:,:]

    # crop
    if w >= h:
        image = image[int((w-h)/2):int((w+h)/2),:,:]
    elif w < h:
        image = image[:,int((h-w)/2):int((h+w)/2),:]

    if flip:
        image = image[::-1,::-1,:]

    return image


def main():
    target_folder = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/train'
    shape_dir_3d = '/scratch2/fml35/datasets/shapenet_v2/ShapeNetRenamed/'
    device = torch.device("cuda:0")

    if not os.path.exists(target_folder + '/valid_objects_render_overlay_test_rokuban'):
        os.mkdir(target_folder + '/valid_objects_render_overlay_test_rokuban')
        # os.mkdir(target_folder + '/valid_objects_render')

    for file in tqdm(sorted(os.listdir(target_folder + '/gt_infos_valid_objects'))):

        if not 'scene0017_00-001000' in file:
            continue

        with open(target_folder + '/gt_infos_valid_objects/' + file,'r') as f:
            gt_infos = json.load(f)

        with open('/scratch2/fml35/datasets/scannet/scannet_frames_25k_own_data_exp_13_calibration_matrix/scene0017_00/info/001000/003.json','r') as f:
            info_dict = json.load(f)
        
        print(gt_infos)
        print(info_dict)

        width,height = gt_infos["img_size"]

        orig_img = cv2.imread(target_folder + '/images/' + file.replace('.json','.jpg'))
        
        for i in range(len(gt_infos["objects"])):
            obj = gt_infos["objects"][i]
            model_path = shape_dir_3d + obj["model"]
            mesh = load_mesh(model_path,obj["rot_mat"],obj["trans_mat"],obj["scaling"],device)
            # mesh = load_mesh(model_path,info_dict["rot_mat"],info_dict["trans_mat"],info_dict["scaling"],device)
            K = convert_K(torch.Tensor(gt_infos['K']),width,height)
            render = render_mesh_calibration(width,height,K,mesh,device)

            # render = render_mesh(width,height,f,mesh,device,sw)

            # render = cv2.resize(render,(480,360))
            overlay = overlay_rendered_image(orig_img,render)

            render = np.round((255*render)).astype(np.uint8)
            # cv2.imwrite(target_folder + '/valid_objects_render/' + file.replace('.json','_' + str(obj['index']).zfill(2) + '.png'),render)
            # cv2.imwrite(target_folder + '/valid_objects_render_overlay/' + file.replace('.json','_' + str(obj['index']).zfill(2) + '.png'),overlay)

            # cv2.imwrite(target_folder + '/valid_objects_render/' + file.replace('.json','_' + str(obj['index']).zfill(2) + '.png'),render)
            cv2.imwrite(target_folder + '/valid_objects_render_overlay_test_rokuban/' + file.replace('.json','_' + str(obj['index']).zfill(2) + '_new_trans.png'),overlay)


if __name__ == '__main__':
    main()