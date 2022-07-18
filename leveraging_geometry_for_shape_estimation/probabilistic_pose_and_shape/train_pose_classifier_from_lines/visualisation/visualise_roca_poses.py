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
import quaternion
import shutil
from scipy.spatial.transform import Rotation as scipy_rot

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (look_at_view_transform,PerspectiveCameras,FoVPerspectiveCameras,RasterizationSettings, MeshRenderer, MeshRasterizer,SoftPhongShader,Textures)

from leveraging_geometry_for_shape_estimation.vis_pose.vis_pose import load_mesh,overlay_rendered_image,render_mesh
from leveraging_geometry_for_shape_estimation.data_conversion_scannet.reproject_scannet_v4 import convert_K


def flip_offset_principal_point_K(K,w,h):
    K[0,2] = w - K[0,2]
    K[1,2] = h - K[1,2]
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

def get_gt_infos(gt_infos,index):
    model_infos = gt_infos["objects"][index]
    R = model_infos['rot_mat']
    T = model_infos['trans_mat']
    S = model_infos['scaling']
    category = model_infos['category']
    catid = model_infos['catid']

    return R,T,S,category,catid


def eval_pred(R,T,S,category,catid,R_gt,T_gt,S_gt,category_gt,catid_gt):

    assert category == category_gt

    threshold_t = 0.2
    threshold_R = 20
    threshold_S = 20

    error_r = calc_rotation_diff_considering_symmetry(q,q_gt,sym_pred,sym_gt)
    error_t = np.linalg.norm(T - T_gt, ord=2)
    error_s = 100.0*np.mean(np.abs(S/S_gt) - 1)

    flag_r = error_r < threshold_R
    flag_t = error_t < threshold_t
    flag_s = error_s < threshold_S
    flag_retrieval = catid == catid_gt

    flag_overall_with_retrieval = flag_r and flag_t and flag_s and flag_retrieval
    flag_overall_without_retrieval = flag_r and flag_t and flag_s

def main():
    target_folder = '/scratches/octopus/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val'
    out_dir = target_folder + '/roca_render/'
    shape_dir_3d = '/scratches/octopus_2/fml35/datasets/shapenet_v2/ShapeNetRenamed/'
    device = torch.device("cuda:0")

    if not os.path.exists(target_folder + '/roca_render_overlay'):
        os.mkdir(target_folder + '/roca_render_overlay')

    for file in tqdm(sorted(os.listdir(target_folder + '/gt_infos'))):


        with open(target_folder + '/gt_infos/' + file,'r') as f:
            gt_infos = json.load(f)

        with open(target_folder + '/bboxes_roca_with_object_infos/' + file,'r') as f:
            bbox_with_roca_objects = json.load(f)


        width,height = gt_infos["img_size"]

        orig_img = cv2.imread(target_folder + '/images_480_360/' + file.replace('.json','.jpg'))

        K = torch.Tensor(gt_infos['K'])
        K = flip_offset_principal_point_K(K,width,height)
        K = convert_K(K,width,height)
        
        for i in range(len(bbox_with_roca_objects["roca_objects"])):
            roca_object = bbox_with_roca_objects["roca_objects"][i]
            index_out = bbox_with_roca_objects["indices_orig_objects"][i]
            if roca_object != None:
                category = roca_object['category'].replace('bookcase','bookshelf')
                catid = roca_object["scene_cad_id"][1]
                model_path = shape_dir_3d + "model/{}/{}/model_normalized.obj".format(category,catid)
                assert os.path.exists(model_path), "model path {} does not exist".format(model_path)
                T = np.array(roca_object['t'])
                q = roca_object['q']
                # q = np.quaternion(q[3], q[0], q[1], q[2])

                q = [q[1],q[2],q[3],q[0]]
                R = scipy_rot.from_quat(q).as_matrix()

                invert = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
                T = np.matmul(invert,T)
                R = np.matmul(invert,R)

                # R = quaternion.as_rotation_matrix(q)
                S = roca_object['s']

                R_gt,T_gt,S_gt,category_gt,catid_gt = get_gt_infos(gt_infos,index_out)

                eval_pred(R,T,S,category,catid,R_gt,T_gt,S_gt,category_gt,catid_gt)

                mesh = load_mesh(model_path,R,T,S,device)

                
                render = render_mesh_calibration(width,height,K,mesh,device)

                # render = render_mesh(width,height,f,mesh,device,sw)

                render = cv2.resize(render,(480,360))
                overlay = overlay_rendered_image(orig_img,render)

                render = np.round((255*render)).astype(np.uint8)
                # cv2.imwrite(target_folder + '/valid_objects_render/' + file.replace('.json','_' + str(obj['index']).zfill(2) + '.png'),render)
                # cv2.imwrite(target_folder + '/valid_objects_render_overlay/' + file.replace('.json','_' + str(obj['index']).zfill(2) + '.png'),overlay)

                cv2.imwrite(out_dir + file.replace('.json','_' + str(index_out).zfill(2) + '.png'),overlay)


if __name__ == '__main__':
    main()