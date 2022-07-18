import cv2
import numpy as np
import sys
import os
import json
from torchvision import models
from tqdm import tqdm
from pytorch3d.renderer import look_at_view_transform
from math import ceil
import torch
from pytorch3d.io import load_obj
from scipy.spatial.transform import Rotation as scipy_rot

from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.pose import init_Rs,init_Ts,get_pb_real_grid,get_R_limits
from leveraging_geometry_for_shape_estimation.utilities.dicts import load_json
from leveraging_geometry_for_shape_estimation.pose_and_shape_optimisation.select_best_v2 import get_angle


from probabilistic_formulation.factors.factor_R import get_factor_reproject_lines_single_R,get_factor_reproject_lines_multiple_R_v2
from probabilistic_formulation.tests.test_reproject_lines import load_lines_2D,load_lines_3D,get_cuboid_line_dirs_3D,plot_vp_orig_size_v2




def find_R_closest_gt(R,R_gt):

    transform_Rs  = [scipy_rot.from_euler('zyx',[0,0,0], degrees=True).as_matrix(),
                        scipy_rot.from_euler('zyx',[0,90,0], degrees=True).as_matrix(),
                        scipy_rot.from_euler('zyx',[0,180,0], degrees=True).as_matrix(),
                        scipy_rot.from_euler('zyx',[0,270,0], degrees=True).as_matrix()]

    rotated_Rs = [np.matmul(R,transform_R) for transform_R in transform_Rs]
    angles = [get_angle(rotated_R,R_gt) for rotated_R in rotated_Rs]
    best_R = rotated_Rs[np.argmin(angles)]

    return best_R,rotated_Rs


def get_pose_for_folder():

    read_folder = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val'
    out_folder = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca'
    path_roca_detection_with_gt = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/all_detection_infos_fixed_category.json'
    path_output_file = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/all_detection_with_rotations_lines.json'
    dir_path_shapenet = '/scratch2/fml35/datasets/shapenet_v2/ShapeNetRenamed/'
    sw = 2
    device = torch.device("cuda:0")

    torch.cuda.set_device(device)

    # rot_limits = {"tilt":{"range": 20,"steps": 20},"elev":{"range": 45,"steps": 45},"azim":{"range": 45,"steps": 45}}
    rot_limits = {"tilt":{"range": 20,"steps": 20},"elev":{"range": 45,"steps": 45},"azim":{"range": 0,"steps": 1}}
    tilts,elevs,azims = get_R_limits(0,30,rot_limits)
    Rs,angles = init_Rs(tilts,elevs,azims)
    # np.save(out_folder + '/rotations_for_lines.npy',Rs)

    # Rs = np.load(out_folder + '/rotations_for_lines.npy')

    with open(path_roca_detection_with_gt,'r') as f:
        roca_detection_with_gt = json.load(f)

    # have two step procedure, first use all lines from image to find up axis (as have visualised currently),
    # then use only lines in bbox (load from filtered) to determine azimuthal

    for gt_name in tqdm(sorted(roca_detection_with_gt)):

        for i in range(len(roca_detection_with_gt[gt_name])):
    
            gt_infos = roca_detection_with_gt[gt_name][i]["associated_gt_infos"]
            # convert pixel to pixel bearing
            f = gt_infos["focal_length"]
            w = gt_infos["img_size"][0]
            h = gt_infos["img_size"][1]

            # get infos
            B = get_pb_real_grid(w,h,f,sw,device)

            # compute factors
            factors = []

            line_path = read_folder + '/lines_2d_cropped/' + gt_name.split('.')[0] + '.npy'
            # line_path = target_folder + '/lines_2d_filtered/' + name.split('.')[0] + '.npy'
            lines_2D = torch.Tensor(np.load(line_path)).long()
            print('before factor')
            # if no lines
            if len(lines_2D.shape) == 1:
                best_R_index = 0
                max_factor = 0
            else:
                mask = ~(lines_2D[:,:2] == lines_2D[:,2:4]).all(dim=1)
                lines_2D = lines_2D[mask]

                lines_3D = np.array([[0,0.,0,1.,0.0,0],[0,0.,0,0.,1.0,0],[0,0.,0,0.,0.0,1.]])

                line_dirs_3D = torch.Tensor([[1.00,  0.0000,  0.0000],[ 0.0000,  1.00,  0.0000],[ 0.,  0.0000,  1.00]])
                factors_batch,factor_all_2d_lines,all_masks = get_factor_reproject_lines_multiple_R_v2(torch.Tensor(Rs).to(device),line_dirs_3D.to(device),lines_2D.to(device),B.to(device))
                factors = factors_batch.cpu().tolist()
                best_R_index = factors.index(max(factors))
                best_angles = angles[best_R_index]

                tilts,elevs,azims = get_R_limits(0,30,rot_limits)
                new_R = 


            R = Rs[best_R_index]
            factors_2d_lines_best_R = factor_all_2d_lines[best_R_index].cpu()
            mask_best_R = all_masks[best_R_index].cpu()

            best_R,rotated_Rs = find_R_closest_gt(R,gt_infos['rot_mat'])

            roca_detection_with_gt[gt_name][i]['rotations_from_lines'] = {'best_R':best_R,'all_Rs':rotated_Rs}

            img_path = read_folder + '/images/' + gt_name.replace('.json','.jpg')
            model_path = dir_path_shapenet + gt_infos["model"]
            gt_T = gt_infos["trans_mat"]
            gt_scaling = gt_infos["scaling"]


            img = plot_vp_orig_size_v2(torch.Tensor(best_R).unsqueeze(0),torch.Tensor(gt_T).unsqueeze(0).unsqueeze(0),gt_scaling,model_path,img_path,sw,device,lines_3D,lines_2D,f,factors_2d_lines_best_R,mask_best_R)
            img = cv2.resize(img,(480,360))
            out_path_img = out_folder + '/rotation_vis/' + roca_detection_with_gt[gt_name][i]['detection'] + '.png'
            cv2.imwrite(out_path_img,img)

    with open(path_output_file,'w') as f:
        json.dump(roca_detection_with_gt)


def main():
    np.random.seed(1)
    torch.manual_seed(0)

    get_pose_for_folder()
    


if __name__ == '__main__':
    print('Rotation from lines')
    main()

