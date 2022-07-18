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

    rotated_Rs = [r_rot.tolist() for r_rot in rotated_Rs]

    return best_R,rotated_Rs


def visualise(read_folder,gt_name,gt_infos,dir_path_shapenet,best_R,sw,device,lines_3D,lines_2D,f,factors_2d_lines_best_R,out_path_img):
    img_path = read_folder + '/images/' + gt_name.replace('.json','.jpg')
    model_path = dir_path_shapenet + gt_infos["model"]
    gt_T = gt_infos["trans_mat"]
    gt_scaling = gt_infos["scaling"]


    img = plot_vp_orig_size_v2(torch.Tensor(best_R).unsqueeze(0),torch.Tensor(gt_T).unsqueeze(0).unsqueeze(0),gt_scaling,model_path,img_path,sw,device,lines_3D,lines_2D,f,factors_2d_lines_best_R)
    img = cv2.resize(img,(480,360))
    cv2.imwrite(out_path_img,img)


def main(name):

    read_folder = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val'
    folder_path = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/'
    path_rotations_file = folder_path + 'rotation_results/{}.json'.format(name)
    out_dir = folder_path + 'rotation_vis/{}/'.format(name)
    os.mkdir(out_dir)
    dir_path_shapenet = '/scratch2/fml35/datasets/shapenet_v2/ShapeNetRenamed/'
    sw = 2
    device = torch.device("cuda:0")

    torch.cuda.set_device(device)


    lines_3D = np.array([[0,0.,0,1.,0.0,0],[0,0.,0,0.,1.0,0],[0,0.,0,0.,0.0,1.]])

    with open(path_rotations_file,'r') as f:
        roca_detection_with_gt = json.load(f)

    # have two step procedure, first use all lines from image to find up axis (as have visualised currently),
    # then use only lines in bbox (load from filtered) to determine azimuthal

    for gt_name in tqdm(sorted(roca_detection_with_gt)):

        for i in range(len(roca_detection_with_gt[gt_name])):
    
            gt_infos = roca_detection_with_gt[gt_name][i]["associated_gt_infos"]
            f = gt_infos["focal_length"]

            out_path_img_1 = out_dir + roca_detection_with_gt[gt_name][i]['detection'] + '_tilt_and_elev.png'
            out_path_img_3 = out_dir + roca_detection_with_gt[gt_name][i]['detection'] + '_full_roca.png'

            if roca_detection_with_gt[gt_name][i]['rotations_from_lines']['rotation_found'] == True:
                min_angle = str(int(np.round(roca_detection_with_gt[gt_name][i]['rotations_from_lines']["stage_2"]["min_angle_gt"]))).zfill(3)
                out_path_img_2 = out_dir + roca_detection_with_gt[gt_name][i]['detection'] + '_full_min_angle_gt_{}.png'.format(min_angle)

                infos_R = roca_detection_with_gt[gt_name][i]['rotations_from_lines']["stage_1"]
                visualise(read_folder,gt_name,gt_infos,dir_path_shapenet,infos_R['best_R'],sw,device,lines_3D,np.array(infos_R['lines_2D']),f,np.array(infos_R['factors']),out_path_img_1)

                infos_R = roca_detection_with_gt[gt_name][i]['rotations_from_lines']["stage_2"]
                visualise(read_folder,gt_name,gt_infos,dir_path_shapenet,infos_R["selected_R_gt"],sw,device,lines_3D,np.array(infos_R['lines_2D']),f,np.array(infos_R['factors']),out_path_img_2)
                visualise(read_folder,gt_name,gt_infos,dir_path_shapenet,infos_R["selected_R_roca"],sw,device,lines_3D,np.array(infos_R['lines_2D']),f,np.array(infos_R['factors']),out_path_img_3)
            
            else:
                out_path_img_2 = out_dir + roca_detection_with_gt[gt_name][i]['detection'] + '_full_min_angle_gt_{}.png'.format(200)



                img_path = read_folder + '/images/' + gt_name.replace('.json','.jpg')
                if not os.path.exists(img_path):
                    continue
                img = cv2.imread(img_path)
                img = cv2.resize(img,(480,360))
                cv2.imwrite(out_path_img_1,img)
                cv2.imwrite(out_path_img_2,img)


    


if __name__ == '__main__':
    print('Rotation from lines vis')
    name = sys.argv[1]
    main(name)

