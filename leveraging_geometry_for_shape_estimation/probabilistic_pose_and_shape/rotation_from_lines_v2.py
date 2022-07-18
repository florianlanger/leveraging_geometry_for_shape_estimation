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

from leveraging_geometry_for_shape_estimation.keypoint_matching.get_matches_3d import load_information_depth_camera,create_pixel_bearing,pb_and_depth_to_wc
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.pose import init_Rs,init_Ts,get_pb_real_grid,get_R_limits,get_T_limits,get_nearest_pose_to_gt_all_R,check_gt_pose_in_limits,create_pose_info_dict
from leveraging_geometry_for_shape_estimation.utilities.dicts import load_json
from leveraging_geometry_for_shape_estimation.pose_and_shape_optimisation.select_best_v2 import get_angle


from probabilistic_formulation.utilities import create_all_possible_combinations,get_uvuv_p_from_superpoint,create_all_possible_combinations_uvuv_p_together
from probabilistic_formulation.factors.factor_R import get_factor_reproject_lines_single_R,get_factor_reproject_lines_multiple_R
from probabilistic_formulation.tests.test_reproject_lines import load_lines_2D,load_lines_3D,get_cuboid_line_dirs_3D,plot_vp_orig_size


def find_R_from_retrieval(R,nn_infos):

    transform_Rs  = [scipy_rot.from_euler('zyx',[0,0,0], degrees=True).as_matrix(),
                        scipy_rot.from_euler('zyx',[0,90,0], degrees=True).as_matrix(),
                        scipy_rot.from_euler('zyx',[0,180,0], degrees=True).as_matrix(),
                        scipy_rot.from_euler('zyx',[0,270,0], degrees=True).as_matrix()]

    elev = float(nn_infos["elev"])
    azim = float(nn_infos["azim"])

    retrieved_R = scipy_rot.from_euler('zyx',[0,180-azim,-elev], degrees=True).as_matrix()
    rotated_Rs = [np.matmul(R,transform_R) for transform_R in transform_Rs]
    angles = [get_angle(rotated_R,retrieved_R) for rotated_R in rotated_Rs]
    best_R = rotated_Rs[np.argmin(angles)]

    return best_R



def get_pose_for_folder(global_config):

    target_folder = global_config["general"]["target_folder"]
    models_folder_read = global_config["general"]["models_folder_read"]
    top_n_retrieval = global_config["keypoints"]["matching"]["top_n_retrieval"]

    with open(target_folder + '/global_stats/visualisation_images.json','r') as open_f:
        visualisation_list = json.load(open_f)

    # print('USE FILTERED LINES')


    pose_config = global_config["pose_and_shape"]["pose"]

    device = torch.device("cuda:{}".format(global_config["general"]["gpu"]))
    torch.cuda.set_device(device)

    # Rs = np.load(target_folder + '/models/rotations/rotations_for_lines.npy')
    tilts,elevs,azims = get_R_limits(0,30,global_config["pose_and_shape_probabilistic"]["pose"])
    Rs = init_Rs(tilts,elevs,azims)
    # np.save(target_folder + '/models/rotations/rotations_for_lines.npy',Rs)
    # Rs = np.load(target_folder + '/models/rotations/rotations_for_lines.npy')
    for name in tqdm(sorted(os.listdir(target_folder + '/cropped_and_masked'))):

        if not os.path.exists(target_folder + '/nn_infos/' + name.split('.')[0] + '.json'):
            continue

        with open(target_folder + '/nn_infos/' + name.split('.')[0] + '.json','r') as open_f:
            retrieval_list = json.load(open_f)["nearest_neighbours"]

        with open(target_folder + '/gt_infos/' + name.rsplit('_',1)[0] + '.json','r') as open_f:
            gt_infos = json.load(open_f)

        with open(target_folder + '/segmentation_infos/' + name.split('.')[0] + '.json','r') as open_f:
            segmentation_infos = json.load(open_f)

        with open(target_folder + '/bbox_overlap/' + name.split('.')[0] + '.json','r') as f:
            bbox_overlap = json.load(f)

        # convert pixel to pixel bearing
        f = gt_infos["focal_length"]
        w = gt_infos["img_size"][0]
        h = gt_infos["img_size"][1]
        sw = pose_config["sensor_width"]

        # get infos
        B = get_pb_real_grid(w,h,f,sw,device)

        gt_T = gt_infos["objects"][bbox_overlap['index_gt_objects']]["trans_mat"]

        scaling = gt_infos["objects"][bbox_overlap['index_gt_objects']]["scaling"]
        xs,ys,zs =  (gt_T[0],gt_T[0],1),(gt_T[1],gt_T[1],1),(gt_T[2],gt_T[2],1)
        Ts = init_Ts(xs,ys,zs)


        best_T_possible,best_R_possible,min_angle = None,None,None
        gt_pose_in_limits = None

        # compute factors
        factors = []
        Ts = torch.Tensor(Ts)

        line_path = target_folder + '/lines_2d_cropped/' + gt_infos["img"].split('.')[0] + '.npy'
        # line_path = target_folder + '/lines_2d_filtered/' + name.split('.')[0] + '.npy'
        lines_2D = torch.Tensor(np.load(line_path)).long()
        # if no lines
        if len(lines_2D.shape) == 1:
            best_R_index = 0
            max_factor = 0
        else:
            mask = ~(lines_2D[:,:2] == lines_2D[:,2:4]).all(dim=1)
            lines_2D = lines_2D[mask]

            lines_3D = np.array([[0,0.,0,1.,0.0,0],[0,0.,0,0.,1.0,0],[0,0.,0,0.,0.0,1.]])

            line_dirs_3D = torch.Tensor([[1.00,  0.0000,  0.0000],[ 0.0000,  1.00,  0.0000],[ 0.,  0.0000,  1.00]])
            factors_batch = get_factor_reproject_lines_multiple_R(torch.Tensor(Rs).to(device),line_dirs_3D.to(device),lines_2D.to(device),B.to(device))
            factors = factors_batch.cpu().tolist()

            # print(factors)
            best_R_index = factors.index(max(factors))
            max_factor = max(factors)

        R = Rs[best_R_index]

        n_indices = 0

        nn_infos = load_json(target_folder + '/nn_infos/' + name.split('.')[0] + '.json')

        n_retrieval = min([len(retrieval_list),top_n_retrieval])




        for i in range(n_retrieval):
            
            output_path = target_folder + '/poses_R/' + name.split('.')[0] + '_' + str(i).zfill(3) + '_' + str(0).zfill(2) + '.json'


            R_closest_retrieval = find_R_from_retrieval(R,nn_infos["nearest_neighbours"][i])

            gt_rot = gt_infos["objects"][bbox_overlap['index_gt_objects']]["rot_mat"]
            gt_trans = gt_infos["objects"][bbox_overlap['index_gt_objects']]["trans_mat"]

            pose_information = create_pose_info_dict(R_closest_retrieval,Ts[0],n_indices,max_factor,gt_pose_in_limits,gt_rot,gt_trans,best_R_possible,best_T_possible,xs,ys,zs,tilts,elevs,azims,np.array(scaling),np.array(scaling),None,None,[1.,1.,1.])

    
            with open(output_path,'w') as open_f:
                json.dump(pose_information,open_f,indent=4)

            # save_path = output_path.replace('/poses_R/','/factors/').replace('.json','.npz')
            # np.savez(save_path,factors=factors,Rs=Rs,Ts=Ts)
            if gt_infos["img"] in visualisation_list:
                img_save_path = target_folder + '/factors_lines_vis/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.png'
                img_path = target_folder + '/images/' + gt_infos["img"]
                model_path = global_config["dataset"]["dir_path"] + retrieval_list[i]["model"]
                # model_path = global_config["dataset"]["dir_path"] + gt_infos["objects"][bbox_overlap['index_gt_objects']]["model"]

                img = plot_vp_orig_size(torch.Tensor(R_closest_retrieval).unsqueeze(0),torch.Tensor(gt_T).unsqueeze(0).unsqueeze(0),scaling,model_path,img_path,sw,device,lines_3D,lines_2D,B,f,gt_infos)
                # img = plot_vp_orig_size(torch.Tensor(gt_rot).unsqueeze(0),torch.Tensor(gt_T).unsqueeze(0).unsqueeze(0),scaling,model_path,img_path,sw,device,lines_3D,lines_2D,B,f,gt_infos)
                cv2.imwrite(output_path.replace('/poses_R/','/factors_lines_vis/').replace('.json','.png'),img)



def main():
    np.random.seed(1)
    torch.manual_seed(0)

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    if global_config["pose_and_shape_probabilistic"]["use_probabilistic"] == "True" and global_config["pose_and_shape_probabilistic"]["pose"]["gt_R"] == False:
        get_pose_for_folder(global_config)
    


if __name__ == '__main__':
    print('Rotation from lines')
    main()

