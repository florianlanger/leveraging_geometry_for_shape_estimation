import cv2
from cv2 import threshold
import numpy as np
import sys
import os
import json
from torchvision import models
from tqdm import tqdm
from pytorch3d.renderer import look_at_view_transform
from math import ceil
import torch
from pytorch3d.io import load_obj, load_ply
from scipy.spatial.transform import Rotation as scipy_rot

from leveraging_geometry_for_shape_estimation.keypoint_matching.get_matches_3d import load_information_depth_camera,create_pixel_bearing,pb_and_depth_to_wc
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.pose import init_Rs,init_Ts,get_pb_real_grid,get_R_limits,get_T_limits, create_pose_info_dict, check_gt_pose_in_limits, get_nearest_pose_to_gt
from probabilistic_formulation.utilities import create_all_possible_combinations,get_uvuv_p_from_superpoint,create_all_possible_combinations_uvuv_p_together
from probabilistic_formulation.factors import get_factor_bbox_multiple_T, get_factor_reproject_lines_multiple_T,get_factor_reproject_kp_multiple_T
from probabilistic_formulation.tests.test_reproject_lines import load_lines_2D,load_lines_3D,get_cuboid_line_dirs_3D,plot_lines_T,plot_bbox, plot_points_T


def get_pose_for_folder(global_config):

    target_folder = global_config["general"]["target_folder"]
    models_folder_read = global_config["general"]["models_folder_read"]
    top_n_retrieval = global_config["keypoints"]["matching"]["top_n_retrieval"]
    top_n_for_translation = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["top_n_for_translation"]

    with open(target_folder + '/global_stats/visualisation_images.json','r') as open_f:
        visualisation_list = json.load(open_f)


    pose_config = global_config["pose_and_shape"]["pose"]
    sensor_width = pose_config["sensor_width"]

    device = torch.device("cuda:{}".format(global_config["general"]["gpu"]))
    torch.cuda.set_device(device)

    # for name in tqdm(os.listdir(target_folder + '/cropped_and_masked')):

    for name in tqdm(os.listdir(target_folder + '/poses_R')):
        # for rand_ind in range(0,100):
            # if not "bed_0861_0" in name:
            #     continue
            retrieval = name.rsplit('_',1)[0]
            detection = name.rsplit('_',2)[0]
            gt_name = name.rsplit('_',3)[0]


            with open(target_folder + '/nn_infos/' + detection + '.json','r') as open_f:
                retrieval_list = json.load(open_f)["nearest_neighbours"]

            with open(target_folder + '/gt_infos/' + gt_name + '.json','r') as open_f:
                gt_infos = json.load(open_f)

            # if gt_infos["img"] in visualisation_list:
            #     continue


            with open(target_folder + '/segmentation_infos/' + detection + '.json','r') as open_f:
                segmentation_infos = json.load(open_f)

            nn_index = int(retrieval.split('_')[-1])

            if nn_index >= len(retrieval_list):
                continue
        
            output_path = target_folder + '/poses/' + name

            # convert pixel to pixel bearing
            f = gt_infos["focal_length"]
            w = gt_infos["img_size"][0]
            h = gt_infos["img_size"][1]
            sw = pose_config["sensor_width"]
            enforce_same_length = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["enforce_same_length"]

            # get infos
            B = get_pb_real_grid(w,h,f,sw,device)

            # print('Use gt R for debugging')
            # R  = gt_infos["rot_mat"]
            with open(target_folder + '/poses_R/' + name,'r') as file:
                R = json.load(file)["predicted_r"]

            sp_rot = scipy_rot.from_matrix(R)
            tilt,azim,elev = sp_rot.as_euler('zyx',degrees=True)

            tilts,azims,elevs = [tilt-0.0001,tilt+0.0001,1],[azim-0.0001,azim+0.0001,1],[elev-0.0001,elev+0.0001,1]

            bbox = segmentation_infos["predictions"]["bbox"]
            gt_z = gt_infos["trans_mat"][2]
            xs,ys,zs = get_T_limits(f,[w,h],sensor_width,global_config["pose_and_shape_probabilistic"]["pose"],bbox,gt_z)
            Ts = init_Ts(xs,ys,zs)

            gt_pose_in_limits = check_gt_pose_in_limits(xs,ys,zs,tilts,elevs,azims,gt_infos["trans_mat"],gt_infos["rot_mat"])
            best_T_possible,best_R_possible = get_nearest_pose_to_gt(xs,ys,zs,tilts,elevs,azims,gt_infos["trans_mat"],gt_infos["rot_mat"])

            signal = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["signal"]

            # compute factors
            factors = []
            Ts = torch.Tensor(Ts)

            if signal == 'bbox' or signal == 'lines':

                path =  global_config["pose_and_shape_probabilistic"]["reproject_lines"]["line_dir_3d_precise"] + "/" + retrieval_list[nn_index]["model"].split('/')[1] + '_' + retrieval_list[nn_index]["model"].split('/')[2] + '.npy'
                if not os.path.exists(path):
                    path = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["line_dir_3d_self_scanned"] + '/' + retrieval_list[nn_index]["model"].split('/')[1] + '_' + retrieval_list[nn_index]["model"].split('/')[2] + '.npy'
                lines_3D = load_lines_3D(path)
                mask_length = np.sum((lines_3D[:,:3] - lines_3D[:,3:6])**2,axis=1)**0.5 > global_config["pose_and_shape_probabilistic"]["reproject_lines"]["min_length_line"]
                lines_3D = lines_3D[mask_length]
                
                line_dirs_3D = torch.from_numpy(lines_3D[:,:3] - lines_3D[:,3:6])
                line_dirs_3D = line_dirs_3D / torch.linalg.norm(line_dirs_3D,axis=1).unsqueeze(1).tile(1,3)
                # because lines saved as points but need as point + direction 
                lines_3D[:,3:6] = lines_3D[:,3:6] - lines_3D[:,:3]


                line_path = target_folder + '/lines_2d_filtered/' + detection + '.npy'
                lines_2D = torch.Tensor(np.load(line_path)).long()

                if len(lines_2D.shape) == 1 or lines_3D.shape[0] == 0:
                    best_T_index = 0
                    max_factor = 0
                    lines_available = False
                else:
                    lines_available = True
                    mask = ~(lines_2D[:,:2] == lines_2D[:,2:4]).all(dim=1)
                    lines_2D = lines_2D[mask]
                
                    if global_config["pose_and_shape_probabilistic"]["reproject_lines"]["signal"] == 'bbox':
                        factors_batch,_ = get_factor_bbox_multiple_T(torch.Tensor(R).to(device),torch.Tensor(Ts).to(device),torch.from_numpy(lines_3D).to(device),torch.Tensor(bbox).to(device),f,w,h,sw)
                    elif global_config["pose_and_shape_probabilistic"]["reproject_lines"]["signal"] == 'lines':
                        multiplier_lines = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["multiplier_lines"]
                        factors_batch,_ = get_factor_reproject_lines_multiple_T(torch.Tensor(R).to(device),torch.Tensor(Ts).to(device),torch.from_numpy(lines_3D).to(device),lines_2D.to(device),B.to(device),top_n_for_translation,f,multiplier_lines,enforce_same_length)
                    factors = factors_batch.cpu().tolist()
                    best_T_index = factors.index(max(factors))
                    max_factor = max(factors)

            elif signal == 'points':
                points_3D,_ = load_ply(global_config["pose_and_shape_probabilistic"]["reproject_lines"]["point_dir"] + '/' + retrieval_list[nn_index]["model"].split('/')[1] + '/' + retrieval_list[nn_index]["model"].split('/')[2] + '.ply')
                # with open(target_folder + '/kp_orig_img_size/' + detection + '.json','r') as file:
                #     points_2D = json.load(file)["pixels_real_orig_size"]
                points_2D_flipped = np.load(target_folder + '/keypoints_filtered/' + detection + '.npy')
                points_2D = points_2D_flipped * 0
                points_2D[:,0] = points_2D_flipped[:,1]
                points_2D[:,1] = points_2D_flipped[:,0]
                points_2D = torch.Tensor(points_2D).long()
                # points_2D = torch.flip(points_2D,dims=[0,1])
                # print(points_2D[:3])
                points_available = True
                multiplier_points = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["multiplier_points"]
                point_threshold = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["point_threshold"]
                use_threshold = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["use_point_threshold"]
                factors_batch,_ = get_factor_reproject_kp_multiple_T(torch.Tensor(R).to(device),torch.Tensor(Ts).to(device),points_3D.to(device),points_2D.to(device),B,top_n_for_translation,f,multiplier_points,point_threshold,use_threshold)
                factors = factors_batch.cpu().tolist()
                best_T_index = factors.index(max(factors))
                max_factor = max(factors)

            T = Ts[best_T_index]
            n_indices = 1
            pose_information = create_pose_info_dict(np.array(R),T,n_indices,max_factor,gt_pose_in_limits,gt_infos["rot_mat"],gt_infos["trans_mat"],best_R_possible,best_T_possible,xs,ys,zs,tilts,elevs,azims)

    
            with open(output_path,'w') as open_f:
                json.dump(pose_information,open_f,indent=4)

            save_path = output_path.replace('/poses/','/factors_T/').replace('.json','.npz')
            np.savez(save_path,factors=factors,Rs=R,Ts=Ts)

            if gt_infos["img"] in visualisation_list:
                img_path = target_folder + '/images/' + gt_infos["img"]
                model_path = global_config["dataset"]["pix3d_path"] + retrieval_list[nn_index]["model"]

                if signal == 'bbox':
                    if not lines_3D.shape[0] == 0:
                        img = plot_bbox(torch.Tensor(R).unsqueeze(0),torch.Tensor(T).unsqueeze(0).unsqueeze(0),model_path,img_path,sw,device,lines_3D,f,torch.Tensor(bbox))
                elif signal == 'lines':
                    if lines_available:
                        img = plot_lines_T(torch.Tensor(R).unsqueeze(0),torch.Tensor(T).unsqueeze(0).unsqueeze(0),model_path,img_path,sw,device,lines_3D,lines_2D,B,f,top_n_for_translation,multiplier_lines,enforce_same_length)
                    else:
                        img = cv2.imread(target_folder + '/images/' + gt_infos["img"])
                elif signal == 'points':
                    if points_available:
                        img = plot_points_T(torch.Tensor(R).unsqueeze(0),torch.Tensor(T).unsqueeze(0).unsqueeze(0),model_path,img_path,sw,device,points_3D,points_2D,B,f,top_n_for_translation,multiplier_points,point_threshold,use_threshold)
                    else:
                        img = cv2.imread(target_folder + '/images/' + gt_infos["img"])
                cv2.imwrite(output_path.replace('/poses/','/T_lines_vis/').replace('.json','.png'),img)



def main():
    np.random.seed(1)
    torch.manual_seed(0)

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    if global_config["pose_and_shape_probabilistic"]["use_probabilistic"] == "True":
        get_pose_for_folder(global_config)
    


if __name__ == '__main__':
    print('Translation from lines')
    main()

