import cv2
from cv2 import threshold
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.translation_from_lines_v2 import get_infos_gt
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
from probabilistic_formulation.tests.test_reproject_lines import load_lines_2D,load_lines_3D,get_cuboid_line_dirs_3D,plot_lines_T,plot_bbox, plot_points_T,plot_lines_T_only_render


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

    # name_out_dir = 'T_lines_vis_new_model_all_gt'
    # name_out_dir = 'T_lines_vis_new_model_filtered_gt'
    # name_out_dir = 'T_lines_vis_old_model_gt'
    name_out_dir = 'T_lines_vis_gt_render_1'

    shape_dir_3d = global_config["general"]["models_folder_read"] + '/models/extract_from_2d/exp_03/lines_3d_combined_reformatted'
    # shape_dir_3d = global_config["general"]["models_folder_read"] + '/models/extract_from_2d/exp_03/lines_3d_reformatted'
    # shape_dir_3d = global_config["general"]["models_folder_read"] + '/models/lines'

    # save number lines
    # out_file_line_number = target_folder + '/global_stats/count_3d_lines/{}.json'.format(name_out_dir)
    # assert os.path.exists(out_file_line_number) == False
    # number_lines_3d = {}

    # for name in tqdm(os.listdir(target_folder + '/cropped_and_masked')):
    for name in tqdm(sorted(os.listdir(target_folder + '/poses_R'))):
        # for rand_ind in range(0,100):
            if not "00.json" in name:
                continue
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

            with open(target_folder + '/bbox_overlap/' + detection + '.json','r') as f:
                bbox_overlap = json.load(f)

            nn_index = int(retrieval.split('_')[-1])

            if nn_index >= len(retrieval_list):
                continue
        
            output_path = target_folder + '/poses/' + name

            # if os.path.exists(output_path):
            #     continue

            # convert pixel to pixel bearing
            f = gt_infos["focal_length"]
            w = gt_infos["img_size"][0]
            h = gt_infos["img_size"][1]
            sw = pose_config["sensor_width"]
            enforce_same_length = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["enforce_same_length"]

            # get infos
            B = get_pb_real_grid(w,h,f,sw,device)

            # print('Use gt R for debugging')
            R  = gt_infos["objects"][bbox_overlap['index_gt_objects']]["rot_mat"]
            # with open(target_folder + '/poses_R/' + name,'r') as file:
            #     R = json.load(file)["predicted_r"]

            sp_rot = scipy_rot.from_matrix(R)
            tilt,azim,elev = sp_rot.as_euler('zyx',degrees=True)

            tilts,azims,elevs = [tilt-0.0001,tilt+0.0001,1],[azim-0.0001,azim+0.0001,1],[elev-0.0001,elev+0.0001,1]

            bbox = segmentation_infos["predictions"]["bbox"]

            gt_rot = gt_infos["objects"][bbox_overlap['index_gt_objects']]["rot_mat"]
            gt_trans = gt_infos["objects"][bbox_overlap['index_gt_objects']]["trans_mat"]
            scaling = gt_infos["objects"][bbox_overlap['index_gt_objects']]["scaling"]

            gt_z = gt_trans[2]
            xs,ys,zs = get_T_limits(f,[w,h],sensor_width,global_config["pose_and_shape_probabilistic"]["pose"],bbox,gt_z)
            Ts = init_Ts(xs,ys,zs)

            gt_pose_in_limits = check_gt_pose_in_limits(xs,ys,zs,tilts,elevs,azims,gt_trans,gt_rot)
            best_T_possible,best_R_possible = get_nearest_pose_to_gt(xs,ys,zs,tilts,elevs,azims,gt_trans,gt_rot)

            signal = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["signal"]

            # compute factors
            factors = []
            T = torch.Tensor(gt_infos["objects"][bbox_overlap['index_gt_objects']]["trans_mat"])

            # print('USE GT RETRIEVAL OTHERWISE PROBLEM WITH SCALE')
            retrieval_list[nn_index]["model"] = gt_infos["objects"][bbox_overlap['index_gt_objects']]["model"]

            if signal == 'bbox' or signal == 'lines':

                path =  shape_dir_3d + "/" + retrieval_list[nn_index]["model"].split('/')[1] + '_' + retrieval_list[nn_index]["model"].split('/')[2] + '.npy'
                if not os.path.exists(path):
                    path = shape_dir_3d + '/' + retrieval_list[nn_index]["model"].split('/')[1] + '_' + retrieval_list[nn_index]["model"].split('/')[2] + '.npy'

                if os.path.exists(path):
                    lines_3D = load_lines_3D(path).astype(np.float32)
                else:
                    lines_3D = np.array([[0.,0.,0,0.5,0.,0]]).astype(np.float32)

                if lines_3D.shape[0] == 0:
                    lines_3D = np.array([[0.,0.,0,0.5,0.,0]]).astype(np.float32)


                lines_3D = lines_3D * np.array(scaling + scaling).astype(np.float32)
                mask_length = np.sum((lines_3D[:,:3] - lines_3D[:,3:6])**2,axis=1)**0.5 > global_config["pose_and_shape_probabilistic"]["reproject_lines"]["min_length_line"]
                lines_3D = lines_3D[mask_length]

                # number_lines_3d[name] = lines_3D.shape[0]
                # with open(out_file_line_number,'w') as file_number_lines:
                #     json.dump(number_lines_3d,file_number_lines)

                line_dirs_3D = torch.from_numpy(lines_3D[:,:3] - lines_3D[:,3:6])
                line_dirs_3D = line_dirs_3D / torch.linalg.norm(line_dirs_3D,axis=1).unsqueeze(1).tile(1,3)
                # because lines saved as points but need as point + direction 
                lines_3D[:,3:6] = lines_3D[:,3:6] - lines_3D[:,:3]




                line_path = target_folder + '/lines_2d_filtered/' + detection + '.npy'
                lines_2D = torch.Tensor(np.load(line_path)).long()

                multiplier_lines = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["multiplier_lines"]
                if len(lines_2D.shape) == 1 or lines_3D.shape[0] == 0:
                    best_T_index = 0
                    max_factor = 0
                    lines_available = False
                else:
                    lines_available = True
                    mask = ~(lines_2D[:,:2] == lines_2D[:,2:4]).all(dim=1)
                    lines_2D = lines_2D[mask]

            if gt_infos["img"] in visualisation_list:
                img_path = target_folder + '/images/' + gt_infos["img"]
                model_path = global_config["dataset"]["dir_path"] + retrieval_list[nn_index]["model"]

                if signal == 'bbox':
                    if not lines_3D.shape[0] == 0:
                        img = plot_bbox(torch.Tensor(R).unsqueeze(0),torch.Tensor(T).unsqueeze(0).unsqueeze(0),scaling,model_path,img_path,sw,device,lines_3D,f,torch.Tensor(bbox))
                elif signal == 'lines':
                    # img = plot_lines_T(torch.Tensor(R).unsqueeze(0),torch.Tensor(T).unsqueeze(0).unsqueeze(0),scaling,model_path,img_path,sw,device,lines_3D,lines_2D,B,f,top_n_for_translation,multiplier_lines,enforce_same_length)
                    img = plot_lines_T_only_render(torch.Tensor(R).unsqueeze(0),torch.Tensor(T).unsqueeze(0).unsqueeze(0),scaling,model_path,img_path,sw,device,lines_3D,lines_2D,B,f,top_n_for_translation,multiplier_lines,enforce_same_length)
                cv2.imwrite(output_path.replace('/poses/','/{}/'.format(name_out_dir)).replace('.json','.png'),img)



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

