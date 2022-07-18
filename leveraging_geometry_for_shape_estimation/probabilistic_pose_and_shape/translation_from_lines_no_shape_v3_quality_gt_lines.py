import cv2
from cv2 import threshold
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.translation_from_lines_v2 import get_infos_gt
from matplotlib import lines
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
from probabilistic_formulation.tests.test_reproject_lines import load_lines_2D,load_lines_3D,get_cuboid_line_dirs_3D,plot_lines_T,plot_bbox, plot_points_T,plot_lines_T_only_render,plot_lines_T_quality_gt


def get_pose_for_folder(global_config):

    target_folder = global_config["general"]["target_folder"]
    top_n_for_translation = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["top_n_for_translation"]

    with open(target_folder + '/global_stats/visualisation_images.json','r') as open_f:
        visualisation_list = json.load(open_f)


    print('Enforce same length FALSE')

    pose_config = global_config["pose_and_shape"]["pose"]
    # when normalise by average
    # multiplier_lines = 50
    multiplier_lines = 400

    device = torch.device("cuda:{}".format(global_config["general"]["gpu"]))
    torch.cuda.set_device(device)

    # name_out_dir = 'T_lines_vis_new_model_all_gt'
    # name_out_dir = 'T_lines_vis_new_model_filtered_gt'
    # name_out_dir = 'T_lines_vis_old_model_gt'
    name_out_dir = 'T_lines_vis_quality_2d_lines_9_absolute_threshold_filtered'

    os.mkdir(target_folder + '/quality_2D_lines/' + name_out_dir)

    # shape_dir_3d = global_config["general"]["models_folder_read"] + '/models/extract_from_2d/exp_03/lines_3d_combined_reformatted'
    shape_dir_3d = global_config["general"]["models_folder_read"] + '/models/extract_from_2d/exp_03/lines_3d_reformatted'
    # shape_dir_3d = global_config["general"]["models_folder_read"] + '/models/lines'

    # save number lines
    out_file_line_number = target_folder + '/global_stats/count_2d_lines_correct/{}.json'.format(name_out_dir)
    # assert os.path.exists(out_file_line_number) == False
    # number_lines_3d = {}
    # /scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_203_new_lines_from_2d_all_vis/T_lines_vis_quality_2d_lines_1/scene0011_00-000100_02_000_00.png

    number_lines_correct = {}

    # for name in tqdm(os.listdir(target_folder + '/cropped_and_masked')):
    for name in tqdm(sorted(os.listdir(target_folder + '/poses_R'))):
        # for rand_ind in range(0,100):
            if not "00.json" in name:
                continue
            # print(name)
                
            # if not 'scene0144_00-' in name:
            #     continue

            # if "scene0011_00-000900_00_000_00" not in name:
            #     continue
            retrieval = name.rsplit('_',1)[0]
            detection = name.rsplit('_',2)[0]
            gt_name = name.rsplit('_',3)[0]


            with open(target_folder + '/nn_infos/' + detection + '.json','r') as open_f:
                retrieval_list = json.load(open_f)["nearest_neighbours"]

            with open(target_folder + '/gt_infos/' + gt_name + '.json','r') as open_f:
                gt_infos = json.load(open_f)


            with open(target_folder + '/bbox_overlap/' + detection + '.json','r') as f:
                bbox_overlap = json.load(f)

            nn_index = int(retrieval.split('_')[-1])

            if nn_index >= len(retrieval_list):
                continue
        
            output_path = target_folder + '/poses/' + name

            # convert pixel to pixel bearing
            f = gt_infos["focal_length"]
            w = gt_infos["img_size"][0]
            h = gt_infos["img_size"][1]
            sw = pose_config["sensor_width"]
            enforce_same_length = False

            # get infos
            B = get_pb_real_grid(w,h,f,sw,device)

            R  = gt_infos["objects"][bbox_overlap['index_gt_objects']]["rot_mat"]
            scaling = gt_infos["objects"][bbox_overlap['index_gt_objects']]["scaling"]
            T = torch.Tensor(gt_infos["objects"][bbox_overlap['index_gt_objects']]["trans_mat"])

            # print('USE GT RETRIEVAL OTHERWISE PROBLEM WITH SCALE')
            retrieval_list[nn_index]["model"] = gt_infos["objects"][bbox_overlap['index_gt_objects']]["model"]

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

            line_dirs_3D = torch.from_numpy(lines_3D[:,:3] - lines_3D[:,3:6])
            line_dirs_3D = line_dirs_3D / torch.linalg.norm(line_dirs_3D,axis=1).unsqueeze(1).tile(1,3)
            # because lines saved as points but need as point + direction 
            lines_3D[:,3:6] = lines_3D[:,3:6] - lines_3D[:,:3]




            line_path = target_folder + '/lines_2d_filtered/' + detection + '.npy'
            # line_path = target_folder + '/lines_2d_cropped/' + gt_name + '.npy'

            lines_2D = np.load(line_path)
            # ensure only have unique lines
            lines_2D = np.unique(lines_2D, axis=0)

            lines_2D = torch.Tensor(lines_2D).long()

            if len(lines_2D.shape) != 1:
                mask = ~(lines_2D[:,:2] == lines_2D[:,2:4]).all(dim=1)
                lines_2D = lines_2D[mask]

            if gt_infos["img"] in visualisation_list:
                img_path = target_folder + '/images/' + gt_infos["img"]
                model_path = global_config["dataset"]["dir_path"] + retrieval_list[nn_index]["model"]
                img,correct_lines = plot_lines_T_quality_gt(torch.Tensor(R).unsqueeze(0),torch.Tensor(T).unsqueeze(0).unsqueeze(0),scaling,model_path,img_path,sw,device,lines_3D,lines_2D,B,f,top_n_for_translation,multiplier_lines,enforce_same_length)
                cv2.imwrite(output_path.replace('/poses/','/quality_2D_lines/{}/'.format(name_out_dir)).replace('.json','.png'),img)

                indices_correct = [i for i, x in enumerate(correct_lines) if x]
                if correct_lines != []:
                    save_lines_correct = lines_2D[indices_correct,:].tolist()
                else:
                    save_lines_correct = []
                n_correct = len(indices_correct)
                number_lines_correct[name] = {"n_correct": n_correct,"n_total": lines_2D.shape[0],"correct_lines":save_lines_correct}
                with open(out_file_line_number,'w') as file_number_lines:
                    json.dump(number_lines_correct,file_number_lines,indent=4)

            
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

