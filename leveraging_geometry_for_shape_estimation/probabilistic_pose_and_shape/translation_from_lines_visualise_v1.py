from re import A
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
from matplotlib import pyplot as plt
import pickle

from leveraging_geometry_for_shape_estimation.utilities.dicts import determine_base_dir,load_json
from leveraging_geometry_for_shape_estimation.utilities.write_on_images import draw_text_block
from leveraging_geometry_for_shape_estimation.keypoint_matching.get_matches_3d import load_information_depth_camera,create_pixel_bearing,pb_and_depth_to_wc
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.pose import init_Rs,init_Ts,get_pb_real_grid,get_R_limits,get_T_limits, create_pose_info_dict, check_gt_pose_in_limits, get_nearest_pose_to_gt
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.ground_plane import get_model_to_infos,sample_Ts_ground_plane,filter_Ts
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.visualise_T_factors import plot_lines_T_correct_visualisation,plot_lines_T_correct_visualisation_v2
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.translation_from_lines_v8 import get_variables,load_infos_v2,get_infos_gt,get_model_path
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.visualise_bbox import plot_bbox_T_v2
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.visualisation.visualisation import plot_points_channels
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.main import process_config

from probabilistic_formulation.utilities import create_all_possible_combinations,get_uvuv_p_from_superpoint,create_all_possible_combinations_uvuv_p_together
from probabilistic_formulation.factors.factors_T.factors_lines_multiple_T import get_factor_reproject_lines_multiple_T, get_factor_reproject_lines_multiple_T_threshold, get_factor_reproject_lines_multiple_T_with_Scale,get_factor_reproject_lines_multiple_T_threshold_map_single_3d_line,get_factor_reproject_lines_multiple_T_threshold_map_single_3d_line_v2
from probabilistic_formulation.factors.factors_T.bbox import get_factor_bbox_multiple_T
from probabilistic_formulation.factors.factors_T.points import get_factor_reproject_kp_multiple_T
from probabilistic_formulation.tests.test_reproject_lines import load_lines_2D,load_lines_3D,get_cuboid_line_dirs_3D,plot_lines_T,plot_bbox, plot_points_T



def get_closest_index(grid,query):
    distance_gt = torch.sum((grid - torch.Tensor(query).unsqueeze(0).repeat(grid.shape[0],1))**2,dim=1)
    closest_gt_t_index = torch.argmin(distance_gt).item()
    return closest_gt_t_index


def visualise_translation(target_folder,gt_infos,global_config,model_path,signal,lines_3D,gt_scaling,sw,device,f,bbox,S,T,R,lines_2D,B,top_n_for_translation,multiplier_lines,enforce_same_length,points_3D,points_2D,multiplier_points,point_threshold,use_threshold,lines_available,points_available,area_threshold,angle_threshold,only_allow_single_mapping_to_3d):


    img_path = target_folder + '/images/' + gt_infos["img"]
    full_model_path = global_config["dataset"]["dir_path"] + model_path

    if signal == 'bbox':
        if not lines_3D.shape[0] == 0:
            img = plot_bbox(torch.Tensor(R).unsqueeze(0),torch.Tensor(T).unsqueeze(0).unsqueeze(0),gt_scaling,full_model_path,img_path,sw,device,lines_3D,f,torch.Tensor(bbox))
    elif signal == 'lines':
        if lines_available:
            img,img_annotation = plot_lines_T_correct_visualisation(torch.Tensor(R).unsqueeze(0),torch.Tensor(T).unsqueeze(0).unsqueeze(0),S,full_model_path,img_path,sw,device,lines_3D,lines_2D,B,f,area_threshold,angle_threshold,only_allow_single_mapping_to_3d)
        else:
            img = cv2.imread(target_folder + '/images/' + gt_infos["img"])
            img_annotation = []
    elif signal == 'points':
        if points_available:
            img = plot_points_T(torch.Tensor(R).unsqueeze(0),torch.Tensor(T).unsqueeze(0).unsqueeze(0),gt_scaling,full_model_path,img_path,sw,device,points_3D,points_2D,B,f,top_n_for_translation,multiplier_points,point_threshold,use_threshold)
        else:
            img = cv2.imread(target_folder + '/images/' + gt_infos["img"])

    return img,img_annotation




def visualise_single_example(global_config,name,results_arrays,img_names,model_to_infos,classifier_config):

    target_folder,top_n_for_translation,pose_config,sensor_width,device = get_variables(global_config)

    retrieval = name.rsplit('_',2)[0]
    detection = name.rsplit('_',3)[0]
    gt_name = name.rsplit('_',4)[0]
    nn_index = int(retrieval.split('_')[-1])

    visualisation_list,retrieval_list,gt_infos,segmentation_infos,bbox_overlap = load_infos_v2(target_folder,detection,gt_name,global_config)

    if nn_index >= len(retrieval_list):
        return [],[]

    model_path = get_model_path(retrieval_list,nn_index,gt_infos,bbox_overlap)
    scaling_lines = (1/(np.array(model_to_infos[model_path.split('/')[1] + '_' + model_path.split('/')[2]]['bbox'])*2)).tolist()
    f,w,h,sw,enforce_same_length,gt_rot,gt_trans,gt_scaling,gt_z = get_infos_gt(gt_infos,pose_config,global_config,bbox_overlap,scaling_lines)
    

    imgs = []
    img_annotations = []
        
    img_path = determine_base_dir(global_config,"segmentation") + '/images/' + gt_infos["img"]

    for which in img_names:
        # factors_specific = {}
        # for key in results_arrays:
        #     if which in key:
        #         factors_specific[key.replace('_' + which,'')] = results_arrays[key]
        factors_specific = results_arrays[which]

        if "n_accepted_all_Ts" in results_arrays:
            img,img_annotation = plot_lines_T_correct_visualisation_v2(img_path,sw,device,f,results_arrays,factors_specific)
        elif "factors" in results_arrays:
            if classifier_config['data']['type'] == 'lines':
                img,img_annotation = visualise_image_classifier(results_arrays,factors_specific,img_path,gt_trans,gt_scaling)
            elif classifier_config['data']['type'] == 'points':
                img,img_annotation = visualise_image_classifier_points(results_arrays,factors_specific,img_path,gt_trans,gt_scaling)
        elif "box_iou" in results_arrays:
            img = plot_bbox_T_v2(img_path,sw,device,f,results_arrays,factors_specific)
            img_annotation = []
        # img,img_annotation = visualise_translation(target_folder,gt_infos,global_config,model_path,signal,lines_3D,gt_scaling,sw,device,f,bbox,S,T,R,lines_2D,B,top_n_for_translation,multiplier_lines,enforce_same_length,points_3D,points_2D,multiplier_points,point_threshold,use_threshold,lines_available,points_available,area_threshold,angle_threshold,only_allow_single_mapping_to_3d)
        imgs.append(img)
        img_annotations.append(img_annotation)

    return imgs,img_annotations


def save_img_annotation(path,img_annotation):
    text = ''
    for line in img_annotation:
        text += line + '\n'
    with open(path,'w') as f:
        f.write(text)

def visualise_image_classifier(results_arrays,factors_specific,img_path,gt_trans,gt_scaling):
  

    img = factors_specific['inputs'][:,:,:3]
    img = np.round((img / 2. + 0.5) * 255).astype(np.uint8)

    img_orig = cv2.imread(img_path)
    img_orig = cv2.resize(img_orig,(img.shape[1],img.shape[0]))
    img_orig = cv2.cvtColor(img_orig,cv2.COLOR_BGR2RGB)

    img_rgb = np.round((img_orig/2 + img/2 )).astype(np.uint8)

    factor = str(np.round(factors_specific['factors'].item(),4))
    offset = np.round(factors_specific['T'] - np.array(gt_trans),2).tolist()
    offset_list = ['x' + str(offset[0]),'y' + str(offset[1]),'z' + str(offset[2])]
    accepted = str(np.linalg.norm(factors_specific['T'] - np.array(gt_trans)) < 0.2)
    img_black = img_rgb * 0
    draw_text_block(img_black,[factor] + offset_list + [accepted],top_left_corner=(5,-18), font_scale=1,font_thickness=1)


    offset = np.round(factors_specific['S'] / np.array(gt_scaling) - 1,2).tolist()
    offset_list = ['x' + str(offset[0]),'y' + str(offset[1]),'z' + str(offset[2])]
    accepted = str(np.mean(np.abs(factors_specific['S'] / np.array(gt_scaling) - 1)) < 0.2)
    draw_text_block(img_black,['S'] + offset_list + [accepted],top_left_corner=(60,-18), font_scale=1,font_thickness=1)

    concat = cv2.hconcat([img,img_rgb,img_black])

    return concat,[]


def visualise_image_classifier_points(results_arrays,factors_specific,img_path,gt_trans,gt_scaling):

    # write infos
    factor = str(np.round(factors_specific['factors'].item(),4))
    offset = np.round(factors_specific['T'] - np.array(gt_trans),2).tolist()
    offset_list = ['x' + str(offset[0]),'y' + str(offset[1]),'z' + str(offset[2])]
    accepted = str(np.linalg.norm(factors_specific['T'] - np.array(gt_trans)) < 0.2)
    text = [factor] + ['T'] + offset_list + [accepted]
    offset = np.round(factors_specific['S'] / np.array(gt_scaling) - 1,2).tolist()
    offset_list = ['x' + str(offset[0]),'y' + str(offset[1]),'z' + str(offset[2])]
    accepted = str(np.mean(np.abs(factors_specific['S'] / np.array(gt_scaling) - 1)) < 0.2)
    text += ['S'] + offset_list + [accepted]


    fig = plt.figure(figsize=(10, 3))
    plt.title(text)
    plt.grid(False)
    plt.axis('off')


    point_size = 0.01
    n_rows = 1
    n_col = 3

    single_example = factors_specific['inputs']

    channel_0 = single_example[single_example[:,2] == 0,:]
    channel_1 = single_example[single_example[:,2] == 50,:]
    channel_2 = single_example[single_example[:,2] == 100,:]

    ax = fig.add_subplot(n_rows, n_col, 1, projection='3d')
    ax = plot_points_channels(ax,channel_0,channel_1,channel_2,point_size)

    ax = fig.add_subplot(n_rows, n_col, 2, projection='3d')
    ax = plot_points_channels(ax,channel_0,channel_1,channel_2,point_size)
    ax.view_init(elev=-90., azim=270)

    ax = fig.add_subplot(n_rows, n_col, 3, xticks=[], yticks=[])
    img_orig = cv2.imread(img_path)
    img_orig = cv2.resize(img_orig,(128,96))
    img_orig = cv2.cvtColor(img_orig,cv2.COLOR_BGR2RGB)
    plt.imshow(img_orig)

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    img = cv2.resize(img,(600,200))

    return img,[]


def get_pose_for_folder(global_config):


    target_folder = global_config["general"]["target_folder"]
    model_to_infos = get_model_to_infos(global_config['dataset']['which_dataset'])
    with open(target_folder + '/global_stats/visualisation_images.json','r') as open_f:
        visualisation_list = json.load(open_f)

    classifier_config = load_json(global_config["pose_and_shape_probabilistic"]["reproject_lines"]["classifier_exp_path"] + '/code/leveraging_geometry_for_shape_estimation/probabilistic_pose_and_shape/train_pose_classifier_from_lines/config.json')
    classifier_config = process_config(classifier_config)
    # img_names = ['selected_T_selected_S','closest_T_closest_S','closest_T_selected_S','selected_T_closest_S']
    img_names = ['selected_T_selected_S','closest_T_closest_S']
    # for name in tqdm(sorted(os.listdir(target_folder + '/poses_R'))):
    for name in tqdm(sorted(os.listdir(target_folder + '/poses_stages'))):
        if name.rsplit('_',4)[0] + '.jpg' in visualisation_list:
            # if not 'scene0088_01-000500_04_000_00' in name:
            #     continue
            results_path = target_folder + '/T_lines_factors/' + name.replace('.json','.pickle')
            assert os.path.exists(results_path),results_path
            with open(results_path, 'rb') as f:
                results_arrays = pickle.load(f)

            imgs,img_annotations = visualise_single_example(global_config,name,results_arrays,img_names,model_to_infos,classifier_config)

            output_path = target_folder + '/poses/' + name

            if imgs != []:
                for which,img,img_annotation in zip(img_names,imgs,img_annotations):
                    cv2.imwrite(output_path.replace('/poses/','/T_lines_vis/').replace('.json','_{}.png'.format(which)),cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    save_img_annotation(output_path.replace('/poses/','/T_lines_vis_annotations/').replace('.json','_{}.txt'.format(which)),img_annotation)


            


def main():
    np.random.seed(1)
    torch.manual_seed(0)

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    if global_config["pose_and_shape_probabilistic"]["use_probabilistic"] == "True":
        get_pose_for_folder(global_config)
    


if __name__ == '__main__':
    print('Visualise Translation from lines')
    main()

