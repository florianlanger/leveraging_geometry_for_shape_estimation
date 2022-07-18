from re import A
from unittest import result
from black import lines_with_leading_tabs_expanded
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
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.pose import init_Rs,init_Ts,get_pb_real_grid,get_R_limits,get_T_limits, create_pose_info_dict, check_gt_pose_in_limits, get_nearest_pose_to_gt,get_T_limits_around_gt
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.ground_plane import get_model_to_infos,sample_Ts_ground_plane,filter_Ts
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.visualise_T_factors import plot_lines_T_correct_visualisation
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.model import Classification_network
from leveraging_geometry_for_shape_estimation.utilities.dicts import open_json_precomputed_or_current,determine_base_dir,load_json
from leveraging_geometry_for_shape_estimation.data_conversion.create_dirs import dict_replace_value

from probabilistic_formulation.utilities import create_all_possible_combinations,get_uvuv_p_from_superpoint,create_all_possible_combinations_uvuv_p_together
from probabilistic_formulation.factors.factors_T.factors_lines_multiple_T import get_factor_reproject_lines_multiple_T, get_factor_reproject_lines_multiple_T_threshold, get_factor_reproject_lines_multiple_T_with_Scale,get_factor_reproject_lines_multiple_T_threshold_map_single_3d_line,get_factor_reproject_lines_multiple_T_threshold_map_single_3d_line_v3
from probabilistic_formulation.factors.factors_T.factor_pose_classifier import get_factor_reproject_lines_multiple_T_classifier,get_factor_reproject_lines_multiple_T_classifier_v2,get_factor_reproject_lines_multiple_T_classifier_v3,get_factor_reproject_lines_multiple_T_classifier_v4,get_factor_reproject_lines_multiple_T_classifier_v5
from probabilistic_formulation.factors.factors_T.bbox import get_factor_bbox_multiple_T
from probabilistic_formulation.factors.factors_T.points import get_factor_reproject_kp_multiple_T
from probabilistic_formulation.tests.test_reproject_lines import load_lines_2D,load_lines_3D,get_cuboid_line_dirs_3D,plot_lines_T,plot_bbox, plot_points_T



def get_variables(global_config):

    target_folder = global_config["general"]["target_folder"]
    top_n_for_translation = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["top_n_for_translation"]
    pose_config = global_config["pose_and_shape"]["pose"]
    sensor_width = pose_config["sensor_width"]
    device = torch.device("cuda:{}".format(global_config["general"]["gpu"]))
    torch.cuda.set_device(device)
    # device = torch.device("cpu")
    # print('CPU !!')
    

    return target_folder,top_n_for_translation,pose_config,sensor_width,device


def load_infos(target_folder,detection,gt_name):

    with open(target_folder + '/global_stats/visualisation_images.json','r') as open_f:
        visualisation_list = json.load(open_f)

    with open(target_folder + '/nn_infos/' + detection + '.json','r') as open_f:
        retrieval_list = json.load(open_f)["nearest_neighbours"]

    with open(target_folder + '/gt_infos/' + gt_name + '.json','r') as open_f:
        gt_infos = json.load(open_f)

    with open(target_folder + '/segmentation_infos/' + detection + '.json','r') as open_f:
        segmentation_infos = json.load(open_f)

    with open(target_folder + '/bbox_overlap/' + detection + '.json','r') as f:
        bbox_overlap = json.load(f)

    return visualisation_list,retrieval_list,gt_infos,segmentation_infos,bbox_overlap

def load_infos_v2(target_folder,detection,gt_name,global_config):

 
    visualisation_list = open_json_precomputed_or_current('/global_stats/visualisation_images.json',global_config,None)
    retrieval_list = open_json_precomputed_or_current('/nn_infos/' + detection + '.json',global_config,'retrieval')["nearest_neighbours"]
    gt_infos = open_json_precomputed_or_current('/gt_infos/' + gt_name + '.json',global_config,'segmentation')
    segmentation_infos = open_json_precomputed_or_current('/segmentation_infos/' + detection + '.json',global_config,'segmentation')
    bbox_overlap = open_json_precomputed_or_current('/bbox_overlap/' + detection + '.json',global_config,'segmentation')

    return visualisation_list,retrieval_list,gt_infos,segmentation_infos,bbox_overlap


def get_infos_gt(gt_infos,pose_config,global_config,bbox_overlap):

    f = gt_infos["focal_length"]
    w = gt_infos["img_size"][0]
    h = gt_infos["img_size"][1]
    sw = pose_config["sensor_width"]
    enforce_same_length = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["enforce_same_length"]

    gt_rot = gt_infos["objects"][bbox_overlap['index_gt_objects']]["rot_mat"]
    gt_trans = gt_infos["objects"][bbox_overlap['index_gt_objects']]["trans_mat"]
    gt_scaling = gt_infos["objects"][bbox_overlap['index_gt_objects']]["scaling"]
    gt_z = gt_trans[2]

    return f,w,h,sw,enforce_same_length,gt_rot,gt_trans,gt_scaling,gt_z


def get_Ts_ground_plane(model_path,model_to_infos,gt_scaling,global_config,segmentation_infos,f,sw,w,h,R):
    catname_model = model_path.split('/')[1] + '_' + model_path.split('/')[2] 
    model_infos = model_to_infos[catname_model]
    height_object_center_above_ground = gt_scaling[1] *  (model_infos['bbox'][1] - model_infos['center'][1])
    Ts,Ts_ground = sample_Ts_ground_plane(np.array(R),height_object_center_above_ground,global_config["pose_and_shape_probabilistic"]["ground_plane_limits"],segmentation_infos["predictions"]["category"])
    Ts = filter_Ts(Ts,f,sw,w,h)
    return Ts


def get_best_index(factors_batch,area_accepted_all_Ts):
    max_accepted = torch.max(factors_batch)

    # only consider those with max number accepted
    mask = factors_batch == max_accepted
    area_accepted_all_Ts[~mask] = area_accepted_all_Ts[~mask] * 0 + 2*torch.max(area_accepted_all_Ts)
    return torch.argmin(area_accepted_all_Ts),area_accepted_all_Ts



def create_storage_dict(factor_output,indices,names,signal):
    assert len(indices) == len(names)

    if signal == 'lines':
        general_names = ['n_accepted_all_Ts','area_accepted_all_Ts','lines_2D','S','R','T','model_path','area_threshold']
        specific_names = ['T','all_factors_T_2d_3d','all_factors_T_2d','all_angles_T_2d_3d','all_angles_T_2d','all_lines_3d_T_3D_4','n_accepted_all_Ts','area_accepted_all_Ts','all_indices_2d_to_3d','all_accepted_T_2d','all_max_indices']
    if signal == 'lines_classifier':
        general_names = ['factors','lines_2D','S','R','T','model_path']
        specific_names = ['T','factors','inputs']
    elif signal == 'bbox':
        general_names = ['box_iou','bbox_each_T','pred_bbox','S','R','T','model_path']
        specific_names = ['T','box_iou','bbox_each_T','all_lines_3d_T_3D_4']

    new_dict = {}
    # for key in factor_output:
    # for key in ['n_accepted_all_Ts','area_accepted_all_Ts','lines_2D','S','R','T','model_path','area_threshold','all_factors_T_2d','all_angles_T_2d','all_accepted_T_2d','all_indices_2d_to_3d']:
    for key in general_names:
        new_dict[key] = np.array(factor_output[key])


    for i in range(len(indices)):
        for name in specific_names:
            new_dict[name + '_' + names[i]] = np.array(factor_output[name][indices[i]])
    return new_dict

def add_to_dict(existing_dict,names,variables):
    assert len(names) ==  len(variables)
    for i in range(len(names)):
        existing_dict[names[i]] = variables[i]
    return existing_dict

def load_3D_lines_even_not_existent(global_config,model_path,line_indices_3d,scaling):

    path =  global_config["pose_and_shape_probabilistic"]["reproject_lines"]["line_dir_3d_precise"] + "/" + model_path.split('/')[1] + '_' + model_path.split('/')[2] + '.npy'
    if not os.path.exists(path):
        path = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["line_dir_3d_self_scanned"] + '/' + model_path.split('/')[1] + '_' + model_path.split('/')[2] + '.npy'

    if os.path.exists(path):
        lines_3D = load_lines_3D(path)
        lines_3D_available = True

        if lines_3D.shape[0] == 0:
            lines_3D = np.array([[0.,0.,0,0.5,0.,0]]).astype(np.float32)
            lines_3D_available = False

    else:
        lines_3D = np.array([[0.,0.,0,0.5,0.,0]]).astype(np.float32)
        lines_3D_available = False

    if line_indices_3d != None:
        lines_3D = lines_3D[line_indices_3d,:]

    lines_3D = lines_3D * np.array(scaling + scaling).astype(np.float32)
    lines_3D[:,3:6] = lines_3D[:,3:6] - lines_3D[:,:3]

    return lines_3D,lines_3D_available

def load_2D_lines_even_not_existent(line_dir_2d,detection,line_indices_2d,w,h):
    line_path = line_dir_2d + '/' + detection + '.npy'
    lines_2D = np.load(line_path)
    lines_2D_available = True

    if len(lines_2D.shape) == 1:
        lines_2D = np.array([[h/2.,w/4.,h/2,3*w/4.]])
        lines_2D_available = False

    lines_2D=lines_2D[:,[1,0,3,2]]
    lines_2D = torch.Tensor(lines_2D).long()

    if line_indices_2d != None:
        lines_2D = lines_2D[line_indices_2d,:]

    return lines_2D,lines_2D_available



def signal_bbox_or_line(global_config,model_path,target_folder,detection,device,R,Ts,bbox,f,w,h,sw,Ss,B,enforce_same_length,top_n_for_translation,line_dir_2d,line_indices_2d=None,line_indices_3d=None,use_gt_retrieval=False,use_gt_R=False,network=None,classifier_config=None):
    multiplier_lines = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["multiplier_lines"]
    
    area_threshold_side_length = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["area_threshold_percentage_side_length"]
    area_threshold = (area_threshold_side_length * np.max([w,h])) **2
    angle_threshold = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["angle_threshold"]
    only_allow_single_mapping_to_3d = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["only_allow_single_mapping_to_3d"]
    min_line_overlap = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["min_line_overlap_percentage_side_length"] * np.max([w,h])

    model_path_full  = global_config['dataset']['dir_path'] + model_path
    scaling = list(Ss[0])

    lines_3D,lines_3D_available = load_3D_lines_even_not_existent(global_config,model_path,line_indices_3d,scaling)
    lines_2D,lines_2D_available = load_2D_lines_even_not_existent(line_dir_2d,detection,line_indices_2d,w,h)

    # line_path = line_dir_2d + '/' + detection + '.npy'
    # lines_2D = np.load(line_path)
    # lines_2D=lines_2D[:,[1,0,3,2]]
    # lines_2D = torch.Tensor(lines_2D).long()

    # if line_indices_2d != None:
    #     lines_2D = lines_2D[line_indices_2d,:]

    # if len(lines_2D.shape) == 1 or lines_3D.shape[0] == 0:
    #     best_T_index = 0
    #     best_S_index = 0
    #     max_factor = 0
    #     lines_available = False
    #     factors = None
    #     area_accepted_all_Ts = None
    #     factor_output = None
    # else:
    # lines_available = True

    if global_config["pose_and_shape_probabilistic"]["reproject_lines"]["signal"] == 'bbox':
        factor_output = get_factor_bbox_multiple_T(torch.Tensor(R).to(device),torch.Tensor(Ts).to(device),torch.from_numpy(lines_3D).to(device),torch.Tensor(bbox).to(device),f,w,h,sw)
        factors_batch = factor_output['box_iou']
        best_index_overall = torch.argmax(factors_batch)

    elif global_config["pose_and_shape_probabilistic"]["reproject_lines"]["signal"] == 'lines':
        # factors_batch,_,_ = get_factor_reproject_lines_multiple_T_threshold(torch.Tensor(R).to(device),torch.Tensor(Ts).to(device),torch.from_numpy(lines_3D).to(device),lines_2D.to(device),B.to(device),f,sw,area_threshold)
        # factors_batch,area_accepted_all_Ts,_,_ = get_factor_reproject_lines_multiple_T_threshold_map_single_3d_line_v3(torch.Tensor(R).to(device),torch.Tensor(Ts).to(device),torch.from_numpy(lines_3D).to(device),lines_2D.to(device),B.to(device),f,sw,area_threshold,angle_threshold,only_allow_single_mapping_to_3d)
        factor_output = get_factor_reproject_lines_multiple_T_threshold_map_single_3d_line_v3(torch.Tensor(R).to(device),torch.Tensor(Ts).to(device),torch.from_numpy(lines_3D).to(device),lines_2D.to(device),B.to(device),f,sw,area_threshold,angle_threshold,only_allow_single_mapping_to_3d,min_line_overlap)
        factors_batch = factor_output['n_accepted_all_Ts']
        area_accepted_all_Ts = factor_output['area_accepted_all_Ts']
        best_index_overall,area_accepted_all_Ts = get_best_index(factors_batch,area_accepted_all_Ts)
        area_accepted_all_Ts = area_accepted_all_Ts.cpu().tolist()

    elif global_config["pose_and_shape_probabilistic"]["reproject_lines"]["signal"] == 'lines_classifier':
        # factors_batch,_,_ = get_factor_reproject_lines_multiple_T_threshold(torch.Tensor(R).to(device),torch.Tensor(Ts).to(device),torch.from_numpy(lines_3D).to(device),lines_2D.to(device),B.to(device),f,sw,area_threshold)
        # factors_batch,area_accepted_all_Ts,_,_ = get_factor_reproject_lines_multiple_T_threshold_map_single_3d_line_v3(torch.Tensor(R).to(device),torch.Tensor(Ts).to(device),torch.from_numpy(lines_3D).to(device),lines_2D.to(device),B.to(device),f,sw,area_threshold,angle_threshold,only_allow_single_mapping_to_3d)

        # factor_output_v1 = get_factor_reproject_lines_multiple_T_classifier(torch.Tensor(R).to(device),torch.Tensor(Ts).to(device),torch.from_numpy(lines_3D).to(device),lines_2D.to(device),B.to(device),f,sw,bbox,network)
        # factor_output_v2 = get_factor_reproject_lines_multiple_T_classifier_v2(torch.Tensor(R).to(device),torch.Tensor(Ts).to(device),torch.from_numpy(lines_3D).to(device),lines_2D.to(device),B.to(device),f,sw,bbox,network)
        gt_name = detection.rsplit('_',1)[0]
        # factor_output_v3 = get_factor_reproject_lines_multiple_T_classifier_v3(torch.Tensor(R).to(device),torch.Tensor(Ts).to(device),torch.from_numpy(lines_3D).to(device),lines_2D.to(device),gt_name,B.to(device),f,sw,bbox,network,classifier_config,bs=200)
        factor_output_v4 = get_factor_reproject_lines_multiple_T_classifier_v5(torch.Tensor(R).to(device),torch.Tensor(Ts).to(device),torch.from_numpy(lines_3D).to(device),lines_2D.to(device),gt_name,B.to(device),f,sw,bbox,network,classifier_config,bs=200)
        # factor_output_v4 = factor_output_v1
        # print(factor_output_v3['factors'])
        # print(factor_output_v4['factors'])
        # mask = (factor_output_v4['factors'] - factor_output_v3['factors'] > 1e-6)
        # print(factor_output_v3['factors'][mask])
        # print(factor_output_v4['factors'][mask])
        # assert (factor_output_v4['inputs'] - factor_output_v3['inputs'] < 1e-9).all()
        # assert (factor_output_v4['factors'] - factor_output_v3['factors'] < 1e-6).all(),(factor_output_v3['factors'][mask],factor_output_v4['factors'][mask])

        # print(' ')
        # print('')
        # print(factor_output_v2["factors"][-10:])
        # for key,threshold in zip(["inputs","factors"],[1e-9,1e-4]):
        # #     for i in range(factor_output_v2[key].shape[0]):
        # #         # print(i)
        # #         path_1 = '/scratches/octopus_2/fml35/experiments/classify_T_from_lines_02/date_2022_03_30_time_17_52_49_bbox_augmentation_ADAM/debug_vis/img_'+str(i).zfill(2)+'_v1.png'
        # #         path_2 = '/scratches/octopus_2/fml35/experiments/classify_T_from_lines_02/date_2022_03_30_time_17_52_49_bbox_augmentation_ADAM/debug_vis/img_'+str(i).zfill(2)+'_v2.png'
        # #         # assert os.path.exists('/scratch2/fml35/experiments/classify_T_from_lines_02/date_2022_03_30_time_17_52_49_bbox_augmentation_ADAM/debug_vis')
        # #         cv2.imwrite(path_1, np.round((factor_output[key][i].numpy() / 2. + 0.5) * 255).astype(np.uint8))
        # #         cv2.imwrite(path_2, np.round((factor_output_v2[key][i].numpy() / 2. + 0.5) * 255).astype(np.uint8))
        # #         # print(factor_output[key].numpy().shape)
        # # #     # print(torch.abs(factor_output[key] - factor_output_v2[key]))
        #     print(factor_output[key])
        #     diff = torch.abs(factor_output[key] - factor_output_v2[key])
        #     print(diff.shape)
        #     assert (diff < threshold).all(),diff[diff>threshold]

        #     print('asserted ',key)
        factor_output = factor_output_v4
        factors_batch = factor_output['factors']
       
        best_index_overall= torch.argmax(factors_batch)
        
        # factors_batch,_ = get_factor_reproject_lines_multiple_T_with_Scale(torch.Tensor(R).to(device),torch.Tensor(Ts).to(device),torch.Tensor(Ss).to(device),torch.from_numpy(lines_3D).to(device),lines_2D.to(device),B.to(device),top_n_for_translation,f,multiplier_lines,enforce_same_length)
    factors = factors_batch.cpu().tolist()
    best_T_index = (best_index_overall // Ss.shape[0]).item()
    best_S_index = (best_index_overall % Ss.shape[0]).item()
    max_factor = max(factors)


    factor_output = add_to_dict(factor_output,['T','R','S','model_path','area_threshold'],[Ts,R,scaling,np.array([model_path_full],dtype=object),area_threshold])

    return best_T_index,best_S_index,max_factor,lines_3D,lines_2D,factor_output,lines_3D_available,lines_2D_available

def signal_points(global_config,model_path,target_folder,detection,device,R,Ts,B,top_n_for_translation,f):
    points_3D,_ = load_ply(global_config["pose_and_shape_probabilistic"]["reproject_lines"]["point_dir"] + '/' + model_path.split('/')[1] + '/' + model_path.split('/')[2] + '.ply')
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
    return best_T_index,max_factor,points_2D,points_available,points_3D,multiplier_points,point_threshold,use_threshold


def get_closest_index(grid,query):
    distance_gt = torch.sum((grid - torch.Tensor(query).unsqueeze(0).repeat(grid.shape[0],1))**2,dim=1)
    closest_gt_t_index = torch.argmin(distance_gt).item()
    return closest_gt_t_index



def get_R(target_folder,name,gt_infos,bbox_overlap,global_config,gt=False):
    if gt == True:
        R  = gt_infos["objects"][bbox_overlap['index_gt_objects']]["rot_mat"]
    else:
        R = open_json_precomputed_or_current('/poses_R/' + name,global_config,'R')["predicted_r"]

        # base_path = determine_base_dir(global_config,'R')
        # poses_R_selected = open_json_precomputed_or_current('/poses_R_selected/' + name,global_config,'R')
        # R = open_json_precomputed_or_current('/poses_R/' + name.split('.')[0] + '_' + str(poses_R_selected["R_index"]).zfill(2) + '.json',global_config,'R')["predicted_r"]

    sp_rot = scipy_rot.from_matrix(R)
    tilt,azim,elev = sp_rot.as_euler('zyx',degrees=True)
    tilts,azims,elevs = [tilt-0.0001,tilt+0.0001,1],[azim-0.0001,azim+0.0001,1],[elev-0.0001,elev+0.0001,1]
    return R,tilts,azims,elevs

def get_S(gt_scaling,gt=True):
    assert gt == True
    # SAMPLE SS
    # Ss = np.concatenate((0.5*np.reshape(np.array(scaling),(1,3)),np.reshape(np.array(scaling),(1,3))))
    # Ss = init_Ts((0.5,3,8),(0.5,3,8),(0.5,3,8))
    Ss = np.reshape(np.array(gt_scaling),(1,3))
    return Ss


def get_Ts(f,w,h,sensor_width,global_config,bbox,gt_z,gt_T,model_path,model_to_infos,gt_scaling,segmentation_infos,R,specific_Ts):
    
    if global_config["pose_and_shape_probabilistic"]["pose"]["sample_around_gt_z"] == False:
        xs,ys,zs = get_T_limits(f,[w,h],sensor_width,global_config["pose_and_shape_probabilistic"]["pose"],bbox,gt_z)
    elif global_config["pose_and_shape_probabilistic"]["pose"]["sample_around_gt_z"] == True:
        xs,ys,zs = get_T_limits_around_gt(gt_T,global_config["pose_and_shape_probabilistic"]["pose"])

    if specific_Ts != None:
        Ts = np.array(specific_Ts)
        assert len(Ts.shape) == 2 and Ts.shape[1] == 3, Ts.shape + 'specific Ts needs to be list of lists'

    else:
        if global_config["pose_and_shape_probabilistic"]["sample_in_ground_plane"] == "True":
            Ts = get_Ts_ground_plane(model_path,model_to_infos,gt_scaling,global_config,segmentation_infos,f,sensor_width,w,h,R)
            
        elif global_config["pose_and_shape_probabilistic"]["sample_in_ground_plane"] == "False":
            Ts = init_Ts(xs,ys,zs)

        if Ts.shape[0] == 0:
            Ts = np.array([[0,0,1]])

    return Ts,xs,ys,zs

def get_model_path(retrieval_list,nn_index,gt_infos,bbox_overlap,gt=False):
    if gt == True:
        model_path = gt_infos["objects"][bbox_overlap['index_gt_objects']]["model"]
    else:
        model_path = retrieval_list[nn_index]["model"]
    return model_path

def get_pose_single_example(global_config,name,model_to_infos,line_indices_2d=None,line_indices_3d=None,use_gt_retrieval=False,use_gt_R=False,specific_Ts=None,network=None,classifier_config=None):

    target_folder,top_n_for_translation,pose_config,sensor_width,device = get_variables(global_config)

    retrieval = name.rsplit('_',1)[0]
    detection = name.rsplit('_',2)[0]
    gt_name = name.rsplit('_',3)[0]
    nn_index = int(retrieval.split('_')[-1])
    line_dir_2d = determine_base_dir(global_config,'lines') + '/' + 'lines_2d_filtered'

    visualisation_list,retrieval_list,gt_infos,segmentation_infos,bbox_overlap = load_infos_v2(target_folder,detection,gt_name,global_config)

    if nn_index >= len(retrieval_list):
        return None,[]


    f,w,h,sw,enforce_same_length,gt_rot,gt_trans,gt_scaling,gt_z = get_infos_gt(gt_infos,pose_config,global_config,bbox_overlap)

    # lines,available = load_2D_lines_even_not_existent(line_dir_2d,detection,line_indices_2d,w,h)
    # assert available == True, (name)
    # if available == False:
    #     print(name)
    # return None,None,None

    B = get_pb_real_grid(w,h,f,sw,device)
    bbox = segmentation_infos["predictions"]["bbox"]
    model_path = get_model_path(retrieval_list,nn_index,gt_infos,bbox_overlap,gt=use_gt_retrieval)

    R,tilts,azims,elevs = get_R(target_folder,name,gt_infos,bbox_overlap,global_config,gt=use_gt_R)
    Ss = get_S(gt_scaling,gt=True)
    Ts,xs,ys,zs = get_Ts(f,w,h,sensor_width,global_config,bbox,gt_z,gt_trans,model_path,model_to_infos,gt_scaling,segmentation_infos,R,specific_Ts)
    

    gt_pose_in_limits = check_gt_pose_in_limits(xs,ys,zs,tilts,elevs,azims,gt_trans,gt_rot)
    best_T_possible,best_R_possible = get_nearest_pose_to_gt(xs,ys,zs,tilts,elevs,azims,gt_trans,gt_rot)



    # compute factors
    Ts = torch.Tensor(Ts)
    signal = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["signal"]
    if signal == 'bbox' or signal == 'lines' or signal == 'lines_classifier':
        with torch.no_grad():
            best_T_index,best_S_index,max_factor,lines_3D,lines_2D,factor_output,lines_3D_available,lines_2D_available = signal_bbox_or_line(global_config,model_path,target_folder,detection,device,R,Ts,bbox,f,w,h,sw,Ss,B,enforce_same_length,top_n_for_translation,line_dir_2d,line_indices_2d,line_indices_3d,use_gt_retrieval,use_gt_R,network,classifier_config)
    elif signal == 'points':
        best_T_index,max_factor,points_2D,points_available,points_3D,multiplier_points,point_threshold,use_threshold = signal_points(global_config,model_path,target_folder,detection,device,R,Ts,B,top_n_for_translation,f)
        

    T = Ts[best_T_index]
    S = Ss[best_S_index]
    n_indices = 1
    pose_information = create_pose_info_dict(np.array(R),T,n_indices,max_factor,gt_pose_in_limits,gt_rot,gt_trans,best_R_possible,best_T_possible,xs,ys,zs,tilts,elevs,azims,gt_scaling,lines_3D_available,lines_2D_available)
    

    closest_gt_t_index = get_closest_index(Ts,gt_trans)

    if factor_output != None:
        factor_output = create_storage_dict(factor_output,[best_T_index,closest_gt_t_index],['selected_T','closest_gt'],signal)

    factor_information = {"Ss":Ss,"Ts":Ts.cpu().numpy(),"R":R,"best_S_index":best_S_index,"best_T_index":best_T_index,"lines_3D":lines_3D,"lines_2D":lines_2D.cpu().numpy()}

    return pose_information,factor_information,factor_output


def save_img_annotation(path,img_annotation):
    text = ''
    for line in img_annotation:
        text += line + '\n'
    with open(path,'w') as f:
        f.write(text)

def get_indices_factors(factors,area_accepted_all_Ts,top_n):
    max_n_accepted = np.max(factors)
    indices_T = []

    for n_accepted in range(int(max_n_accepted),0,-1):
        # only consider those with max number accepted
        mask = factors == n_accepted
        number_T_with_n_accepted = np.sum(mask)
        area_accepted_current_n = area_accepted_all_Ts.copy()
        area_accepted_current_n[~mask] = area_accepted_current_n[~mask] * 0 + 2*np.max(area_accepted_all_Ts)

        sort_indices = np.argsort(area_accepted_current_n)
        indices_T += sort_indices.tolist()[:number_T_with_n_accepted]
        if len(indices_T) > max(top_n):
            break
    return indices_T


def eval_factors(factor_output,gt_T,T_threshold=0.2,top_n=[1,5,10,30,100]):

    if "n_accepted_all_Ts" in factor_output:
        indices_T = get_indices_factors(factor_output["n_accepted_all_Ts"],factor_output["area_accepted_all_Ts"],top_n)
    elif "box_iou" in factor_output:
        indices_T = np.argsort(-1*factor_output['box_iou'])
    elif "factors" in factor_output:
        indices_T = np.argsort(-1*factor_output['factors'])

    all_Ts = factor_output["T"]
    sorted_Ts = all_Ts[indices_T]
    dists = np.linalg.norm(sorted_Ts - np.repeat(np.expand_dims(gt_T,axis=0),sorted_Ts.shape[0],axis=0),axis=1)
    correct = dists < T_threshold
    results = {}
    for n in top_n:
        results["top_" + str(n)] = np.sum(correct[:n]).item()
    
    if correct.shape[0] == 0:
        i = 10000

    else:
        for i in range(correct.shape[0]):
            if correct[i] == True:
                break
    
    results['first_correct'] = i + 1
        
    return results

def get_classifier_path_add_on(global_config):
    if global_config["pose_and_shape_probabilistic"]["reproject_lines"]["classifier_epoch"] == 'last':
        return 'last_epoch.pth'
    else:
        return 'epoch_{}.pth'.format(str(global_config["pose_and_shape_probabilistic"]["reproject_lines"]["classifier_epoch"]).zfill(6))

def load_classifier(global_config,classifier_config):
    if global_config["pose_and_shape_probabilistic"]["reproject_lines"]["signal"] == 'lines_classifier':
        device = torch.device("cuda:{}".format(global_config["general"]["gpu"]))
        print(device)
        print('load classifier')
        network = Classification_network(classifier_config,torch.device("cpu"))
        network_path = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["classifier_exp_path"] + '/saved_models/' + get_classifier_path_add_on(global_config)
        assert os.path.exists(network_path),'classifier path does not exist'
        network.load_state_dict(torch.load(network_path,map_location="cpu"))
        # print('dont load network')
        network.set_device(device)
        network.eval()
        network.pretrained_model.eval()
        print('loaded classifier')
    else:
        network = None
    return network

def get_pose_for_folder(global_config):

    target_folder = global_config["general"]["target_folder"]
    model_to_infos = get_model_to_infos()

    classifier_config = load_json(global_config["pose_and_shape_probabilistic"]["reproject_lines"]["classifier_exp_path"] + '/code/leveraging_geometry_for_shape_estimation/probabilistic_pose_and_shape/train_pose_classifier_from_lines/config.json')
    if global_config["general"]["run_on_octopus"] == 'False':
        classifier_config = dict_replace_value(classifier_config,'/scratch/fml35/','/scratches/octopus/fml35/')
        classifier_config = dict_replace_value(classifier_config,'/scratch2/fml35/','/scratches/octopus_2/fml35/')

    network = load_classifier(global_config,classifier_config)
    # network = None

    use_gt_R = global_config["pose_and_shape_probabilistic"]["pose"]["gt_R"]

    for name in tqdm(sorted(os.listdir(determine_base_dir(global_config,'R') + '/poses_R'))):
    # for name in tqdm(sorted(os.listdir('/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_210_line_min_overlap/poses_R'))):
        if use_gt_R == True and not '_00.json' in name:
            continue

        rotation_index = int(name.split('_')[-1].split('.')[0])

        if global_config["R"]["choose_R"] == "closest_gt" or global_config["R"]["choose_R"] == "retrieval":

            if not rotation_index == load_json(determine_base_dir(global_config,'R') + '/poses_R_selected/' + name.rsplit('_',1)[0] + '.json')["R_index"]:
                continue

        pose_information,factor_information,factor_output = get_pose_single_example(global_config,name,model_to_infos,use_gt_R=use_gt_R,network=network,classifier_config=classifier_config)
        
        if pose_information == None:
            # print('None')
            continue
        

        eval_factors_results = eval_factors(factor_output,np.array(pose_information["gt_T"]),T_threshold=0.2,top_n=[1,5,10,30,100])

        pose_information = {**pose_information,**eval_factors_results}
        output_path = target_folder + '/poses/' + name
        with open(output_path,'w') as open_f:
            json.dump(pose_information,open_f,indent=4)

        np.savez(target_folder + '/T_lines_factors/' + name.replace('.json','.npz'), **factor_output)


            


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

