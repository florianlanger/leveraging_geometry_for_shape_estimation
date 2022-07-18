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

from leveraging_geometry_for_shape_estimation.keypoint_matching.get_matches_3d import load_information_depth_camera,create_pixel_bearing,pb_and_depth_to_wc
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.pose import init_Rs,init_Ts,get_pb_real_grid,get_R_limits,get_T_limits, create_pose_info_dict, check_gt_pose_in_limits, get_nearest_pose_to_gt
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.ground_plane import get_model_to_infos,sample_Ts_ground_plane,filter_Ts
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.visualise_T_factors import plot_lines_T_correct_visualisation
from probabilistic_formulation.utilities import create_all_possible_combinations,get_uvuv_p_from_superpoint,create_all_possible_combinations_uvuv_p_together
from probabilistic_formulation.factors.factors_T.factors_lines_multiple_T import get_factor_reproject_lines_multiple_T, get_factor_reproject_lines_multiple_T_threshold, get_factor_reproject_lines_multiple_T_with_Scale,get_factor_reproject_lines_multiple_T_threshold_map_single_3d_line,get_factor_reproject_lines_multiple_T_threshold_map_single_3d_line_v2
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


def signal_bbox_or_line(global_config,model_path,target_folder,detection,device,R,Ts,bbox,f,w,h,sw,Ss,B,enforce_same_length,top_n_for_translation,line_dir_2d,line_indices_2d=None,line_indices_3d=None,use_gt_retrieval=False,use_gt_R=False):
    multiplier_lines = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["multiplier_lines"]
    
    area_threshold_side_length = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["area_threshold_percentage_side_length"]
    area_threshold = (area_threshold_side_length * np.max([w,h])) **2
    angle_threshold = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["angle_threshold"]
    only_allow_single_mapping_to_3d = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["only_allow_single_mapping_to_3d"]

    path =  global_config["pose_and_shape_probabilistic"]["reproject_lines"]["line_dir_3d_precise"] + "/" + model_path.split('/')[1] + '_' + model_path.split('/')[2] + '.npy'
    if not os.path.exists(path):
        path = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["line_dir_3d_self_scanned"] + '/' + model_path.split('/')[1] + '_' + model_path.split('/')[2] + '.npy'

    if os.path.exists(path):
        lines_3D = load_lines_3D(path)
    else:
        lines_3D = np.array([[0.,0.,0,0.5,0.,0]]).astype(np.float32)

    if line_indices_3d != None:
        lines_3D = lines_3D[line_indices_3d,:]

    # print('USE SCALIGN for mask 3d lines')
    scaling = list(Ss[0])
    # lines_3D_just_for_scaling = lines_3D * np.array(scaling + scaling).astype(np.float32)
    # mask_length = np.sum((lines_3D_just_for_scaling[:,:3] - lines_3D_just_for_scaling[:,3:6])**2,axis=1)**0.5 > global_config["pose_and_shape_probabilistic"]["reproject_lines"]["min_length_line"]
    # lines_3D = lines_3D[mask_length]


    lines_3D = lines_3D * np.array(scaling + scaling).astype(np.float32)
    # mask_length = np.sum((lines_3D[:,:3] - lines_3D[:,3:6])**2,axis=1)**0.5 > global_config["pose_and_shape_probabilistic"]["reproject_lines"]["min_length_line"]
    # lines_3D = lines_3D[mask_length]

    # print('No more masking 3d lines in script or 2d lines')

    # print('only first four lines')
    # lines_3D = lines_3D[[1,2,3]]
    
    line_dirs_3D = torch.from_numpy(lines_3D[:,:3] - lines_3D[:,3:6])
    line_dirs_3D = line_dirs_3D / torch.linalg.norm(line_dirs_3D,axis=1).unsqueeze(1).tile(1,3)
    # because lines saved as points but need as point + direction 
    lines_3D[:,3:6] = lines_3D[:,3:6] - lines_3D[:,:3]


    line_path = line_dir_2d + '/' + detection + '.npy'
    lines_2D = torch.Tensor(np.load(line_path)).long()

    if line_indices_2d != None:
        lines_2D = lines_2D[line_indices_2d,:]

    if len(lines_2D.shape) == 1 or lines_3D.shape[0] == 0:
        best_T_index = 0
        best_S_index = 0
        max_factor = 0
        lines_available = False
        factors = None
        area_accepted_all_Ts = None
    else:
        lines_available = True
        # mask = ~(lines_2D[:,:2] == lines_2D[:,2:4]).all(dim=1)
        # lines_2D = lines_2D[mask]
    
        if global_config["pose_and_shape_probabilistic"]["reproject_lines"]["signal"] == 'bbox':
            factors_batch,_ = get_factor_bbox_multiple_T(torch.Tensor(R).to(device),torch.Tensor(Ts).to(device),torch.from_numpy(lines_3D).to(device),torch.Tensor(bbox).to(device),f,w,h,sw)
        elif global_config["pose_and_shape_probabilistic"]["reproject_lines"]["signal"] == 'lines':
            # factors_batch,_,_ = get_factor_reproject_lines_multiple_T_threshold(torch.Tensor(R).to(device),torch.Tensor(Ts).to(device),torch.from_numpy(lines_3D).to(device),lines_2D.to(device),B.to(device),f,sw,area_threshold)
            # print('R',R)
            # print('ts',Ts)
            # print('lines 3d',lines_3D)
            # print('lines 2d',lines_2D)
            # print('B',B[:3,:3])
            factors_batch,area_accepted_all_Ts,_,_ = get_factor_reproject_lines_multiple_T_threshold_map_single_3d_line_v2(torch.Tensor(R).to(device),torch.Tensor(Ts).to(device),torch.from_numpy(lines_3D).to(device),lines_2D.to(device),B.to(device),f,sw,area_threshold,angle_threshold,only_allow_single_mapping_to_3d)
            # factors_batch,_ = get_factor_reproject_lines_multiple_T_with_Scale(torch.Tensor(R).to(device),torch.Tensor(Ts).to(device),torch.Tensor(Ss).to(device),torch.from_numpy(lines_3D).to(device),lines_2D.to(device),B.to(device),top_n_for_translation,f,multiplier_lines,enforce_same_length)
        factors = factors_batch.cpu().tolist()
        # area_accepted_all_Ts_list = area_accepted_all_Ts.cpu().tolist()
        # best_index_overall = factors.index(max(factors))
        best_index_overall,area_accepted_all_Ts = get_best_index(factors_batch,area_accepted_all_Ts)
        best_T_index = best_index_overall // Ss.shape[0]
        best_S_index = best_index_overall % Ss.shape[0]
        max_factor = max(factors)
        area_accepted_all_Ts = area_accepted_all_Ts.cpu().tolist()

    return best_T_index,best_S_index,max_factor,lines_3D,lines_2D,lines_available,multiplier_lines,factors,area_accepted_all_Ts,area_threshold,angle_threshold,only_allow_single_mapping_to_3d

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


def visualise_translation(target_folder,gt_infos,global_config,model_path,signal,lines_3D,gt_scaling,sw,device,f,bbox,S,T,R,lines_2D,B,top_n_for_translation,multiplier_lines,enforce_same_length,points_3D,points_2D,multiplier_points,point_threshold,use_threshold,lines_available,points_available,area_threshold,angle_threshold,only_allow_single_mapping_to_3d):

    # save_path = output_path.replace('/poses/','/factors_T/').replace('.json','.npz')
    # np.savez(save_path,factors=factors,Rs=R,Ts=Ts)

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

    # cv2.putText(img, 'gt' + str(np.round(gt_trans,3)), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 128, 0), 1, cv2.LINE_AA)
    # cv2.putText(img, str(np.round(T.numpy(),3)), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 128, 0), 1, cv2.LINE_AA)
    return img,img_annotation


def get_R(target_folder,name,gt_infos,bbox_overlap,gt=False):
    if gt == True:
        R  = gt_infos["objects"][bbox_overlap['index_gt_objects']]["rot_mat"]
    else:
        with open(target_folder + '/poses_R/' + name,'r') as file:
            R = json.load(file)["predicted_r"]

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


def get_Ts(f,w,h,sensor_width,global_config,bbox,gt_z,model_path,model_to_infos,gt_scaling,segmentation_infos,R,specific_Ts):
    xs,ys,zs = get_T_limits(f,[w,h],sensor_width,global_config["pose_and_shape_probabilistic"]["pose"],bbox,gt_z)

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

def get_pose_single_example(global_config,name,model_to_infos,line_indices_2d=None,line_indices_3d=None,use_gt_retrieval=False,use_gt_R=False,specific_Ts=None):

    target_folder,top_n_for_translation,pose_config,sensor_width,device = get_variables(global_config)

    retrieval = name.rsplit('_',1)[0]
    detection = name.rsplit('_',2)[0]
    gt_name = name.rsplit('_',3)[0]
    nn_index = int(retrieval.split('_')[-1])
    line_dir_2d = target_folder + '/' + 'lines_2d_filtered'

    visualisation_list,retrieval_list,gt_infos,segmentation_infos,bbox_overlap = load_infos(target_folder,detection,gt_name)

    if nn_index >= len(retrieval_list):
        return None,[]


    f,w,h,sw,enforce_same_length,gt_rot,gt_trans,gt_scaling,gt_z = get_infos_gt(gt_infos,pose_config,global_config,bbox_overlap)
    B = get_pb_real_grid(w,h,f,sw,device)
    bbox = segmentation_infos["predictions"]["bbox"]
    model_path = get_model_path(retrieval_list,nn_index,gt_infos,bbox_overlap,gt=use_gt_retrieval)

    R,tilts,azims,elevs = get_R(target_folder,name,gt_infos,bbox_overlap,gt=use_gt_R)
    Ss = get_S(gt_scaling,gt=True)
    Ts,xs,ys,zs = get_Ts(f,w,h,sensor_width,global_config,bbox,gt_z,model_path,model_to_infos,gt_scaling,segmentation_infos,R,specific_Ts)
    

    gt_pose_in_limits = check_gt_pose_in_limits(xs,ys,zs,tilts,elevs,azims,gt_trans,gt_rot)
    best_T_possible,best_R_possible = get_nearest_pose_to_gt(xs,ys,zs,tilts,elevs,azims,gt_trans,gt_rot)


    # compute factors
    Ts = torch.Tensor(Ts)
    signal = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["signal"]
    if signal == 'bbox' or signal == 'lines':
        best_T_index,best_S_index,max_factor,lines_3D,lines_2D,lines_available,multiplier_lines,factors,area_accepted_all_Ts,area_threshold,angle_threshold,only_allow_single_mapping_to_3d = signal_bbox_or_line(global_config,model_path,target_folder,detection,device,R,Ts,bbox,f,w,h,sw,Ss,B,enforce_same_length,top_n_for_translation,line_dir_2d,line_indices_2d,line_indices_3d,use_gt_retrieval,use_gt_R)
        points_2D,points_available,points_3D,multiplier_points,point_threshold,use_threshold = None,None,None,None,None,None
    elif signal == 'points':
        best_T_index,max_factor,points_2D,points_available,points_3D,multiplier_points,point_threshold,use_threshold = signal_points(global_config,model_path,target_folder,detection,device,R,Ts,B,top_n_for_translation,f)
        

    T = Ts[best_T_index]
    S = Ss[best_S_index]

    print('best_index_t',best_T_index)
    n_indices = 1
    pose_information = create_pose_info_dict(np.array(R),T,n_indices,max_factor,gt_pose_in_limits,gt_rot,gt_trans,best_R_possible,best_T_possible,xs,ys,zs,tilts,elevs,azims,gt_scaling)
    
    imgs = []
    img_annotations = []
    if gt_infos["img"] in visualisation_list:
        
        closest_gt_t_index = get_closest_index(Ts,gt_trans)
        closest_gt_s_index = get_closest_index(torch.Tensor(Ss),gt_scaling)

       
        
        for which,l,s in zip(['selected','closest_gt'],[best_T_index,closest_gt_t_index],[best_S_index,closest_gt_s_index]):
            T = Ts[l]
            # if which == 'closest_gt':
            #     T = torch.Tensor(gt_trans)
            S = Ss[s]
            img,img_annotation = visualise_translation(target_folder,gt_infos,global_config,model_path,signal,lines_3D,gt_scaling,sw,device,f,bbox,S,T,R,lines_2D,B,top_n_for_translation,multiplier_lines,enforce_same_length,points_3D,points_2D,multiplier_points,point_threshold,use_threshold,lines_available,points_available,area_threshold,angle_threshold,only_allow_single_mapping_to_3d)
            imgs.append(img)
            img_annotations.append(img_annotation)

    return pose_information,imgs,Ts,factors,area_accepted_all_Ts,img_annotations


# def get_visualise_single_example(global_config,name,model_to_infos):

#     target_folder,top_n_for_translation,pose_config,sensor_width,device = get_variables(global_config)

#     retrieval = name.rsplit('_',1)[0]
#     detection = name.rsplit('_',2)[0]
#     gt_name = name.rsplit('_',3)[0]
#     nn_index = int(retrieval.split('_')[-1])
#     line_dir_2d = target_folder + '/' + 'lines_2d_filtered'

#     visualisation_list,retrieval_list,gt_infos,segmentation_infos,bbox_overlap = load_infos(target_folder,detection,gt_name)

#     if nn_index >= len(retrieval_list):
#         return None


#     f,w,h,sw,enforce_same_length,gt_rot,gt_trans,gt_scaling,gt_z = get_infos_gt(gt_infos,pose_config,global_config,bbox_overlap)
#     B = get_pb_real_grid(w,h,f,sw,device)
#     bbox = segmentation_infos["predictions"]["bbox"]
#     model_path = get_model_path(retrieval_list,nn_index,gt_infos,bbox_overlap,gt=False)

#     R,tilts,azims,elevs = get_R(target_folder,name,gt_infos,bbox_overlap)
#     Ss = get_S(gt_scaling,gt=True)
#     Ts,xs,ys,zs = get_Ts(f,w,h,sensor_width,global_config,bbox,gt_z,model_path,model_to_infos,gt_scaling,segmentation_infos,R)


#     Ts = torch.Tensor(Ts)
#     signal = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["signal"]
#     if signal == 'bbox' or signal == 'lines':
#         best_T_index,best_S_index,max_factor,lines_3D,lines_2D,lines_available,multiplier_lines,factors,area_threshold,angle_threshold = signal_bbox_or_line(global_config,model_path,target_folder,detection,device,R,Ts,bbox,f,w,h,sw,Ss,B,enforce_same_length,top_n_for_translation,line_dir_2d)
#         points_2D,points_available,points_3D,multiplier_points,point_threshold,use_threshold = None,None,None,None,None,None
#     elif signal == 'points':
#         best_T_index,max_factor,points_2D,points_available,points_3D,multiplier_points,point_threshold,use_threshold = signal_points(global_config,model_path,target_folder,detection,device,R,Ts,B,top_n_for_translation,f)
        

#     T = Ts[best_T_index]
#     S = Ss[best_S_index]
    

#     if gt_infos["img"] in visualisation_list:
        
#         closest_gt_t_index = get_closest_index(Ts,gt_trans)
#         closest_gt_s_index = get_closest_index(torch.Tensor(Ss),gt_scaling)
        
#         for which,l,s in zip(['selected','closest_gt'],[best_T_index,closest_gt_t_index],[best_S_index,closest_gt_s_index]):
#             T = Ts[l]
#             S = Ss[s]
#             img = visualise_translation(target_folder,gt_infos,global_config,model_path,signal,lines_3D,gt_scaling,sw,device,f,bbox,S,T,R,lines_2D,B,top_n_for_translation,multiplier_lines,enforce_same_length,points_3D,points_2D,multiplier_points,point_threshold,use_threshold,lines_available,points_available,area_threshold,angle_threshold)
#             cv2.imwrite(output_path.replace('/poses/','/T_lines_vis/').replace('.json','_{}.png'.format(which)),img)

#     return 
def save_img_annotation(path,img_annotation):
    text = ''
    for line in img_annotation:
        text += line + '\n'
    with open(path,'w') as f:
        f.write(text)

def get_pose_for_folder(global_config):


    target_folder = global_config["general"]["target_folder"]
    model_to_infos = get_model_to_infos()

    for name in tqdm(sorted(os.listdir(target_folder + '/poses_R'))):
        pose_information,imgs,Ts,factors,area_accepted_all_Ts,img_annotations = get_pose_single_example(global_config,name,model_to_infos)
        if pose_information == None:
            continue

        output_path = target_folder + '/poses/' + name
        with open(output_path,'w') as open_f:
            json.dump(pose_information,open_f,indent=4)

        if imgs != []:
            for which,img,img_annotation in zip(['selected','closest_gt'],imgs,img_annotations):
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
    print('Translation from lines')
    main()

