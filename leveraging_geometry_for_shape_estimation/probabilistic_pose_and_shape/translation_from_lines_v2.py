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
from probabilistic_formulation.utilities import create_all_possible_combinations,get_uvuv_p_from_superpoint,create_all_possible_combinations_uvuv_p_together
from probabilistic_formulation.factors import get_factor_bbox_multiple_T, get_factor_reproject_lines_multiple_T,get_factor_reproject_kp_multiple_T, get_factor_reproject_lines_multiple_T_with_Scale
from probabilistic_formulation.tests.test_reproject_lines import load_lines_2D,load_lines_3D,get_cuboid_line_dirs_3D,plot_lines_T,plot_bbox, plot_points_T



def get_variables(global_config):

    target_folder = global_config["general"]["target_folder"]
    models_folder_read = global_config["general"]["models_folder_read"]
    top_n_retrieval = global_config["keypoints"]["matching"]["top_n_retrieval"]
    top_n_for_translation = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["top_n_for_translation"]
    pose_config = global_config["pose_and_shape"]["pose"]
    sensor_width = pose_config["sensor_width"]
    device = torch.device("cuda:{}".format(global_config["general"]["gpu"]))
    torch.cuda.set_device(device)

    return target_folder,models_folder_read,top_n_retrieval,top_n_for_translation,pose_config,sensor_width,device


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


def signal_bbox_or_line(global_config,model_path,target_folder,detection,device,R,Ts,bbox,f,w,h,sw,Ss,B,enforce_same_length,top_n_for_translation,line_indices=None):
    path =  global_config["pose_and_shape_probabilistic"]["reproject_lines"]["line_dir_3d_precise"] + "/" + model_path.split('/')[1] + '_' + model_path.split('/')[2] + '.npy'
    if not os.path.exists(path):
        path = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["line_dir_3d_self_scanned"] + '/' + model_path.split('/')[1] + '_' + model_path.split('/')[2] + '.npy'

    if os.path.exists(path):
        lines_3D = load_lines_3D(path)
    else:
        lines_3D = np.array([[0.,0.,0,0.5,0.,0]]).astype(np.float32)

    print('USE SCALIGN for mask 3d lines')
    scaling = list(Ss[0])
    # lines_3D_just_for_scaling = lines_3D * np.array(scaling + scaling).astype(np.float32)
    # mask_length = np.sum((lines_3D_just_for_scaling[:,:3] - lines_3D_just_for_scaling[:,3:6])**2,axis=1)**0.5 > global_config["pose_and_shape_probabilistic"]["reproject_lines"]["min_length_line"]
    # lines_3D = lines_3D[mask_length]


    lines_3D = lines_3D * np.array(scaling + scaling).astype(np.float32)
    mask_length = np.sum((lines_3D[:,:3] - lines_3D[:,3:6])**2,axis=1)**0.5 > global_config["pose_and_shape_probabilistic"]["reproject_lines"]["min_length_line"]
    lines_3D = lines_3D[mask_length]
    
    line_dirs_3D = torch.from_numpy(lines_3D[:,:3] - lines_3D[:,3:6])
    line_dirs_3D = line_dirs_3D / torch.linalg.norm(line_dirs_3D,axis=1).unsqueeze(1).tile(1,3)
    # because lines saved as points but need as point + direction 
    lines_3D[:,3:6] = lines_3D[:,3:6] - lines_3D[:,:3]


    line_path = target_folder + '/lines_2d_filtered/' + detection + '.npy'
    lines_2D = torch.Tensor(np.load(line_path)).long()
    if line_indices != None:
        lines_2D = lines_2D[line_indices,:]

    if len(lines_2D.shape) == 1 or lines_3D.shape[0] == 0:
        best_T_index = 0
        best_S_index = 0
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
            print('use old function without scale')
            factors_batch,_,_ = get_factor_reproject_lines_multiple_T(torch.Tensor(R).to(device),torch.Tensor(Ts).to(device),torch.from_numpy(lines_3D).to(device),lines_2D.to(device),B.to(device),top_n_for_translation,f,multiplier_lines,enforce_same_length)
            # factors_batch,_ = get_factor_reproject_lines_multiple_T_with_Scale(torch.Tensor(R).to(device),torch.Tensor(Ts).to(device),torch.Tensor(Ss).to(device),torch.from_numpy(lines_3D).to(device),lines_2D.to(device),B.to(device),top_n_for_translation,f,multiplier_lines,enforce_same_length)
        factors = factors_batch.cpu().tolist()
        best_index_overall = factors.index(max(factors))
        best_T_index = best_index_overall // Ss.shape[0]
        best_S_index = best_index_overall % Ss.shape[0]
        max_factor = max(factors)

    return best_T_index,best_S_index,max_factor,lines_3D,lines_2D,lines_available,multiplier_lines,factors

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


def visualise_translation(target_folder,gt_infos,global_config,model_path,signal,lines_3D,gt_scaling,sw,device,f,bbox,S,T,R,lines_2D,B,top_n_for_translation,multiplier_lines,enforce_same_length,points_3D,points_2D,multiplier_points,point_threshold,use_threshold,lines_available,points_available,gt_trans,output_path,which):

    # save_path = output_path.replace('/poses/','/factors_T/').replace('.json','.npz')
    # np.savez(save_path,factors=factors,Rs=R,Ts=Ts)

    img_path = target_folder + '/images/' + gt_infos["img"]
    full_model_path = global_config["dataset"]["dir_path"] + model_path

    if signal == 'bbox':
        if not lines_3D.shape[0] == 0:
            img = plot_bbox(torch.Tensor(R).unsqueeze(0),torch.Tensor(T).unsqueeze(0).unsqueeze(0),gt_scaling,full_model_path,img_path,sw,device,lines_3D,f,torch.Tensor(bbox))
    elif signal == 'lines':
        if lines_available:
            img = plot_lines_T(torch.Tensor(R).unsqueeze(0),torch.Tensor(T).unsqueeze(0).unsqueeze(0),S,full_model_path,img_path,sw,device,lines_3D,lines_2D,B,f,top_n_for_translation,multiplier_lines,enforce_same_length)
        else:
            img = cv2.imread(target_folder + '/images/' + gt_infos["img"])
    elif signal == 'points':
        if points_available:
            img = plot_points_T(torch.Tensor(R).unsqueeze(0),torch.Tensor(T).unsqueeze(0).unsqueeze(0),gt_scaling,full_model_path,img_path,sw,device,points_3D,points_2D,B,f,top_n_for_translation,multiplier_points,point_threshold,use_threshold)
        else:
            img = cv2.imread(target_folder + '/images/' + gt_infos["img"])

    cv2.putText(img, 'gt' + str(np.round(gt_trans,3)), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 128, 0), 1, cv2.LINE_AA)
    cv2.putText(img, str(np.round(T.numpy(),3)), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 128, 0), 1, cv2.LINE_AA)
    return img

def get_pose_for_folder(global_config):
    target_folder,models_folder_read,top_n_retrieval,top_n_for_translation,pose_config,sensor_width,device = get_variables(global_config)
    print('dont multuiply by scale here anymore')
    model_to_infos = get_model_to_infos()
    for name in tqdm(os.listdir(target_folder + '/poses_R')):

            retrieval = name.rsplit('_',1)[0]
            detection = name.rsplit('_',2)[0]
            gt_name = name.rsplit('_',3)[0]
            nn_index = int(retrieval.split('_')[-1])

            visualisation_list,retrieval_list,gt_infos,segmentation_infos,bbox_overlap = load_infos(target_folder,detection,gt_name)

            if nn_index >= len(retrieval_list):
                continue
        
            output_path = target_folder + '/poses/' + name

            # if os.path.exists(output_path):
            #     continue
            # print('USE GT RETRIEVAL OTHERWISE PROBLEM WITH SCALE')
            # retrieval_list[nn_index]["model"] = gt_infos["objects"][bbox_overlap['index_gt_objects']]["model"]

            f,w,h,sw,enforce_same_length,gt_rot,gt_trans,gt_scaling,gt_z = get_infos_gt(gt_infos,pose_config,global_config,bbox_overlap)
            B = get_pb_real_grid(w,h,f,sw,device)
            bbox = segmentation_infos["predictions"]["bbox"]
            model_path = retrieval_list[nn_index]["model"]

            # GET PREDICTED R
            # R  = gt_infos["objects"][bbox_overlap['index_gt_objects']]["rot_mat"]
            with open(target_folder + '/poses_R/' + name,'r') as file:
                R = json.load(file)["predicted_r"]

            sp_rot = scipy_rot.from_matrix(R)
            tilt,azim,elev = sp_rot.as_euler('zyx',degrees=True)
            tilts,azims,elevs = [tilt-0.0001,tilt+0.0001,1],[azim-0.0001,azim+0.0001,1],[elev-0.0001,elev+0.0001,1]

            # SAMPLE SS
            # Ss = np.concatenate((0.5*np.reshape(np.array(scaling),(1,3)),np.reshape(np.array(scaling),(1,3))))
            # Ss = init_Ts((0.5,3,8),(0.5,3,8),(0.5,3,8))
            Ss = np.reshape(np.array(gt_scaling),(1,3))

            
            # SAMPLES TS
            xs,ys,zs = get_T_limits(f,[w,h],sensor_width,global_config["pose_and_shape_probabilistic"]["pose"],bbox,gt_z)
            if global_config["pose_and_shape_probabilistic"]["sample_in_ground_plane"] == "True":
                Ts = get_Ts_ground_plane(model_path,model_to_infos,gt_scaling,global_config,segmentation_infos,f,sw,w,h,R)
                
            elif global_config["pose_and_shape_probabilistic"]["sample_in_ground_plane"] == "False":
                Ts = init_Ts(xs,ys,zs)

            if Ts.shape[0] == 0:
                Ts = np.array([[0,0,1]])
            

            gt_pose_in_limits = check_gt_pose_in_limits(xs,ys,zs,tilts,elevs,azims,gt_trans,gt_rot)
            best_T_possible,best_R_possible = get_nearest_pose_to_gt(xs,ys,zs,tilts,elevs,azims,gt_trans,gt_rot)


            # compute factors
            Ts = torch.Tensor(Ts)
            signal = global_config["pose_and_shape_probabilistic"]["reproject_lines"]["signal"]
            if signal == 'bbox' or signal == 'lines':
                best_T_index,best_S_index,max_factor,lines_3D,lines_2D,lines_available,multiplier_lines = signal_bbox_or_line(global_config,model_path,target_folder,detection,device,R,Ts,bbox,f,w,h,sw,Ss,B,enforce_same_length,top_n_for_translation)

            elif signal == 'points':
                best_T_index,max_factor,points_2D,points_available,points_3D,multiplier_points,point_threshold,use_threshold = signal_points(global_config,model_path,target_folder,detection,device,R,Ts,B,top_n_for_translation,f)
                

            T = Ts[best_T_index]
            S = Ss[best_S_index]
            n_indices = 1
            pose_information = create_pose_info_dict(np.array(R),T,n_indices,max_factor,gt_pose_in_limits,gt_rot,gt_trans,best_R_possible,best_T_possible,xs,ys,zs,tilts,elevs,azims,gt_scaling)
            with open(output_path,'w') as open_f:
                json.dump(pose_information,open_f,indent=4)

            if gt_infos["img"] in visualisation_list:
                
                closest_gt_t_index = get_closest_index(Ts,gt_trans)
                closest_gt_s_index = get_closest_index(Ss,gt_scaling)
                
                for which,l,s in zip(['selected','closest_gt'],[best_T_index,closest_gt_t_index],[best_S_index,closest_gt_s_index]):
                    T = Ts[l]
                    S = Ss[s]
                    img = visualise_translation(target_folder,gt_infos,global_config,model_path,signal,lines_3D,gt_scaling,sw,device,f,bbox,S,T,R,lines_2D,B,top_n_for_translation,multiplier_lines,enforce_same_length,points_3D,points_2D,multiplier_points,point_threshold,use_threshold,lines_available,points_available,gt_trans,output_path,which)
                    cv2.imwrite(output_path.replace('/poses/','/T_lines_vis/').replace('.json','_{}.png'.format(which)),img)


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

