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


from pose_selection import create_all_4_indices,sample_points_inside_segmentation,compute_selection_metric_shape,get_points_from_predicted_mesh_shape
from pose_estimation import estimate_pose
from utilities import add_minimal_pose_info_dict_shape,get_information_best_pose
from shape_loop import single_poses_loop

def convert_pixel_to_bearings(pixels,f,w,h,sensor_width):
    """Input of pixel is (py,px), Output pixel bearing are in (x,y,z)"""
    bearings = np.zeros((pixels.shape[0],3))

    x = -(pixels[:,1] - w/2.) * sensor_width / w
    y = -(pixels[:,0] - h/2.) * sensor_width / w
    z = x * 0 + f
    bearings = np.stack([x,y,z],axis=1)

    # normalise as opengl expects normalised pb
    bearings = bearings / np.repeat(np.expand_dims(np.sum(bearings**2,axis=1)**0.5,1),3,axis=1)

    intrinsic_camera_matrix = np.array([[-f*w/sensor_width,0,w/2],[0,-f*w/sensor_width,h/2],[0,0,1]])
    return bearings,intrinsic_camera_matrix


def remove_world_coordinates(wc_matches,pixels_real,indices):
    mask = (wc_matches > -100).all(axis=1)
    wc_matches = wc_matches[mask]
    pixels_real = pixels_real[mask]
    indices = indices[mask]
    return wc_matches,pixels_real,indices

def get_pred_t_gt_z(predicted_t,T_gt,device):
    T_gt = torch.Tensor(T_gt).to(device)
    predicted_t[:,0,2] = T_gt[2]
    return predicted_t


def get_pose_for_folder(global_config):

    target_folder = global_config["general"]["target_folder"]
    models_folder_read = global_config["general"]["models_folder_read"]
    top_n_retrieval = global_config["keypoints"]["matching"]["top_n_retrieval"]


    pose_config = global_config["pose_and_shape"]["pose"]

    device = torch.device("cuda:{}".format(global_config["general"]["gpu"]))
    torch.cuda.set_device(device)

    for name in tqdm(os.listdir(target_folder + '/cropped_and_masked')):
        
        with open(target_folder + '/nn_infos/' + name.split('.')[0] + '.json','r') as f:
            retrieval_list = json.load(f)["nearest_neighbours"]

        with open(target_folder + '/gt_infos/' + name.rsplit('_',1)[0] + '.json','r') as f:
            gt_infos = json.load(f)

        
        for i in range(top_n_retrieval):
        # for i in range(1):
            retrieval_infos = retrieval_list[i]
            elev = retrieval_list[i]["elev"]
            azim = retrieval_list[i]["azim"]
            model_path = models_folder_read + "/models/remeshed/" + retrieval_list[i]["model"].replace('model/','')

            with open(target_folder + '/matches_orig_img_size/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.json','r') as f:
                matches_orig_img_size = json.load(f)

            wc_matches_all = np.load(target_folder + '/wc_matches/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.npy','r')
            pixels_real_all = np.array(matches_orig_img_size["pixels_real_orig_size"])

            indices = np.array(range(0,wc_matches_all.shape[0]))
            wc_matches_all,pixels_real_all,indices_all = remove_world_coordinates(wc_matches_all,pixels_real_all,indices)

            correct_matches = None
            n_matches = wc_matches_all.shape[0]
            all_4_indices,n_keypoints_pose = create_all_4_indices(n_matches,pose_config,correct_matches)



            # convert pixel to pixel bearing
            f = gt_infos["focal_length"]
            w = gt_infos["img_size"][0]
            h = gt_infos["img_size"][1]
            sensor_width = pose_config["sensor_width"]
            real_bearings_all,intrinsic_camera_matrix = convert_pixel_to_bearings(pixels_real_all,f,w,h,sensor_width)

        

            predicted_obj = load_obj(model_path, device=device,create_texture_atlas=False, load_textures=False)
            segmentation_mask = cv2.imread(target_folder + '/segmentation_masks/' + name.rsplit('.')[0] + '.png')
            pixel_inside_segmentation = sample_points_inside_segmentation(segmentation_mask,pose_config["n_points_finding_best"],device)


            all_pose_information = []
            planes = np.array(global_config["pose_and_shape"]["shape"]["planes"])

            output = single_poses_loop(global_config,all_4_indices,wc_matches_all,pixels_real_all,real_bearings_all,indices_all,w,h,f,device,planes,gt_infos,retrieval_infos,n_keypoints_pose)

            all_predicted_r,all_predicted_t,all_predicted_stretching,all_world_coordinates_matches_for_poses,all_pixels_real_original_image_for_poses,all_indices,all_param_history = output[0],output[1],output[2],output[3],output[4],output[5],output[6]
            
            
            planes = torch.Tensor(planes).to(device)
            bs = pose_config["batch_size"]
            for idx in range(ceil(all_predicted_r.shape[0]/bs)):
                predicted_r = all_predicted_r[idx*bs:(idx+1)*bs]
                predicted_t = all_predicted_t[idx*bs:(idx+1)*bs]
                predicted_stretching = all_predicted_stretching[idx*bs:(idx+1)*bs]
                
                if pose_config["use_gt_z"] == "True":
                    T_gt = gt_infos["trans_mat"]
                    predicted_t = get_pred_t_gt_z(predicted_t,T_gt,device)


                indices_4 = all_indices[idx*bs:(idx+1)*bs]


                # find criteria
                world_coordinates_batch = all_world_coordinates_matches_for_poses[idx*bs:(idx+1)*bs]
                pixels_real_original_image_batch = all_pixels_real_original_image_for_poses[idx*bs:(idx+1)*bs]


                print('Used to load decimated meshes here to make faster')
                points_from_predicted_mesh = get_points_from_predicted_mesh_shape(predicted_obj,elev,azim,pose_config["n_points_finding_best"],device,predicted_stretching,planes)

                avg_dist_pointclouds,avg_dist_furthest,avg_dist_reprojected_keypoints,combined = compute_selection_metric_shape(predicted_r,predicted_t,predicted_stretching,planes,points_from_predicted_mesh,f,w,h,global_config["pose_and_shape"]["pose"],segmentation_mask,pixel_inside_segmentation,device,world_coordinates_batch,pixels_real_original_image_batch)

                for l in range(predicted_r.shape[0]):
                    F1_score = None
                    pose_information = add_minimal_pose_info_dict_shape(indices_4[l],predicted_r[l],predicted_t[l],predicted_stretching[l],avg_dist_pointclouds[l],avg_dist_furthest[l],avg_dist_reprojected_keypoints[l],combined[l],F1_score)
                    all_pose_information.append(pose_information)

            

            n_poses_evaluate = min(pose_config["number_visualisations_per_object"],len(all_pose_information))
            # setting_to_metric = {'segmentation': 'avg_dist_furthest', 'keypoints': 'avg_dist_reprojected_keypoints', 'combined':'combined', 'F1':'F1'}
            print('use avg dist poinrtclouds')
            setting_to_metric = {'segmentation': 'avg_dist_pointclouds', 'keypoints': 'avg_dist_reprojected_keypoints', 'combined':'combined', 'F1':'F1'}
            setting_to_min_max = {'segmentation': 'min', 'keypoints': 'min', 'combined':'min', 'F1':'max'}
            metric = setting_to_metric[pose_config["choose_best_based_on"]]
            min_max = setting_to_min_max[pose_config["choose_best_based_on"]]


            metrics = [metric] * pose_config["number_visualisations_per_object"]
            min_maxs = [min_max] * pose_config["number_visualisations_per_object"]
            indices = range(1,pose_config["number_visualisations_per_object"]+1)

            for which_metric,max_or_min,index_pose in zip(metrics[:n_poses_evaluate],min_maxs[:n_poses_evaluate],indices[:n_poses_evaluate]):
                pose_information = get_information_best_pose(all_pose_information,which_metric,max_or_min,index_pose)
                # get original indices as those are different to the ones used in loop, in original some wc were removed

                print(indices_all)
                print(pose_information["indices"])
                pose_information["indices"] = indices_all[pose_information["indices"]].tolist()

                with open(target_folder + '/poses/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.json','w') as f:
                    json.dump(pose_information,f)





def main():
    np.random.seed(1)
    torch.manual_seed(0)

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    if global_config["pose_and_shape_probabilistic"]["use_probabilistic"] == "False":
        if global_config["pose_and_shape"]["shape"]["optimise_shape"] == "True":
            get_pose_for_folder(global_config)
    

    



if __name__ == '__main__':
    main()

