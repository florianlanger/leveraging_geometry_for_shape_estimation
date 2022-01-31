import cv2
import numpy as np
import sys
import os
import json
from tqdm import tqdm
import itertools

from translation_optimisation_given_rotation import single_poses_loop

def create_all_2_indices(n_matches):

    n_keypoints_pose = min(n_matches,2)
    if n_matches < 2:
        all_2_indices = [list(range(n_keypoints_pose))]
    else:
        all_2_indices = []
        for combo in itertools.combinations(range(n_matches), n_keypoints_pose):
            all_2_indices.append(list(combo))
        np.random.shuffle(all_2_indices)

    return all_2_indices,n_keypoints_pose


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

def create_pose_information(indices,R,T,reprojection_dists,T_history):
    pose_information = {}
    pose_information["indices"] = indices.tolist()
    pose_information["predicted_r"] = R.tolist()
    pose_information["predicted_t"] = T.tolist()
    pose_information["avg_dist_reprojected_keypoints"] = reprojection_dists
    pose_information["t_history"] = T_history.tolist()
    return pose_information


def get_pose_for_folder(global_config):

    target_folder = global_config["general"]["target_folder"]
    top_n_retrieval = global_config["keypoints"]["matching"]["top_n_retrieval"]

    for name in tqdm(os.listdir(target_folder + '/cropped_and_masked')):
        
        with open(target_folder + '/gt_infos/' + name.rsplit('_',1)[0] + '.json','r') as file:
            gt_infos = json.load(file)
        f = gt_infos["focal_length"]
        w,h = gt_infos["img_size"]

        
        for i in range(top_n_retrieval):
            with open(target_folder + '/matches_orig_img_size/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.json','r') as file:
                matches_orig_img_size = json.load(file)

            with open(target_folder + '/poses_R/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.json','r') as file:
                R = json.load(file)["predicted_r"]
            R = np.array(R)

            wc_matches_all = np.load(target_folder + '/wc_matches/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.npy','r')
            pixels_real_all = np.array(matches_orig_img_size["pixels_real_orig_size"])

            indices = np.array(range(0,wc_matches_all.shape[0]))
            # e.g. indices = [0,1,2,3,4], indices_all = [0,2,4] all_4_indices will now only be from 0 to 2 [0,1,2] and at the end need to map back to indices_all i.e. 0 -> 0, 1 -> 2, 2 -> 4
            wc_matches_all,pixels_real_all,indices_all = remove_world_coordinates(wc_matches_all,pixels_real_all,indices)

            n_matches = wc_matches_all.shape[0]
            all_4_indices,n_keypoints_pose = create_all_2_indices(n_matches)

            all_predicted_t,all_reprojection_dists,all_indices,all_param_history = single_poses_loop(global_config,all_4_indices,wc_matches_all,pixels_real_all,w,h,f,gt_infos,n_keypoints_pose,R)
            
            # save best
            index_best = np.argmin(all_reprojection_dists)
            orig_indices = indices_all[all_indices[index_best]]
            pose_information = create_pose_information(orig_indices,R,all_predicted_t[index_best],all_reprojection_dists[index_best],all_param_history[index_best])

            with open(target_folder + '/poses/' + name.split('.')[0] + '_' + str(i).zfill(3) + '.json','w') as file:
                json.dump(pose_information,file)





def main():
    np.random.seed(1)
    print('Should not use gt bbox in initialisation!!!')

    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    if global_config["pose_and_shape_probabilistic"]["use_probabilistic"] == "True":
        get_pose_for_folder(global_config)
    

    



if __name__ == '__main__':
    main()

