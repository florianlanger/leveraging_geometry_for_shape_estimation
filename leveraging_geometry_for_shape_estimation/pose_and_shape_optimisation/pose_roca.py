
from curses import raw
import os
import shutil
import json
import sys
import random
from black import get_string_prefix
import imagesize
from tqdm import tqdm
from pycocotools import mask as mutils
from pycocotools import _mask as coco_mask
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as scipy_rot

from leveraging_geometry_for_shape_estimation.data_conversion_scannet.get_infos import get_scene_pose


def main():
    global_info = sys.argv[1] + '/global_information.json'
    with open(global_info,'r') as f:
        global_config = json.load(f)

    if global_config["dataset"]["which_dataset"] == 'scannet' and global_config["segmentation"]["use_gt"] == 'roca':

        with open('/scratch2/fml35/datasets/scannet/scan2cad_annotations/full_annotations.json') as json_file:
            all_data = json.load(json_file)


        target_folder = global_config["general"]["target_folder"]

        with open('/scratch2/fml35/results/ROCA/per_frame_best_no_null.json','r') as file:
            roca = json.load(file)


        for img in tqdm(roca):

            if not os.path.exists(target_folder + '/gt_infos/' + img.split('/')[0] + '-' + img.split('/')[2].split('.')[0] + '.json'):
                continue

            scene = img.split('/')[0]
            frame = img.split('/')[2].split('.')[0]


            scene_info = None
            for j in range(len(all_data)):
                if all_data[j]["id_scan"] == scene:
                    scene_info = all_data[j]

            with open(target_folder + '/gt_infos/' + scene + '-' + frame + '.json','r') as file:
                gt_infos = json.load(file)

            # with open(target_folder + '/bbox_overlap/' + name.split('.')[0] + '.json','r') as f:
            #     bbox_overlap = json.load(f)
        
    
            counter = 0
            for detection in roca[img]:

                with open(target_folder + '/bbox_overlap/' + scene + '-' + frame + '_' + str(counter).zfill(2) + '.json','r') as f:
                    bbox_overlap = json.load(f)


                new_dict = {}
                new_dict["indices"] = []

                t = detection["t"]
                q = detection["q"]

                # q = [q[3],q[0],q[1],q[2]]
                q = [q[1],q[2],q[3],q[0]]
                Rcad = scipy_rot.from_quat(q).as_matrix()


                # frame_4by4_path = global_config["dataset"]["dir_path_images"] + scene + '/pose/' + frame + '.txt'
                # scene_trs = scene_info["trs"]
                # R_scene_pose,T_scene_pose = get_scene_pose(frame_4by4_path,scene_trs)

                # R_no_scaling = np.matmul(R_scene_pose,Rcad)

                # T = np.matmul(R_scene_pose,np.array(t)) + T_scene_pose

                # # invert axes to match our convention
                invert = np.array([[-1,0,0],[0,-1,0],[0,0,1.]])
                # R_no_scaling = np.matmul(invert,R_no_scaling)
                # T = np.matmul(invert,T)

                # invert = np.array([[-1,0,0],[0,-1,0],[0,0,1.]])
                R_final = np.matmul(invert,Rcad)
                T_final = np.matmul(invert,np.array(detection["t"]))

                # R_final = Rcad
                # T_final = np.array(detection["t"])


                new_dict["predicted_r"] = R_final.tolist()
                new_dict["predicted_t"] = T_final.tolist()
                new_dict["predicted_s"] = detection["s"]
                # new_dict["predicted_r"] = gt_infos["objects"][bbox_overlap['index_gt_objects']]["rot_mat"]
                # new_dict["predicted_t"] = gt_infos["objects"][bbox_overlap['index_gt_objects']]["trans_mat"]
                # new_dict["predicted_s"] = gt_infos["objects"][bbox_overlap['index_gt_objects']]["scaling"]
                new_dict["combined"] = 1.
                new_dict["factor"] = 1.

                detection_name = img.split('/')[0] + '-' + img.split('/')[2].split('.')[0] + '_' + str(counter).zfill(2)


                # with open(target_folder + '/gt_infos/' + scene + '-' + frame + '.json','r') as file:
                #     gt_infos = json.load(file)

                # with open(target_folder + '/bbox_overlap/' + detection_name + '.json','r') as f:
                #     bbox_overlap = json.load(f)

                # print('gt',gt_infos["objects"][bbox_overlap['index_gt_objects']]["rot_mat"])
                # print('gt',gt_infos["objects"][bbox_overlap['index_gt_objects']]["trans_mat"])

                with open(target_folder + '/poses/' + detection_name + '_000_00.json','w') as file:
                    json.dump(new_dict,file)

                # cat = detection["category"].replace('bookcase','bookshelf')
                # nn_dict = {}
                # nn_dict["model"] = "model/" + cat + '/' + detection["scene_cad_id"][1] + '/model_normalized.obj'
                # nn_dict["category"] = cat
                # nn_dict["name"] = cat + '_' + detection["scene_cad_id"][1]
                
                # nn_all = {}
                # nn_all["nearest_neighbours"] = []
                # nn_all["nearest_neighbours"].append(nn_dict)

                # with open(target_folder + '/nn_infos/' + detection_name + '.json','w') as file:
                #     json.dump(nn_all,file)
                
                counter += 1
    

if __name__ == '__main__':
    main()