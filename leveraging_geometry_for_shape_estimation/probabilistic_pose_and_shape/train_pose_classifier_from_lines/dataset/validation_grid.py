from xml.parsers.expat import model
import torch
from torch.utils import data
import numpy as np
import json
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import imageio
import os
import random
import cv2
from scipy.spatial.transform import Rotation as scipy_rot

from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.dataset.dataset import load_numpy_dir,get_gt_infos,get_gt_infos_by_detection,check_config,reproject_3d_lines, Dataset_lines

from probabilistic_formulation.factors.factors_T.factors_lines import lines_3d_to_pixel



class Dataset_val_grid(Dataset_lines):
    def __init__(self,config,kind):
        'Initialization'

        dir_key = "dir_path_2d_" + kind
        self.dir_path_2d = config["data"][dir_key]

        print('Loading 3d lines')
        self.lines_3d = load_numpy_dir(config["data"]["dir_path_3d_lines"],max_n=100000)
        print('Loading GT infos')
        self.gt_infos = get_gt_infos(self.dir_path_2d + '/gt_infos',max_n=config["training"]['max_number'])

        if kind == 'train':
            self.gt_infos_by_detection = get_gt_infos_by_detection(self.dir_path_2d + '/gt_infos',max_n=config["training"]['max_number'])
            print('Loading 2d lines')
            self.lines_2d = load_numpy_dir(self.dir_path_2d + '/lines_2d_filtered_only_object_center',max_n=config["training"]['max_number'])
        elif kind == 'val':
            print('Loading 2d lines')
            self.lines_2d = load_numpy_dir(self.dir_path_2d + '/lines_2d_filtered',max_n=config["training"]['max_number'])
            self.gt_infos_by_detection = self.get_gt_infos_by_detection_predicted()

        self.config = config
        self.index_to_names = self.get_index_to_names(kind)
        self.threshold_correct = config["data"]["threshold_correct_T"]
        self.image_size = config["data"]["img_size"]
        self.kind = kind
        self.grid_val_T = self.get_grid(config["training"]["val_grid_points_per_example"])

        check_config(config)



    def __len__(self):
        return len(self.index_to_names) * self.config["training"]["val_grid_points_per_example"]

    def get_grid(self,n_points):
        Ts = np.array([[1,1,1],[1,-1,1],[-1,1,1],[-1,-1,1],[1,1,-1],[1,-1,-1],[-1,1,-1],[-1,-1,-1],
                       [3,3,3],[3,-3,3],[-3,3,3],[-3,-3,3],[3,3,-3],[3,-3,-3],[-3,3,-3],[-3,-3,-3],
                       [3,0,0],[-3,0,0],[0,3,0],[0,-3,0],[0,0,3],[0,0,-3]])
        Ts = Ts * self.config["data"]["threshold_correct_T"] * 0.5
        assert n_points == len(Ts)
        return Ts


    def get_T(self,gt_infos_detection,index_T):

    
        gt_T = gt_infos_detection["trans_mat"]

        offset = self.grid_val_T[index_T]
        T = gt_T + offset
        correct = np.linalg.norm(T - gt_infos_detection["trans_mat"]) < self.threshold_correct
        return T,correct * np.array([1]).astype(np.float32),offset

    def __getitem__(self, index_total):

        index = index_total // self.config["training"]["val_grid_points_per_example"]
        index_T = index_total % self.config["training"]["val_grid_points_per_example"]

        detection_name,gt_name,model_3d_name = self.index_to_names[index]
        
        lines_2d = self.lines_2d[detection_name]
        lines_2d = lines_2d[:,[1,0,3,2]]

        lines_3d = self.lines_3d[model_3d_name]

        lines_3d = self.mask_random(lines_3d,self.config["data_augmentation"]["percentage_lines_3d"])
        lines_2d = self.mask_random(lines_2d,self.config["data_augmentation"]["percentage_lines_2d"])

        assert lines_2d.shape[0] != 0,lines_2d
        assert lines_3d.shape[0] != 0,lines_3d
        
        # print('self.gt_infos_by_detection',self.gt_infos_by_detection)
        bbox = self.gt_infos_by_detection[detection_name]['bbox']
        assert len(bbox) == 4,bbox
        augmented_bbox = self.augment_bbox(bbox,self.gt_infos_by_detection[detection_name])
        R = self.gt_infos_by_detection[detection_name]['rot_mat']
        R = self.augment_R(R)
        gt_infos = self.gt_infos_by_detection[detection_name]

        lines_3d = self.convert_lines_3d(lines_3d,gt_infos['scaling'])

        T,correct,offset = self.get_T(gt_infos,index_T)


        img = self.process_image(lines_3d,lines_2d,R,T,gt_infos,bbox,augmented_bbox)
        extra_infos = {'detection_name': detection_name,'gt_name': gt_name,'model_3d_name': model_3d_name, 'category': model_3d_name.split('_')[0],"offset":offset}
            
        return img,correct,extra_infos


if __name__ == '__main__':
    config_path = '/home/mifs/fml35/code/shape/leveraging_geometry_for_shape_estimation/probabilistic_pose_and_shape/train_pose_classifier_from_lines/config.json'
    with open(config_path,'r') as file:
        config = json.load(file)


    dataset = Dataset_lines(config)

    out_path = '/scratch2/fml35/experiments/classify_T_from_lines/debug/'
    for i in range(3):
        img,correct = dataset[i]
        cv2.imwrite(out_path + str(i).zfill(2) + '.png',img)

       