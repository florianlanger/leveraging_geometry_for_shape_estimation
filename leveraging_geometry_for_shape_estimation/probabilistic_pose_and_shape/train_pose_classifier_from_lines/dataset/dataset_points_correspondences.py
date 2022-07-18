from asyncio import run_coroutine_threadsafe
from xml.parsers.expat import model
from xml.sax.handler import DTDHandler
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


from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.dataset.dataset import load_numpy_dir,get_gt_infos,get_gt_infos_by_detection,bbox_to_lines,reproject_3d_lines,check_config,compute_lines_resized,Dataset_lines
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.dataset.dataset_points import Dataset_points

from probabilistic_formulation.factors.factors_T.factors_lines import lines_3d_to_pixel
from probabilistic_formulation.utilities import get_device
from probabilistic_formulation.factors.factors_T.compare_lines import sample_points_from_lines


class Dataset_points_correspondences(Dataset_points):

    def __init__(self,config,kind):
        super().__init__(config,kind)

    # def __init__(self,config,kind):
    #     'Initialization'

    #     dir_key = "dir_path_2d_" + kind
    #     self.dir_path_2d = config["data"][dir_key]
    #     print('Loading 3d lines')
    #     self.lines_3d = load_numpy_dir(config["data"]["dir_path_3d"],max_n=100000)
    #     print('Loading GT infos')
    #     self.gt_infos = get_gt_infos(self.dir_path_2d + '/gt_infos',max_n=config["training"]['max_number'])

    #     if kind == 'train':
    #         self.gt_infos_by_detection = get_gt_infos_by_detection(self.dir_path_2d + '/gt_infos',max_n=config["training"]['max_number'])
    #         print('Loading 2d lines')
    #         self.lines_2d = load_numpy_dir(self.dir_path_2d + '/lines_2d_filtered_only_object_center',max_n=config["training"]['max_number'])
    #     elif kind == 'val':
    #         print('Loading 2d lines')
    #         self.lines_2d = load_numpy_dir(self.dir_path_2d + '/lines_2d_filtered',max_n=config["training"]['max_number'])
    #         self.gt_infos_by_detection = self.get_gt_infos_by_detection_predicted()

    #     self.config = config
    #     self.index_to_names = self.get_index_to_names(kind)
    #     self.threshold_correct = config["data"]["threshold_correct_T"]
    #     self.image_size = config["data"]["img_size"]
    #     self.kind = kind
    #     self.half_width_sampling_cube = 1.

    #     self.get_modelid_to_sym(config["data"]["dir_path_scan2cad_anno"])

        # check_config(config)



    def __getitem__(self, index):

        # print('always 0 debug')
        # index = 0

        max_n_lines = 100
        detection_name,gt_name,model_3d_name = self.index_to_names[index]
        gt_infos = self.gt_infos_by_detection[detection_name]
        
        lines_2d,lines_3d = self.process_lines(detection_name,model_3d_name,gt_infos)
        
        # print('self.gt_infos_by_detection',self.gt_infos_by_detection)
        bbox = self.gt_infos_by_detection[detection_name]['bbox']
        assert len(bbox) == 4,bbox
        augmented_bbox = self.augment_bbox(bbox,self.gt_infos_by_detection[detection_name])

        R,T,correct,offset,r_correct,sym = self.get_R_and_T(detection_name,gt_name,augmented_bbox)

        # take max 100 lines for reprojected and 2d detected
        max_n_points = 2000
        empty_points = np.zeros((max_n_points,3),dtype=np.float32)

        points_base,points_reprojected = self.process_points(lines_3d[:max_n_lines],lines_2d[:max_n_lines],R,T,gt_infos,bbox,augmented_bbox)
        # points_reprojected = points_reprojected[:1,:]

        points = torch.cat([points_base,points_reprojected],dim=0).numpy()
        empty_points[:min(points_base.shape[0],1000),:] = points[:min(points_base.shape[0],1000),:]
        empty_points[1000:1000+points_reprojected.shape[0],:] = points_reprojected.numpy()

        _,points_reprojected_gt = self.process_points(lines_3d[:max_n_lines],lines_2d[:max_n_lines],gt_infos['rot_mat'],gt_infos['trans_mat'],gt_infos,bbox,augmented_bbox)
        # points_reprojected_gt = points_reprojected_gt[:1,:]
        # print('only single point')

        offsets = points_reprojected_gt - points_reprojected
        offsets_padded = np.concatenate([offsets[:,:2],np.zeros((1000-offsets.shape[0],2))]).astype(np.float32)



        extra_infos = {'detection_name': detection_name,'gt_name': gt_name,'model_3d_name': model_3d_name, 'category': model_3d_name.split('_')[0],"offset":offset,"r_correct":r_correct,"sym":sym,"index_start_reprojected": 1000,"n_reprojected": points_reprojected.shape[0]}


        return empty_points,offsets_padded,extra_infos


       