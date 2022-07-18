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

from probabilistic_formulation.factors.factors_T.factors_lines import lines_3d_to_pixel
from probabilistic_formulation.utilities import get_device
from probabilistic_formulation.factors.factors_T.compare_lines import sample_points_from_lines


def create_base_points(config,target_size,n_points_per_line,lines_2d=None,bbox=None,device=torch.device('cpu')):

    points = []

    if config["data"]["use_bbox"] == True:
        # img = draw_boxes(img,np.array([bbox]),thickness=1,color=(0,0,255))
        bbox_lines = bbox_to_lines(bbox)
        points.append(get_points_3d_torch(bbox_lines.to(device),target_size,bbox_lines.shape[0],channel=1,n_samples=n_points_per_line).squeeze(0))

    if config["data"]["use_lines_2d"] == True:
        # img = plot_matches_individual_color_no_endpoints(img,lines_2d[:,:2],lines_2d[:,2:4],line_colors=[0,255,0],thickness=1)
        points.append(get_points_3d_torch(lines_2d.to(device),target_size,lines_2d.shape[0],channel=0,n_samples=n_points_per_line).squeeze(0))
    
    points = torch.cat(points,dim=0)
    return points


def get_points_3d_torch(lines_2d_batched,base_img_shape,n_lines_2d,samples_per_pixel=1,channel=0,n_samples=10):
        assert lines_2d_batched.shape[1] == 4 and len(lines_2d_batched.shape) == 2, lines_2d_batched.shape
    
        assert lines_2d_batched.shape[0] % n_lines_2d == 0, (lines_2d_batched,n_lines_2d)
        t1 = time.time()
        n_images = lines_2d_batched.shape[0] // n_lines_2d

        device = get_device(lines_2d_batched)

        # max_length_lines = torch.max(torch.linalg.norm(lines_2d_batched[:,2:4] - lines_2d_batched[:,0:2],dim=1))
        # max_length_lines = np.linalg.norm(base_img_shape[0:2])
        # n_samples = int(np.round(max_length_lines.item() * samples_per_pixel))

        t2 = time.time()
        points = sample_points_from_lines(lines_2d_batched,n_samples)
        # points = torch.round(points).long()

        assert points.shape == (lines_2d_batched.shape[0],n_samples,2), (points.shape,[lines_2d_batched.shape[0],n_samples,2])
        t3 = time.time()

        channel_index = channel
        channel_points = torch.ones((points.shape[0],n_samples,1),dtype=torch.float32).to(device) * channel_index
        points = torch.cat([points,channel_points],dim=2)        

        # points = points.view(lines_2d_batched.shape[0]*n_samples,2)

        points = points.view(n_images,n_lines_2d*n_samples,3)

        assert points.shape == (n_images,n_lines_2d*n_samples,3), (n_images,n_lines_2d*n_samples,3)

        # normalise_points
        reshaped_size = torch.Tensor([base_img_shape[0],base_img_shape[1],1]).to(device).repeat((n_images,n_lines_2d*n_samples,1))

        points = points / reshaped_size

        return points


class Dataset_points(Dataset_lines):


    def __init__(self,config,kind):
        super().__init__(config,kind)


    def create_points(self,lines_2d,lines_3d_reprojected,augmented_bbox,bbox,orig_img_size,target_size,n_points_per_line):
        t1 = time.time()
        lines_2d,lines_3d_reprojected,augmented_bbox,bbox = compute_lines_resized([lines_2d,lines_3d_reprojected,augmented_bbox,bbox],orig_img_size,target_size)
        lines_3d_reprojected = torch.from_numpy(lines_3d_reprojected)
        t2 = time.time()
        points_base = create_base_points(self.config,target_size,n_points_per_line,lines_2d=lines_2d,bbox=augmented_bbox,device=self.device)
        t3 = time.time()
        points_reprojected = get_points_3d_torch(lines_3d_reprojected.to(self.device),target_size,lines_3d_reprojected.shape[0],channel=2,n_samples=n_points_per_line).squeeze(0)
        t4 = time.time()
        # print('create_points',t2-t1,t3-t2,t4-t3)
        return points_base,points_reprojected

   

    def process_points(self,lines_3d,lines_2d,R,T,gt_infos,bbox,augmented_bbox,n_points_per_line):
        t1 = time.time()
        lines_3d_reprojected = reproject_3d_lines(lines_3d,R,T,gt_infos,self.config)
        t2 = time.time()
        points_base,points_reprojected = self.create_points(lines_2d,lines_3d_reprojected,augmented_bbox,bbox,gt_infos["img_size"],self.config["data"]["img_size"],n_points_per_line)
        t3 = time.time()
        # print('process points',t2-t1,t3-t2)
        return points_base,points_reprojected

    def create_info_RST(self,R,S,T):
        info = torch.zeros((4,3))
        R = scipy_rot.from_matrix(R).as_quat()
        info[0,:] = torch.tensor(R[:3])
        info[1,0] = R[3]
        info[2,:] = torch.tensor(S)
        info[3,:] = torch.tensor(T)
        return info

    def __getitem__(self, index):
        t1 = time.time()
        detection_name,gt_name,model_3d_name = self.index_to_names[index]
        gt_infos = self.gt_infos_by_detection[detection_name]
        
        t2 = time.time()
        lines_2d,lines_3d = self.process_lines(detection_name,model_3d_name)
        t3 = time.time()
        # print('self.gt_infos_by_detection',self.gt_infos_by_detection)
        bbox = self.gt_infos_by_detection[detection_name]['bbox']
        assert len(bbox) == 4,bbox
        augmented_bbox = self.augment_bbox(bbox,self.gt_infos_by_detection[detection_name])

        R,T,S,correct,offset_t,offset_s,r_correct,sym,S_normalised = self.get_R_and_T_and_S(detection_name,gt_name,augmented_bbox)
        lines_3d = lines_3d * torch.Tensor(S).repeat(2)
        t4 = time.time()
        # 4 for bbox
        max_n_points = (self.config['data']['n_lines_3d'] + self.config['data']['n_lines_2d'] + 4) * self.config['data']['n_points_per_line']
        padded_points = torch.zeros((max_n_points,3),dtype=torch.float32).to(self.device)

        points_base,points_reprojected = self.process_points(lines_3d[:self.config['data']['n_lines_3d']],lines_2d[:self.config['data']['n_lines_2d']],R,T,gt_infos,bbox,augmented_bbox,self.config['data']['n_points_per_line'])
        t5 = time.time()

        if self.config["data"]["input_RST"] == True:
            info_RST = self.create_info_RST(R,S_normalised,T).to(self.device)
            list_info = [info_RST,points_base,points_reprojected]
        elif self.config["data"]["input_RST"] == False:
            list_info = [points_base,points_reprojected]

        points = torch.cat([points_base,points_reprojected],dim=0)#.numpy()
        padded_points[:min(points.shape[0],max_n_points),:] = points[:min(points.shape[0],max_n_points),:]


        extra_infos = {'detection_name': detection_name,'gt_name': gt_name,'model_3d_name': model_3d_name, 'category': model_3d_name.split('_')[0],"offset_t":offset_t,"offset_s":offset_s,"r_correct":r_correct,"sym":sym}
        
        # print('time',t2-t1,t3-t2,t4-t3,t5-t4)
        # print('t5-t1',t5-t1)

        return padded_points,correct,extra_infos


       