
import os
import numpy as np
from tqdm import tqdm
import torch
import json
import cv2




def combine_img_dir(dir_path,save_path,type):
    flag = cv2.IMREAD_COLOR
    if type == 'depth':
        flag = cv2.IMREAD_UNCHANGED

    all_files = {}
    for file in tqdm(sorted(os.listdir(dir_path))):
        loaded = cv2.imread(os.path.join(dir_path, file),flag)
        if type == 'depth':
            loaded = loaded.astype(float) / 1000
        elif type == 'normal':
            loaded = cv2.cvtColor(loaded,cv2.COLOR_BGR2RGB)
            loaded = (- (np.array(loaded).astype(np.float32) / 255.0) + 0.5) * 2.0
                
        loaded = torch.from_numpy(loaded)
        all_files[file] = loaded

    torch.save(all_files,save_path)


if __name__ == '__main__':

    base_dir = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/'


    for type_data in ['val','train']:
        base_path = base_dir + type_data + '/'
        for type in ['normal']:
            if type == 'depth':
                dir_path = base_path + 'depth_gb_medium_160_120'
            elif type == 'normal':
                dir_path = base_path + 'norm_gb_medium_160_120'
            elif type == 'img':
                dir_path = base_path + 'images_160_120'
        
            save_path = dir_path + '.pt'

            combine_img_dir(dir_path,save_path,type)

