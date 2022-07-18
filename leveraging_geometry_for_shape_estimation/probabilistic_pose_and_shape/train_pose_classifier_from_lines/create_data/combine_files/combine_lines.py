
import os
import numpy as np
from tqdm import tqdm
import torch
import json

dir_path = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/canny_points_480_360_n_points_5000'
save_path = dir_path + '.pt'

all_files = {}
for file in tqdm(sorted(os.listdir(dir_path))):
    data = np.load(os.path.join(dir_path, file), allow_pickle=True)
    all_files[file] = torch.from_numpy(data)

torch.save(all_files, save_path)