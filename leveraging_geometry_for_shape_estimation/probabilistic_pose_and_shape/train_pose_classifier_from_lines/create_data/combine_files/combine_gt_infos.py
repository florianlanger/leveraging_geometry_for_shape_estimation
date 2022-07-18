
import os
import numpy as np
from tqdm import tqdm
import torch
import json

dir_path = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/gt_infos'
save_path = dir_path + '.json'

all_files = {}
for file in tqdm(sorted(os.listdir(dir_path))):
    with open(os.path.join(dir_path, file), 'r') as f:
        data = json.load(f)
    all_files[file] = data

with open(save_path, 'w') as f:
    json.dump(all_files, f)
