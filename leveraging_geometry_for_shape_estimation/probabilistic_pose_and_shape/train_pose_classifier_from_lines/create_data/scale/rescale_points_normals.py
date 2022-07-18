


import os
import numpy as np
from tqdm import tqdm
import torch

from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.ground_plane import get_model_to_infos_scannet


model_to_infos_scannet = get_model_to_infos_scannet()

dir_path = '/scratch/fml35/datasets/shapenet_v2/ShapeNetRenamed/representation_points_and_normals/exp_08_100_random_same_normals/points_and_normals'
save_path = dir_path + '_combined_normalised.pt'

assert os.path.exists(save_path) == False

all_files = {}
for file in tqdm(os.listdir(dir_path)[:100000]):
    if file.endswith(".npz"):
        masks = np.load(os.path.join(dir_path, file))
        for key in masks:
            # print(key)
            name = file.split('.')[0] + '_' + key
            # print(name)
            content = torch.from_numpy(masks[key])

            if key == 'points':
                factor = torch.Tensor(model_to_infos_scannet[file.split('.')[0]]['bbox']).unsqueeze(0).repeat(content.shape[0],1) * 2 
                content = content / factor

            all_files[name] = content
            # break

torch.save(all_files, save_path)
