
import os
import numpy as np
from tqdm import tqdm
import torch

dir_path = '/scratch/fml35/datasets/shapenet_v2/ShapeNetRenamed/representation_points_and_normals/exp_08_100_random_same_normals/points_and_normals'
save_path = dir_path + '_combined.pt'

assert os.path.exists(save_path) == False

all_files = {}
for file in tqdm(os.listdir(dir_path)[:100000]):
    if file.endswith(".npz"):
        masks = np.load(os.path.join(dir_path, file))
        for key in masks:
            
            name = file.split('.')[0] + '_' + key
            all_files[name] = torch.from_numpy(masks[key])
            # break
# np.savez(save_path, **all_files)
torch.save(all_files, save_path)

# saved = np.load(save_path, allow_pickle=True)
# for key in saved:
#     print(key)
#     print(saved[key][:10])