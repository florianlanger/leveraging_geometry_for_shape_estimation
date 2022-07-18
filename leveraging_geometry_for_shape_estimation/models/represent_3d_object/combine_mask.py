
import os
import numpy as np
from tqdm import tqdm
import torch

dir_path = '/scratch/fml35/datasets/shapenet_v2/ShapeNetRenamed/representation_points_and_normals/exp_04_add_random_points/masks'
save_path = dir_path + '_combined.pt'

assert os.path.exists(save_path) == False

all_files = {}
for file in tqdm(sorted(os.listdir(dir_path)[:100000])):
    if file.endswith(".npz"):
        masks = np.load(os.path.join(dir_path, file))
        all_mask_array = np.zeros((64,600),dtype=bool)
        indices_used = []
        for i,key in enumerate(sorted(masks)):

            elev = int(key.split('_')[1])
            azim = float(key.split('_')[3].replace('.npy',''))
            
            index = int((elev / 15 ) * 16 + int(np.round(azim / 22.5)))
            all_mask_array[index] = masks[key]
            indices_used.append(index)
        assert len(set(indices_used)) == 64
            # all_files[name] = masks[key]
        all_files[file.split('.')[0]] = torch.from_numpy(all_mask_array)

torch.save(all_files, save_path)


# files = torch.load(save_path)
# for file in files:
#     print(file,files[file].shape)
