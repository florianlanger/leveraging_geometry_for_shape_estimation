

import tarfile
import os
target_folder = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/3d_future/data_01'

gt_infos = os.listdir(target_folder + '/gt_infos')

for file in os.listdir(target_folder + '/images'):
    if file.replace('.jpg','.json') not in gt_infos:
        # print(target_folder + '/images/' + file)
        os.remove(target_folder + '/images/' + file)
