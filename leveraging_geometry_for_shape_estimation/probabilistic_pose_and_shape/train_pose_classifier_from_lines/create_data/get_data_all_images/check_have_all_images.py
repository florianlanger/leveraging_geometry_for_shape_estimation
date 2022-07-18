

import json
import os
with open('/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/all_detection_infos_fixed_category.json','r') as f:
    annos = json.load(f)

dir_images = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/images/'

for img in annos:
    path = dir_images + img.replace('.json','.jpg')
    assert os.path.exists(path), path
