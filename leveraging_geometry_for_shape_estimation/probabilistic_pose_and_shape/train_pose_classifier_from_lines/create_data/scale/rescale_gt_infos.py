
import os
import json
import numpy as np
from tqdm import tqdm

from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.ground_plane import get_model_to_infos_scannet


model_to_infos_scannet = get_model_to_infos_scannet()


path = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_04_small/val/gt_infos_valid_objects_roca_bbox.json'
out_path = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_04_small/val/gt_infos_valid_objects_normalised_scale_roca_bbox.json'
# out_path = path.split('.')[0] + '_normalised_scale.json'
assert os.path.exists(out_path) == False

model_to_infos_scannet = get_model_to_infos_scannet()


with open(path,'r') as f:
    gt_infos = json.load(f)

for img in tqdm(gt_infos):
    for object in gt_infos[img]['objects']:
        model_name = object['model'].split('/')[1] + '_' + object['model'].split('/')[2]
        factor = np.array(model_to_infos_scannet[model_name]['bbox']) * 2 
        object['scaling'] = (object['scaling'] * factor).tolist()

with open(out_path,'w') as f:
    json.dump(gt_infos,f)