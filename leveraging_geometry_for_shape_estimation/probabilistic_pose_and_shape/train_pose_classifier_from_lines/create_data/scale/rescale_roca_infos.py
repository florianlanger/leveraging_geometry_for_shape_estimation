import os
import json
import numpy as np
from tqdm import tqdm

from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.ground_plane import get_model_to_infos_scannet



model_to_infos_scannet = get_model_to_infos_scannet()

roca_infos_path = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/val_roca/all_detection_infos_all_images.json'
out_path = roca_infos_path.split('.')[0] + '_normalised_scale.json'

assert os.path.exists(out_path) == False

with open(roca_infos_path,'r') as f:
    roca_infos = json.load(f)

for img in tqdm(roca_infos):
    for object in roca_infos[img]:
        model_name = object['category'] + '_' + object["scene_cad_id"][1]
        factor = np.array(model_to_infos_scannet[model_name]['bbox']) * 2 
        object['s'] = (object['s'] * factor).tolist()

        if object['associated_gt_infos']["matched_to_gt_object"] == True:
            object['associated_gt_infos']["scaling"] = (object['associated_gt_infos']["scaling"] * factor).tolist()
            object['associated_gt_infos']["orig_s"] = (object['associated_gt_infos']["orig_s"] * factor).tolist()

with open(out_path,'w') as f:
    json.dump(roca_infos,f)