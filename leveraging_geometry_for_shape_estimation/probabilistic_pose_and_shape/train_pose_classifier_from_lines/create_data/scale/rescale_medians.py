
import os
import json
import numpy as np
from tqdm import tqdm

from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.ground_plane import get_model_to_infos_scannet_just_id


model_to_infos_scannet = get_model_to_infos_scannet_just_id()


path = '/scratches/octopus/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/scaling/median_scale_by_model_id_category_median_if_not_in_train.json'

out_path = path.split('.')[0] + '_normalised_scale.json'
assert os.path.exists(out_path) == False


with open(path,'r') as f:
    medians = json.load(f)


for id in medians:
    factor = np.array(model_to_infos_scannet[id]['bbox']) * 2 
    medians[id] = (medians[id] * factor).tolist()

with open(out_path,'w') as f:
    json.dump(medians,f)