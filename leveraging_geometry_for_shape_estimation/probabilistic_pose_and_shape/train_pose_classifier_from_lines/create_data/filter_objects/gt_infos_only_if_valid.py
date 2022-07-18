import os
import json
from tqdm import tqdm

target_dir = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/train/'

target_name = 'gt_infos_center_in_image'

# if not os.path.exists(target_dir + target_name):
os.mkdir(target_dir + target_name)

for file in tqdm(sorted(os.listdir(target_dir + 'gt_infos'))):

    with open(target_dir + 'gt_infos/' + file,'r') as f:
        gt_infos = json.load(f)

    new_gt_infos =  {}
    for key in gt_infos:
        new_gt_infos[key] = gt_infos[key]
    new_gt_infos['objects'] = []
    
    for i in range(len(gt_infos["objects"])):

        # with open(target_dir + 'accept_model_reprojected_depth/' + file.replace('.json','_{}.json'.format(str(i).zfill(2))),'r') as f:
        #     accept_model_reprojected_depth = json.load(f)
        
        with open(target_dir + 'T_in_image/' + file.replace('.json','_{}.json'.format(str(i).zfill(2))),'r') as f:
            T_in_image = json.load(f)

        # if T_in_image["in_image"] == True and accept_model_reprojected_depth['accept'] == True:
        if T_in_image["in_image"] == True:
            object_info = gt_infos["objects"][i]
            object_info['index'] = i
            new_gt_infos['objects'].append(object_info)

    with open(target_dir + target_name + '/' + file,'w') as f:
        json.dump(new_gt_infos, f, indent=4)
        