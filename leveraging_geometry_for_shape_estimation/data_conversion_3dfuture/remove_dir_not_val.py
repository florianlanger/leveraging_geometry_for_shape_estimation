
from operator import countOf
import os
import json
import shutil
from tqdm import tqdm


path_infos = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/3d_future/data_01/models/model_list.json'
with open(path_infos,'r') as f:
    model_list = json.load(f)['models']
model_list = [item['model'].split('/')[-2] for item in model_list]

dir_path = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/3d_future/data_01/models/extract_from_2d/exp_01_filter_5/lines_2d_vis'

count_keep = 0
for cat in tqdm(sorted(os.listdir(dir_path))):
    for model in tqdm(os.listdir(dir_path + '/' + cat)):
        if model not in model_list:
            # print('remove {}'.format(model))
            shutil.rmtree(dir_path + '/' + cat + '/' + model)
        else:
            # print('keep {}'.format(model))
            count_keep += 1
print(count_keep)