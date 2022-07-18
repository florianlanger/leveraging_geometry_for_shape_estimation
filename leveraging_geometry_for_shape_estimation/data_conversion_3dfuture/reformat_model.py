import os
import json
import shutil
from tqdm import tqdm

dir_path = '/scratch/fml35/datasets/3d_front/3D-FUTURE-model'

target_path = '/scratch/fml35/datasets/3d_future/3D-FUTURE-model_reformatted/model'

path_infos = '/scratch/fml35/datasets/3d_future/3D-FUTURE-scene/GT/model_infos.json'
with open(path_infos,'r') as f:
    model_infos = json.load(f)

for i in tqdm(range(len(model_infos))):
    info = model_infos[i]
    id = info['model_id']
    category = info["super-category"].replace('/','').lower()
    shutil.copytree(dir_path + '/' + id,target_path + '/' + category + '/' + id)