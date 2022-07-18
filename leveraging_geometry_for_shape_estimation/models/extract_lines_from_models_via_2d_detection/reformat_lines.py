
import os
import shutil
from tqdm import tqdm
import sys
import json


exp_dir = sys.argv[1]
line_exp = sys.argv[2]

which = 'lines_3d_filtered'


with open(exp_dir + '/global_information.json','r') as f:
    global_config = json.load(f)

input_dir = global_config["general"]["models_folder_read"]


dir_1 = input_dir + '/models/extract_from_2d/{}/{}'.format(line_exp,which)
dir_2 = input_dir + '/models/extract_from_2d/{}/{}_reformatted'.format(line_exp,which)

if not os.path.exists(dir_2):
    os.mkdir(dir_2)

for cat in tqdm(sorted(os.listdir(dir_1))):
    for model in sorted(os.listdir(dir_1 + '/' + cat)):
        src = dir_1 + '/' + cat + '/' + model
        target = dir_2 + '/' + cat + '_' + model
        shutil.copy(src,target)
