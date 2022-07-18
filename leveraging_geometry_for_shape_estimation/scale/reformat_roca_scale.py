import json
from tqdm import tqdm
import os

roca_path = '/scratches/octopus_2/fml35/results/ROCA/per_frame_best_no_null.json'

target_path = '/scratch2/fml35/datasets/own_data/data_leveraging_geometry_for_shape/data_01/'

with open(roca_path,'r') as f:
    roca_data = json.load(f)

counter = 0
for img in tqdm(roca_data):
    img_name_base = img.split('/')[0] + '-' + img.split('/')[2].split('.')[0]
    for i in range(len(roca_data[img])):
        file_name = img_name_base + '_' + str(i).zfill(2) + '.json'

        if not os.path.exists(target_path + 'segmentation_infos/' + file_name):
            counter += 1
            # print(target_path + 'segmentation_infos/' + file_name)
            continue

        # with open(target_path + 'segmentation_infos/' + file_name,'r') as f:
        #     seg_infos = json.load(f)

        # assert seg_infos['predictions']['score'] == roca_data[img][i]['score'],(seg_infos['predictions']['score'],roca_data[img][i]['score'])
        # infos = {'scale': roca_data[img][i]['s']}

        # with open(target_path + 'scale_roca/' + file_name,'w') as f:
        #     json.dump(infos,f)
print(counter)
