

import os
import json
import numpy as np

from tqdm import tqdm

def main():
    dir_in = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_04_small/val/gt_infos_valid_objects'
    dir_in_bbox = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_04_small/val/bboxes_roca'
    dir_out = dir_in + '_roca_bbox'

    os.mkdir(dir_out)

    for file in tqdm(sorted(os.listdir(dir_in))):
        with open(os.path.join(dir_in,file),'r') as f:
            gt_infos = json.load(f)
        
        with open(os.path.join(dir_in_bbox,file),'r') as f:
            bbox_roca = json.load(f)


        for object in gt_infos['objects']:
            index_list_bbox = bbox_roca["indices_orig_objects"].index(object['index'])
            object['bbox'] = np.round(bbox_roca['bboxes'][index_list_bbox]).astype(int).tolist()
            object['use_roca_bbox'] = bbox_roca['use_roca'][index_list_bbox]


        with open(os.path.join(dir_out,file),'w') as f:
            json.dump(gt_infos,f)




if __name__ == '__main__':
    main()