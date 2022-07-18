import json
from tqdm import tqdm
import numpy as np
from pycocotools import _mask as coco_mask
import pycocotools
import cv2

def main():
    outdir = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/3d_future/data_01'

    path_future = '/scratch/fml35/datasets/3d_future/3D-FUTURE-scene/GT/test_set.json'
    with open(path_future,'r') as f:
        annos = json.load(f)

    all_new_annos =  {}

    for i in tqdm(range(len(annos['annotations'][:1000000]))):

        image_id = annos['annotations'][i]['image_id'] - 1
        image_name = annos['images'][image_id]['file_name']

        fov = annos['annotations'][i]['fov']

        if fov != None and fov != '':

            if image_id not in all_new_annos:
                all_new_annos[image_id] = 0
            else:
                all_new_annos[image_id] += 1

            rle = annos['annotations'][i]['segmentation']
            
            out_path = outdir + '/masks/' + image_name + '_' +str(all_new_annos[image_id]).zfill(2) + '.png'

            mask = coco_mask.frPyObjects([rle], rle['size'][0], rle['size'][1])
            decoded = coco_mask.decode(mask)
            binaryMask = decoded.astype('bool') * 255
            cv2.imwrite(out_path,binaryMask)
    
    

if __name__ == '__main__':
    main()